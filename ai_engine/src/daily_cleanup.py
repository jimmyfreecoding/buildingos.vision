
import os
import sys
import json
import cv2
import base64
import requests
import time
from datetime import datetime, timedelta

# --- 路径自适应 ---
def get_real_path(p):
    if os.path.exists("/.dockerenv"):
        return p
    
    # 宿主机环境下，尝试找到项目根目录
    # 假设脚本在 ai_engine/src 下
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    if p.startswith("/app/www"):
        return p.replace("/app/www", os.path.join(project_root, "zlm/www"))
    if p.startswith("/app/ai_engine/config"):
        return p.replace("/app/ai_engine/config", os.path.join(project_root, "ai_engine/config"))
    return p

# --- 配置加载 ---
CONFIG_PATH = get_real_path("/app/ai_engine/config/config.json")
try:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
except:
    config = {}

LOG_DIR_BASE = get_real_path(config.get("storage_quota", {}).get("occupancy_log_dir", "/app/www/occupancy_logs"))
GEMMA_URL = "http://127.0.0.1:8080/v1/chat/completions"
GEMMA_SLOTS_URL = "http://127.0.0.1:8080/slots/0"

def convert_to_webp(img_path, max_width=800, quality=50):
    """转换为 WebP 格式并压缩 (增加安全检查和重复压缩检查)"""
    try:
        # 只处理 .jpg，如果是 .webp 说明已经处理过了
        if not img_path.lower().endswith(".jpg"): return False

        # 1. 检查文件是否处于“稳定”状态 (最后修改时间在 5 分钟前)
        file_mtime = os.path.getmtime(img_path)
        if time.time() - file_mtime < 300: 
            return False

        img = cv2.imread(img_path)
        if img is None: return False
        
        h, w = img.shape[:2]
        # 2. 如果宽度已经小于等于目标宽度，且已经是 webp (虽然逻辑上 jpg 不会进这里)，则不缩放
        if w > max_width:
            new_h = int(h * (max_width / w))
            img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 WebP
        webp_path = os.path.splitext(img_path)[0] + ".webp"
        
        # 检查 webp 是否已存在且较新
        if os.path.exists(webp_path):
            os.remove(img_path) # 如果 webp 已经存在了，直接删除旧 jpg 即可
            return True

        cv2.imwrite(webp_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        
        if os.path.exists(webp_path):
            os.remove(img_path)
            return True
        return False
    except Exception as e:
        return False

def update_json_references(area_path):
    """更新 JSON 文件中的图片后缀，并防止重复修改"""
    updated_count = 0
    for f in os.listdir(area_path):
        if f.endswith(".json") and f != "daily_summary.json":
            json_path = os.path.join(area_path, f)
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                
                # 检查是否包含 .jpg 引用
                if "images" in data and any(".jpg" in img for img in data["images"]):
                    data["images"] = [img.replace(".jpg", ".webp") for img in data["images"]]
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(data, jf, ensure_ascii=False, indent=2)
                    updated_count += 1
            except:
                pass
    return updated_count

def generate_gemma_summary(aggregated_data):
    """调用 Gemma 生成深度日报总结"""
    prompt = f"""
    以下是办公区一天的 AI 检测深度统计数据：
    {json.dumps(aggregated_data, ensure_ascii=False, indent=2)}
    
    请根据以上数据，生成一份每日办公区占用深度分析报告（使用 Markdown 格式）。
    要求必须包含以下内容：
    1. 判定效率分析：提到多少次是一级 Detector 直接确认，多少次触发了二级 Gemma 复核。
    2. 复核准确性：在二级复核中，Gemma 确认了多少次“有人”，否决（排除误报）了多少次。
    3. 区域活跃度详情：
       - 针对每个区域，列出其“有人”的具体时间段（例如：09:15 - 10:30）。
       - 总结该区域的总“有人时间”和“无人时间”。
    4. 整体结论：办公区今天的活跃程度和安全状态总结。
    
    注意：使用专业的 Markdown 标题、列表和加粗。语言专业、客观，不要输出思考过程，直接给出报告。
    """
    
    payload = {
        "model": "buildingos_review_engine",
        "messages": [
            {"role": "system", "content": "You are an AI data analyst. Summarize detection stats with focus on Level 1 vs Level 2 decision counts and detailed occupancy timelines for each area."},
            {"role": "user", "content": prompt}
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.4, # 降低随机性
        "max_tokens": 1500 # 增加长度以容纳时间段
    }
    
    try:
        resp = requests.post(GEMMA_URL, json=payload, timeout=90)
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get('choices', [{}])[0].get('message', {})
            return msg.get('content', '').strip() or msg.get('reasoning_content', '').strip()
        return "Gemma 总结生成失败"
    except Exception as e:
        return f"Gemma 总结生成失败: {e}"

def calculate_time_segments(logs):
    """根据日志计算有人/无人的时间段和总时长"""
    if not logs: return {"segments": [], "total_occupied_min": 0, "total_empty_min": 0}
    
    # 按时间排序
    sorted_logs = sorted(logs, key=lambda x: x['timestamp'])
    
    segments = []
    total_occupied_sec = 0
    total_empty_sec = 0
    
    if not sorted_logs: return {"segments": [], "total_occupied_min": 0, "total_empty_min": 0}
    
    current_state = sorted_logs[0]['is_occupied']
    first_log_time = sorted_logs[0]['timestamp']
    if 'Z' in first_log_time:
        start_time = datetime.fromisoformat(first_log_time.replace('Z', '+00:00'))
    else:
        start_time = datetime.fromisoformat(first_log_time)
    
    for i in range(1, len(sorted_logs)):
        log = sorted_logs[i]
        state = log['is_occupied']
        log_time_str = log['timestamp']
        if 'Z' in log_time_str:
            time = datetime.fromisoformat(log_time_str.replace('Z', '+00:00'))
        else:
            time = datetime.fromisoformat(log_time_str)
        
        if state != current_state:
            # 状态切换，结束当前段
            duration = (time - start_time).total_seconds()
            segments.append({
                "state": "Occupied" if current_state else "Empty",
                "start": start_time.strftime("%H:%M:%S"),
                "end": time.strftime("%H:%M:%S"),
                "duration_min": round(duration / 60, 1)
            })
            if current_state:
                total_occupied_sec += duration
            else:
                total_empty_sec += duration
            
            # 开始新段
            current_state = state
            start_time = time
            
    # 闭合最后一段 (假设到最后一个日志时间)
    last_log_time = sorted_logs[-1]['timestamp']
    # 处理带 Z 或不带 Z 的 ISO 格式
    if 'Z' in last_log_time:
        last_time = datetime.fromisoformat(last_log_time.replace('Z', '+00:00'))
    else:
        last_time = datetime.fromisoformat(last_log_time)
        
    duration = (last_time - start_time).total_seconds()
    if duration >= 0: # 即使是 0 也记录，代表最后的状态点
        segments.append({
            "state": "Occupied" if current_state else "Empty",
            "start": start_time.strftime("%H:%M:%S"),
            "end": last_time.strftime("%H:%M:%S"),
            "duration_min": round(duration / 60, 1)
        })
        if current_state: total_occupied_sec += duration
        else: total_empty_sec += duration

    return {
        "segments": segments,
        "total_occupied_min": round(total_occupied_sec / 60, 1),
        "total_empty_min": round(total_empty_sec / 60, 1)
    }

def process_day(target_date):
    """处理指定日期的所有日志"""
    day_dir = os.path.join(LOG_DIR_BASE, target_date)
    if not os.path.exists(day_dir):
        print(f"❌ 目录不存在: {day_dir}")
        return

    print(f"📅 开始处理日期: {target_date} (WebP 50% 质量 + 增强深度统计)")
    
    aggregated_data = {
        "date": target_date,
        "summary_stats": {
            "total_samples": 0,
            "lvl1_direct_confirm": 0,
            "lvl2_gemma_reviews": 0,
            "lvl2_gemma_confirmed": 0,
            "lvl2_gemma_denied": 0
        },
        "areas": {}
    }

    # 1. 遍历区域目录
    for area_name in os.listdir(day_dir):
        area_path = os.path.join(day_dir, area_name)
        if not os.path.isdir(area_path): continue
        
        print(f"  📂 处理区域: {area_name}")
        area_logs_for_timeline = []
        area_stats = {
            "samples": 0, 
            "lvl1_count": 0, 
            "lvl2_count": 0,
            "lvl2_yes": 0,
            "lvl2_no": 0,
            "lvl2_details": [],
            "timeline": {} # 存放时间段统计
        }
        
        # 2. 转换图片
        files = os.listdir(area_path)
        for f in files:
            if f.endswith(".jpg"):
                convert_to_webp(os.path.join(area_path, f))
        
        # 3. 更新 JSON 引用
        update_json_references(area_path)
        
        # 4. 深度解析 JSON
        for f in os.listdir(area_path):
            if f.endswith(".json") and f != "daily_summary.json":
                try:
                    with open(os.path.join(area_path, f), 'r', encoding='utf-8') as jf:
                        log = json.load(jf)
                        area_stats["samples"] += 1
                        aggregated_data["summary_stats"]["total_samples"] += 1
                        
                        raw = log.get("raw_payload", {})
                        chain = raw.get("decision_chain", [])
                        chain_str = " ".join(chain)
                        
                        # 记录用于时间线计算
                        is_occupied = raw.get("result") == "occupied"
                        area_logs_for_timeline.append({
                            "timestamp": log.get("timestamp"),
                            "is_occupied": is_occupied
                        })
                        
                        # 统计判定层级
                        if "直接确认有人" in chain_str:
                            area_stats["lvl1_count"] += 1
                            aggregated_data["summary_stats"]["lvl1_direct_confirm"] += 1
                        elif "Gemma 二级裁决" in chain_str:
                            area_stats["lvl2_count"] += 1
                            aggregated_data["summary_stats"]["lvl2_gemma_reviews"] += 1
                            
                            ts = log.get("timestamp", "Unknown Time")
                            if "Gemma 复核: 确认" in chain_str:
                                area_stats["lvl2_yes"] += 1
                                aggregated_data["summary_stats"]["lvl2_gemma_confirmed"] += 1
                                area_stats["lvl2_details"].append({"time": ts, "res": "YES", "reason": chain})
                            elif "Gemma 复核: 否决" in chain_str:
                                area_stats["lvl2_no"] += 1
                                aggregated_data["summary_stats"]["lvl2_gemma_denied"] += 1
                                area_stats["lvl2_details"].append({"time": ts, "res": "NO", "reason": chain})
                except:
                    pass
        
        # 5. 计算时间段
        area_stats["timeline"] = calculate_time_segments(area_logs_for_timeline)
        aggregated_data["areas"][area_name] = area_stats

    # 6. 生成报告
    print("  🧠 正在生成 Gemma 增强深度分析报告...")
    summary_text = generate_gemma_summary(aggregated_data)
    
    report = {
        "version": "3.0",
        "generated_at": datetime.now().isoformat(),
        "stats": aggregated_data,
        "summary": summary_text
    }
    
    with open(os.path.join(day_dir, "daily_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！日报内容：\n\n{summary_text}")

    # 7. 主动释放 Gemma 内存
    try:
        print("  🧹 正在释放 Gemma 插槽内存...")
        requests.delete(GEMMA_SLOTS_URL, timeout=5.0)
    except:
        pass

if __name__ == "__main__":
    # 默认处理今天，或者通过参数指定日期 YYYY-MM-DD
    target = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    process_day(target)
