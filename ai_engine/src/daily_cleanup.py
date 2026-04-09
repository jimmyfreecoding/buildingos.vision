
import os
import sys
import json
import cv2
import base64
import requests
import time
import argparse
import gc
from datetime import datetime, timedelta

# --- 路径自适应 ---
def get_real_path(p):
    if os.path.exists("/.dockerenv"):
        return p
    
    # 宿主机环境下，尝试找到项目根目录
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
    """转换为 WebP 格式并压缩 (50% 质量，支持 JPG -> WebP)"""
    try:
        if not img_path.lower().endswith(".jpg"): return False

        # 1. 检查文件是否稳定 (5 分钟前)
        file_mtime = os.path.getmtime(img_path)
        if time.time() - file_mtime < 300: 
            return False

        img = cv2.imread(img_path)
        if img is None: return False
        
        h, w = img.shape[:2]
        if w > max_width:
            new_h = int(h * (max_width / w))
            img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        
        # 转换为 WebP
        webp_path = os.path.splitext(img_path)[0] + ".webp"
        
        # 写入 WebP
        success = cv2.imwrite(webp_path, img, [int(cv2.IMWRITE_WEBP_QUALITY), quality])
        
        # 释放 OpenCV 图像对象
        del img
        
        if success and os.path.exists(webp_path):
            os.remove(img_path)
            return True
        return False
    except Exception as e:
        return False

def update_json_references(area_path, to_webp=False):
    """更新 JSON 文件中的图片后缀，支持 jpg <-> webp 互转"""
    updated_count = 0
    from_ext = ".webp" if not to_webp else ".jpg"
    target_ext = ".jpg" if not to_webp else ".webp"
    
    for f in os.listdir(area_path):
        if f.endswith(".json") and f != "daily_summary.json":
            json_path = os.path.join(area_path, f)
            try:
                with open(json_path, 'r', encoding='utf-8') as jf:
                    data = json.load(jf)
                
                # 检查是否包含需要替换的引用
                if "images" in data and any(from_ext in img for img in data["images"]):
                    data["images"] = [img.replace(from_ext, target_ext) for img in data["images"]]
                    with open(json_path, 'w', encoding='utf-8') as jf:
                        json.dump(data, jf, ensure_ascii=False, indent=2)
                    updated_count += 1
            except:
                pass
    return updated_count

def generate_gemma_summary(summary_text_base):
    """调用 Gemma 生成深度日报总结，仅传递预处理后的文本摘要"""
    
    print(f"  📡 正在准备总结请求 (摘要长度: {len(summary_text_base)} 字符)...")

    prompt = f"""
    请根据以下办公区 AI 检测数据的汇总摘要，生成一份专业、详细且排版精美的每日深度分析报告：
    
    【数据汇总摘要】：
    {summary_text_base}
    
    要求：
    1. 必须使用标准 Markdown 格式。
    2. 包含一个总括性的二级标题 (##)。
    3. 针对每个区域，使用三级标题 (###)，并详细润色其“有人/无人”时间段、总时长及检测准确率（一级直认 vs 二级复核）。
    4. 报告应包含对异常点（如频繁切换状态、高频复核区域）的简要推测或建议。
    5. 语言应专业且具有洞察力，直接输出 Markdown 报告内容。
    """
    
    payload = {
        "model": "buildingos_review_engine",
        "messages": [
            {"role": "system", "content": "You are a professional administrative data analyst. Your task is to transform raw statistical summaries into insightful, well-structured Markdown reports for office management."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7,
        "max_tokens": 4096,
        "stream": False
    }
    
    try:
        # 预清理内存
        for i in range(8): requests.delete(f"http://127.0.0.1:8080/slots/{i}", timeout=0.5)
        
        start_time = time.time()
        resp = requests.post(GEMMA_URL, json=payload, timeout=90)
        duration = time.time() - start_time
        
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get('choices', [{}])[0].get('message', {})
            content = msg.get('content', '').strip()
            reasoning = msg.get('reasoning_content', '').strip()
            
            # 合并逻辑：优先取正文，如果正文太短则取思维链
            final_report = content if len(content) > 100 else reasoning
            if not final_report:
                final_report = content or reasoning or "报告内容生成为空"
                
            print(f"  ✅ Gemma 总结生成成功 (耗时: {duration:.1f}s, 报告长度: {len(final_report)} 字符)")
            return final_report
        else:
            print(f"  ❌ Gemma 请求失败 (HTTP {resp.status_code}): {resp.text}")
            return f"Gemma 总结生成失败: API 返回 {resp.status_code}"
    except Exception as e:
        print(f"  ❌ Gemma 请求异常: {e}")
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

def process_day(target_date, only_summary=False, should_reboot=False):
    """处理指定日期的所有日志"""
    day_dir = os.path.join(LOG_DIR_BASE, target_date)
    if not os.path.exists(day_dir):
        print(f"❌ 目录不存在: {day_dir}")
        return

    mode_str = "仅总结模式" if only_summary else "压缩+总结模式"
    print(f"📅 开始处理日期: {target_date} ({mode_str} / WebP 50% 质量)")
    
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
    for area_name in sorted(os.listdir(day_dir)):
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
            "timeline": {} 
        }
        
        # 2. 如果不是仅总结模式，则转换为 WebP
        if not only_summary:
            files = os.listdir(area_path)
            convert_count = 0
            for f in files:
                if f.lower().endswith(".jpg"):
                    if convert_to_webp(os.path.join(area_path, f)):
                        convert_count += 1
            
            # 3. 更新 JSON 引用为 .webp
            if convert_count > 0:
                update_json_references(area_path, to_webp=True)
        
        # 4. 深度解析 JSON
        json_files = [f for f in os.listdir(area_path) if f.endswith(".json") and f != "daily_summary.json"]
        for f in json_files:
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
                            area_stats["lvl2_yes" ] += 1
                            aggregated_data["summary_stats"]["lvl2_gemma_confirmed"] += 1
                            area_stats["lvl2_details"].append({"time": ts, "res": "YES", "reason": chain})
                        elif "Gemma 复核: 否决" in chain_str:
                            area_stats["lvl2_no"] += 1
                            aggregated_data["summary_stats"]["lvl2_gemma_denied"] += 1
                            area_stats["lvl2_details"].append({"time": ts, "res": "NO", "reason": chain})
            except:
                pass
        
        # 5. 计算时间段
        if area_logs_for_timeline:
            area_stats["timeline"] = calculate_time_segments(area_logs_for_timeline)
        
        aggregated_data["areas"][area_name] = area_stats
        
        # 显式清理局部大变量
        del area_logs_for_timeline
        gc.collect()

    # 6. 构造文本摘要用于大模型生成报告
    summary_lines = [
        f"日期: {target_date}",
        f"全天总样本数: {aggregated_data['summary_stats']['total_samples']}",
        f"一级 Detector 直认次数: {aggregated_data['summary_stats']['lvl1_direct_confirm']}",
        f"二级 Gemma 复核总数: {aggregated_data['summary_stats']['lvl2_gemma_reviews']} (确认: {aggregated_data['summary_stats']['lvl2_gemma_confirmed']}, 否决: {aggregated_data['summary_stats']['lvl2_gemma_denied']})",
        ""
    ]
    
    for area_name, area_stats in aggregated_data["areas"].items():
        summary_lines.append(f"【区域: {area_name}】")
        summary_lines.append(f"- 样本总数: {area_stats['samples']}")
        summary_lines.append(f"- 判定分布: 一级确认 {area_stats['lvl1_count']} 次, 二级复核 {area_stats['lvl2_count']} 次 (复核通过 {area_stats['lvl2_yes']} / 拒绝 {area_stats['lvl2_no']})")
        summary_lines.append(f"- 时间统计: 有人总时长 {area_stats['timeline']['total_occupied_min']} 分钟, 无人总时长 {area_stats['timeline']['total_empty_min']} 分钟")
        
        # 只取有人时间段
        occupied_segments = [s for s in area_stats["timeline"]["segments"] if s["state"] == "Occupied"]
        if occupied_segments:
            summary_lines.append("- 有人时段详情:")
            for s in occupied_segments[:20]: # 限制段数，防止超长
                summary_lines.append(f"  * {s['start']} - {s['end']} (时长: {s['duration_min']} 分)")
            if len(occupied_segments) > 20:
                summary_lines.append(f"  * ... (共 {len(occupied_segments)} 个时段)")
        else:
            summary_lines.append("- 该区域今日全天无人。")
        summary_lines.append("")

    summary_text_base = "\n".join(summary_lines)

    # 7. 生成报告
    print("  🧠 正在生成 Gemma 增强深度分析报告...")
    summary_text = generate_gemma_summary(summary_text_base)
    
    report = {
        "version": "3.3",
        "generated_at": datetime.now().isoformat(),
        "stats": aggregated_data,
        "summary": summary_text
    }
    
    with open(os.path.join(day_dir, "daily_summary.json"), 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！报告已保存至: {os.path.join(day_dir, 'daily_summary.json')}")

    # 8. 强力释放 Gemma 内存
    try:
        print("  🧹 正在强力释放 Gemma 插槽内存 (Slots 0-7)...")
        for i in range(8): # 扩大清理范围
            requests.delete(f"http://127.0.0.1:8080/slots/{i}", timeout=1.0)
    except:
        pass
    
    # 9. 脚本最终垃圾回收
    del aggregated_data
    gc.collect()

    # 10. 重启逻辑
    if should_reboot:
        print("🚀 [CRITICAL] 任务全部完成，正在准备系统重启...")
        time.sleep(5)
        os.system("sudo reboot")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BuildingOS Vision 每日数据清理与总结脚本")
    
    # 默认日期改为昨天
    yesterday_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    parser.add_argument("date", nargs="?", default=yesterday_str, help="目标日期 (YYYY-MM-DD)，默认为昨天")
    parser.add_argument("--only-summary", action="store_true", help="仅生成总结报告，跳过图片压缩")
    parser.add_argument("--reboot", action="store_true", help="处理完成后执行 sudo reboot")
    
    args = parser.parse_args()
    process_day(args.date, only_summary=args.only_summary, should_reboot=args.reboot)
