
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
    home = os.path.expanduser("~")
    project_root = os.path.join(home, "buildingos.vision")
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

def compress_image(img_path, max_width=800, quality=70):
    """压缩图片：缩放 + 降低质量 (增加安全检查)"""
    try:
        # 1. 检查文件是否处于“稳定”状态 (最后修改时间在 5 分钟前)
        file_mtime = os.path.getmtime(img_path)
        if time.time() - file_mtime < 300: # 300秒 = 5分钟
            return False

        img = cv2.imread(img_path)
        if img is None: return False
        
        h, w = img.shape[:2]
        # 2. 如果已经压缩过了，跳过
        if w <= max_width:
            return True
        
        new_h = int(h * (max_width / w))
        img = cv2.resize(img, (max_width, new_h), interpolation=cv2.INTER_AREA)
        
        # 覆盖写入
        cv2.imwrite(img_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        return True
    except Exception as e:
        # 忽略可能的“文件被占用”错误
        return False

def generate_gemma_summary(aggregated_data):
    """调用 Gemma 生成日报总结"""
    prompt = f"""
    以下是办公区一天的 AI 检测统计数据：
    {json.dumps(aggregated_data, ensure_ascii=False, indent=2)}
    
    请根据以上数据，生成一份简短的每日办公区占用日报（中文）。
    要求：
    1. 总结整体占用情况（繁忙时段、平均人数）。
    2. 提及任何异常事件（如吸烟告警、长时间无人或异常拥挤）。
    3. 语言自然、专业，不要输出思考过程，直接给出报告内容。
    """
    
    payload = {
        "model": "buildingos_review_engine",
        "messages": [
            {"role": "system", "content": "You are a professional administrative assistant summarizing office occupancy data."},
            {"role": "user", "content": prompt}
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
        "max_tokens": 512
    }
    
    try:
        resp = requests.post(GEMMA_URL, json=payload, timeout=60)
        if resp.status_code == 200:
            data = resp.json()
            msg = data.get('choices', [{}])[0].get('message', {})
            content = msg.get('content', '').strip()
            reasoning = msg.get('reasoning_content', '').strip()
            return content if content else reasoning
        return "Gemma 总结生成失败 (API 错误)"
    except Exception as e:
        return f"Gemma 总结生成失败: {e}"

def process_day(target_date):
    """处理指定日期的所有日志"""
    day_dir = os.path.join(LOG_DIR_BASE, target_date)
    if not os.path.exists(day_dir):
        print(f"❌ 目录不存在: {day_dir}")
        return

    print(f"📅 开始处理日期: {target_date}")
    
    aggregated_data = {
        "date": target_date,
        "total_samples": 0,
        "occupied_samples": 0,
        "areas": {}
    }

    # 1. 遍历区域目录
    for area_name in os.listdir(day_dir):
        area_path = os.path.join(day_dir, area_name)
        if not os.path.isdir(area_path): continue
        
        print(f"  📂 处理区域: {area_name}")
        area_stats = {"samples": 0, "occupied": 0, "events": []}
        
        # 2. 遍历文件
        files = os.listdir(area_path)
        for f in files:
            f_path = os.path.join(area_path, f)
            
            # 处理图片
            if f.endswith(".jpg"):
                compress_image(f_path)
            
            # 处理 JSON
            elif f.endswith(".json") and f != "daily_summary.json":
                try:
                    with open(f_path, 'r', encoding='utf-8') as jf:
                        log = json.load(jf)
                        area_stats["samples"] += 1
                        aggregated_data["total_samples"] += 1
                        
                        is_occupied = False
                        if "is_occupied" in log:
                            is_occupied = log["is_occupied"]
                        elif "raw_payload" in log:
                            is_occupied = log["raw_payload"].get("result") == "occupied"
                        
                        if is_occupied:
                            area_stats["occupied"] += 1
                            aggregated_data["occupied_samples"] += 1
                        
                        # 记录异常 (如吸烟)
                        if log.get("event") == "Smoking Alert":
                            area_stats["events"].append({
                                "time": log.get("timestamp"),
                                "type": "Smoking"
                            })
                except:
                    pass
        
        aggregated_data["areas"][area_name] = area_stats

    # 3. 生成 Gemma 总结
    print("  🧠 正在生成 Gemma 日报总结...")
    summary_text = generate_gemma_summary(aggregated_data)
    
    # 4. 保存总结
    report = {
        "generated_at": datetime.now().isoformat(),
        "stats": aggregated_data,
        "summary": summary_text
    }
    
    summary_path = os.path.join(day_dir, "daily_summary.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"✅ 处理完成！报告已保存至: {summary_path}")
    print("-" * 30)
    print(f"日报摘要:\n{summary_text}")
    print("-" * 30)

if __name__ == "__main__":
    # 默认处理今天，或者通过参数指定日期 YYYY-MM-DD
    target = sys.argv[1] if len(sys.argv) > 1 else datetime.now().strftime("%Y-%m-%d")
    process_day(target)
