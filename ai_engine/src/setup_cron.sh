#!/bin/bash
# 自动设置 BuildingOS Vision 每日清理与重启定时任务

SCRIPT_PATH="/home/buildingos/buildingos.vision/ai_engine/src/daily_cleanup.py"
LOG_PATH="/home/buildingos/buildingos.vision/ai_engine/cleanup.log"

# 检查脚本是否存在
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "❌ 错误: 找不到脚本 $SCRIPT_PATH"
    exit 1
fi

# 构造 crontab 行 (每天凌晨 2:00 执行，默认处理昨天的数据并重启)
CRON_JOB="0 2 * * * /usr/bin/python3 $SCRIPT_PATH --reboot >> $LOG_PATH 2>&1"

# 检查是否已经存在该任务
(crontab -l 2>/dev/null | grep -F "$SCRIPT_PATH") > /dev/null
if [ $? -eq 0 ]; then
    echo "⚠️ 定时任务已存在，正在更新..."
    (crontab -l 2>/dev/null | grep -vF "$SCRIPT_PATH"; echo "$CRON_JOB") | crontab -
else
    echo "✅ 正在添加新的定时任务..."
    (crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -
fi

echo "🚀 定时任务设置成功！"
echo "📅 执行时间: 每天凌晨 02:00"
echo "📝 日志路径: $LOG_PATH"
echo "🔄 任务内容: 压缩昨日图片 + 生成 AI 日报 + 重启系统"
