import time
from datetime import datetime
import threading
import collections

class TimePeriod:
    WORKTIME = "worktime"
    OVERTIME = "overtime"
    NIGHT = "night"

class PresenceStateMachine:
    """
    文档 4. Presence（人员存在）判定逻辑
    实现分时段窗口策略，防误关灯。
    """
    def __init__(self, camera_id, config):
        self.camera_id = camera_id
        
        # 加载配置 (强制默认值)
        self.window_worktime = config.get("presence_window_default_minutes", 10) * 60
        self.window_overtime = config.get("presence_window_overtime_minutes", 15) * 60
        self.window_night = config.get("presence_window_night_minutes", 5) * 60
        
        # 状态机状态: IDLE, WINDOW_TRACKING, CONFIRM_OCCUPIED, CONFIRM_EMPTY
        self.state = "IDLE"
        
        # 当前窗口信息
        self.window_start_time = 0
        self.window_period_type = None
        self.window_duration = 0
        self.has_person_in_window = False
        
        self.lock = threading.Lock()

    def _get_current_period(self):
        """根据当前系统时间判断所属时段 (简化实现，实际可解析 HH:MM)"""
        hour = datetime.now().hour
        if 9 <= hour < 19:
            return TimePeriod.WORKTIME, self.window_worktime
        elif 19 <= hour < 23:
            return TimePeriod.OVERTIME, self.window_overtime
        else:
            return TimePeriod.NIGHT, self.window_night

    def update(self, has_person_this_frame):
        """
        每次采样 (默认60s) 后调用此方法更新状态机。
        返回 (事件是否触发, 最终状态, 窗口时长分钟数, 所属时段)
        """
        with self.lock:
            now = time.time()
            event_triggered = False
            final_status = None

            # 1. 状态迁移: IDLE -> WINDOW_TRACKING
            if self.state == "IDLE":
                self.state = "WINDOW_TRACKING"
                self.window_start_time = now
                self.has_person_in_window = False
                # 时段边界规则 (强制): 窗口策略按启动时刻固定
                self.window_period_type, self.window_duration = self._get_current_period()
                print(f"[{self.camera_id}] Presence: 开启新窗口 ({self.window_period_type}, {self.window_duration//60}分钟)")

            # 2. 记录当前帧结果
            if has_person_this_frame:
                self.has_person_in_window = True
                
                # 状态迁移: WINDOW_TRACKING -> CONFIRM_OCCUPIED
                # 只要窗口内任一次有人，状态就是 CONFIRM_OCCUPIED
                if self.state == "WINDOW_TRACKING":
                    self.state = "CONFIRM_OCCUPIED"

            # 3. 检查窗口是否结束
            elapsed = now - self.window_start_time
            if elapsed >= self.window_duration:
                # 窗口收敛规则
                if self.has_person_in_window:
                    final_status = "occupied"
                else:
                    final_status = "empty"
                
                event_triggered = True
                
                print(f"[{self.camera_id}] Presence: 窗口结束. 结果={final_status}")
                
                # 重置状态并立即开启下一个窗口，确保连续性
                self.state = "WINDOW_TRACKING"
                self.window_start_time = now
                self.has_person_in_window = False
                self.window_period_type, self.window_duration = self._get_current_period()
                
            return event_triggered, final_status, self.window_duration // 60, self.window_period_type

class SmokingStateMachine:
    """
    Smoking（吸烟）判定逻辑 - 简化版
    不再管理时间窗口，仅负责记录告警状态。
    """
    def __init__(self, camera_id, config):
        self.camera_id = camera_id
        # 状态: IDLE, ALERT
        self.state = "IDLE"
        self.lock = threading.Lock()

    def confirm_smoke(self):
        """当检测到吸烟后调用"""
        with self.lock:
            self.state = "ALERT"
            print(f"[{self.camera_id}] Smoking: 发现吸烟动作，更新状态。")
            return True

    def reset(self):
        """重置状态"""
        with self.lock:
            self.state = "IDLE"

# 测试代码
if __name__ == "__main__":
    config = {}
    sm = PresenceStateMachine("Cam-01", config)
    
    # 模拟 60s 抓拍一次，且一直无人
    for i in range(6):
        print(f"--- 采样 {i+1} ---")
        # 模拟时间流逝 (这里强行修改内部时间加速测试)
        sm.window_start_time -= 60
        evt, status, mins, period = sm.update(has_person_this_frame=False)
        if evt:
            print(f"触发 MQTT 发送: {status}")
