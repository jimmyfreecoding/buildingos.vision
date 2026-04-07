<template>
  <el-card class="box-card heatmap-card">
    <template #header>
      <div class="card-header">
        <span><el-icon><Calendar /></el-icon> 场景状态热力图 (Occupancy Heatmap)</span>
        <el-button @click="fetchLogs" type="primary" plain size="small" :loading="loading">刷新数据</el-button>
      </div>
    </template>

    <div class="filter-section">
      <el-select v-model="selectedArea" placeholder="选择场景 (Area)" style="width: 250px; margin-right: 15px;" @change="handleFilterChange" filterable>
        <el-option v-for="area in uniqueAreas" :key="area" :label="area" :value="area"></el-option>
      </el-select>
      
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        range-separator="至"
        start-placeholder="开始日期"
        end-placeholder="结束日期"
        value-format="YYYY-MM-DD"
        @change="handleFilterChange"
        style="width: 300px; margin-right: 15px;"
      />
      <div class="legend">
        <span>无人</span>
        <ul class="legend-colors">
          <li class="color-level-0"></li>
          <li class="color-level-1"></li>
          <li class="color-level-2"></li>
          <li class="color-level-3"></li>
          <li class="color-level-4"></li>
        </ul>
        <span>有人(多)</span>
      </div>
    </div>

    <div v-loading="loading" class="heatmaps-wrapper">
      <div v-if="!selectedArea" class="no-data">
        <el-empty description="请先选择场景 (Area)"></el-empty>
      </div>
      <div v-else-if="displayDays.length === 0" class="no-data">
        <el-empty description="该时间段内暂无检测数据"></el-empty>
      </div>
      
      <div v-else v-for="dayData in displayDays" :key="dayData.date" class="day-heatmap">
        <h4 class="day-title">{{ dayData.date }}</h4>
        <div class="heatmap-container">
          <!-- Y-axis (Minutes): 50m at top, 0m at bottom -->
          <div class="y-axis">
            <div class="y-label" v-for="m in 6" :key="m">{{ (6 - m) * 10 }}m</div>
          </div>
          
          <!-- Grid -->
          <div class="grid-content">
             <div class="grid-columns">
                <div class="column" v-for="hour in 24" :key="hour">
                   <!-- 反转 minuteIdx 渲染顺序，使 DOM 节点也自下而上排列 0m, 10m...50m -->
                   <el-tooltip
                      v-for="minuteIdx in 6" :key="minuteIdx"
                      placement="top"
                      :content="getTooltip(dayData, hour-1, 6 - minuteIdx)"
                      :show-after="200"
                   >
                     <div 
                        class="cell" 
                        :class="'color-level-' + getCellIntensity(dayData, hour-1, 6 - minuteIdx)"
                        @click="openDetail(dayData, hour-1, 6 - minuteIdx)"
                     ></div>
                   </el-tooltip>
                </div>
             </div>
             <!-- X-axis (Hours) -->
             <div class="x-axis">
               <div class="x-label" v-for="hour in 24" :key="hour">{{ (hour - 1).toString().padStart(2, '0') }}:00</div>
             </div>
          </div>
        </div>
      </div>
    </div>

    <!-- 详情弹窗 -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="1000px" top="5vh">
      <div v-if="dialogLogs.length === 0">
        <el-empty description="该时间段无记录"></el-empty>
      </div>
      <el-timeline v-else>
        <el-timeline-item
          v-for="group in groupedLogs"
          :key="group.time"
          :timestamp="group.time"
          type="primary"
        >
          <el-card shadow="hover" class="log-detail-card">
            <div class="log-header">
              <el-tag :type="getGroupResultTagType(group)" size="small" effect="dark">
                {{ formatGroupResult(group) }}
              </el-tag>
              <span class="log-event">人员状态更新 (多路综合)</span>
            </div>
            
            <div class="log-meta" style="margin-top: 10px; font-size: 13px; color: #606266;">
              <p><strong>策略链路:</strong> yolo26m+gemma</p>
              <p style="margin-top: 5px;">
                <strong>最终裁决:</strong> 
                <el-popover placement="bottom" title="1-minute sample 裁决过程" width="400" trigger="click">
                  <template #reference>
                    <el-link type="primary" :underline="false">1-minute sample (点击查看)</el-link>
                  </template>
                  <div style="font-size: 13px;">
                    <p v-for="log in group.logs" :key="log.id" style="margin-bottom: 5px;">
                      <b><el-icon><VideoCamera /></el-icon> {{ log.camera_id }}:</b> 
                      <span v-if="log.raw_payload?.result === 'occupied'" style="color: #67C23A; margin-left: 5px;">判定有人</span>
                      <span v-else style="color: #909399; margin-left: 5px;">判定无人</span>
                      <span style="margin-left: 5px; color: #E6A23C;" v-if="log.raw_payload?.yolo_count > 0">(YOLO检测到 {{ log.raw_payload?.yolo_count }} 人)</span>
                    </p>
                    <el-divider style="margin: 10px 0;"></el-divider>
                    <p><b>场景综合结果:</b> 
                      <span v-if="group.logs.some(l => l.raw_payload?.result === 'occupied')" style="color: #67C23A; font-weight: bold;">有人 (Occupied)</span>
                      <span v-else style="color: #909399; font-weight: bold;">无人 (Empty)</span>
                    </p>
                  </div>
                </el-popover>
              </p>
            </div>

            <div style="margin-top: 15px; font-size: 13px; font-weight: bold; color: #303133; margin-bottom: 10px;">
              现场抓拍与复核证据 (同频对比):
            </div>
            
            <!-- 多摄像头左右排列 -->
            <el-row :gutter="15">
              <el-col :span="24 / Math.min(group.logs.length, 4)" v-for="log in group.logs" :key="log.id">
                <div class="camera-evidence">
                  <p style="font-weight: bold; margin-bottom: 5px; color: #409EFF; font-size: 13px;">
                    <el-icon><VideoCamera /></el-icon> {{ log.camera_id }}
                  </p>
                  <el-image 
                    v-if="log.images && log.images.length > 0"
                    :src="getImageUrl(log.images[log.images.length - 1])" 
                    :preview-src-list="log.images.map(i => getImageUrl(i))"
                    fit="contain"
                    class="log-image"
                  />
                  <div class="evidence-chain" style="margin-top: 10px; font-size: 12px; color: #606266; background: #f5f7fa; padding: 8px; border-radius: 4px; min-height: 80px;">
                    <b style="color: #303133;">判断证据链:</b>
                    <ul style="padding-left: 20px; margin-top: 5px; margin-bottom: 0;">
                      <li v-for="(step, idx) in (log.raw_payload?.decision_chain || ['直接采信无日志'])" :key="idx" style="margin-bottom: 3px;">
                        {{ step }}
                      </li>
                    </ul>
                  </div>
                </div>
              </el-col>
            </el-row>
          </el-card>
        </el-timeline-item>
      </el-timeline>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { Calendar, VideoCamera } from '@element-plus/icons-vue'

const loading = ref(false)
const allLogs = ref([])
const selectedArea = ref('')
const dateRange = ref([])

const dialogVisible = ref(false)
const dialogTitle = ref('')
const dialogLogs = ref([])

// 默认显示最近8天（包含今天）
const getDefaultDateRange = () => {
  const dates = []
  for (let i = 0; i < 8; i++) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    // 补齐两位数，确保与后端的 date 文件夹名称对齐
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    dates.push(`${year}-${month}-${day}`)
  }
  return dates
}

const defaultDates = getDefaultDateRange()

const fetchLogs = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/occupancy/logs')
    allLogs.value = res.data || []
    
    if (!selectedArea.value && uniqueAreas.value.length > 0) {
      selectedArea.value = uniqueAreas.value[0]
    }
  } catch (e) {
    ElMessage.error('获取日志失败')
  }
  loading.value = false
}

const uniqueAreas = computed(() => {
  const areas = new Set(allLogs.value.map(l => l.areaCode))
  return Array.from(areas)
})

const handleFilterChange = () => {
  // Computed property will auto update
}

// 核心：按选定日期和场景，生成二维热力图数据
const displayDays = computed(() => {
  if (!selectedArea.value) return []

  let datesToDisplay = []
  if (dateRange.value && dateRange.value.length === 2) {
    const start = new Date(dateRange.value[0])
    const end = new Date(dateRange.value[1])
    for (let d = new Date(end); d >= start; d.setDate(d.getDate() - 1)) {
      const year = d.getFullYear()
      const month = String(d.getMonth() + 1).padStart(2, '0')
      const day = String(d.getDate()).padStart(2, '0')
      datesToDisplay.push(`${year}-${month}-${day}`)
    }
  } else {
    datesToDisplay = defaultDates
  }

  const result = []
  const areaLogs = allLogs.value.filter(log => log.areaCode === selectedArea.value)

  datesToDisplay.forEach(dateStr => {
    // 根据 dateStr 获取该日期的边界，过滤掉未来的时间点
    const todayStr = new Date().toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\//g, '-')
    const isToday = dateStr === todayStr
    const currentHour = new Date().getHours()
    const currentMin = new Date().getMinutes()

    // 匹配特定日期的日志 (使用 log.date 而非 log.timestamp 解析，避免跨时区导致日期漂移)
    const dayLogs = areaLogs.filter(log => log.date === dateStr)
    
    if (dayLogs.length > 0 || datesToDisplay.length <= 8) {
      const grid = Array.from({ length: 24 }, () => Array.from({ length: 6 }, () => []))
      
      dayLogs.forEach(log => {
        if (!log.timestamp) return
        
        // 修复：解析ISO字符串时，确保时区一致。如果是本地记录的timestamp且没有带Z，
        // 解析出来可能会因为浏览器本地时区有偏差。
        // 为了安全，如果timestamp中没有T或者Z，可以假设它是本地时间
        let d = new Date(log.timestamp)
        // 如果后端返回的是 UTC (带 Z) 但你希望按照设备本地时间显示，需要确保时区
        
        const hour = d.getHours()
        const min = d.getMinutes()
        const minIdx = Math.floor(min / 10)
        
        // 过滤掉超过当前时间点的未来数据 (可能是时区偏差导致的数据漂移)
        if (isToday) {
            if (hour > currentHour || (hour === currentHour && min > currentMin)) {
                return // 跳过未来的记录
            }
        }

        if (hour >= 0 && hour < 24 && minIdx >= 0 && minIdx < 6) {
          grid[hour][minIdx].push(log)
        }
      })
      
      result.push({
        date: dateStr,
        grid: grid
      })
    }
  })
  
  return result
})

const getCellLogs = (dayData, hour, minuteIdx) => {
  return dayData.grid[hour][minuteIdx] || []
}

const getCellIntensity = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  if (logs.length === 0) return 0
  
  // 基于有人记录的数量决定绿色深浅
  const occupiedLogs = logs.filter(l => l.raw_payload?.result === 'occupied')
  if (occupiedLogs.length === 0) {
    // 只有“无人”记录，显示浅色，或者也可以定为另一种颜色。GitHub Heatmap中我们用最低亮度的绿色表示少量活动
    return 1 
  }
  
  const count = occupiedLogs.length
  if (count === 1) return 2
  if (count <= 3) return 3
  return 4
}

const getTooltip = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  const timeStr = `${hour.toString().padStart(2, '0')}:${(minuteIdx * 10).toString().padStart(2, '0')} - ${hour.toString().padStart(2, '0')}:${(minuteIdx * 10 + 9).toString().padStart(2, '0')}`
  if (logs.length === 0) return `${timeStr} (无检测记录)`
  
  const occupiedLogs = logs.filter(l => l.raw_payload?.result === 'occupied')
  return `${timeStr} | 有人: ${occupiedLogs.length}次, 总记录: ${logs.length}次`
}

// 提取并聚合多摄像头的按分钟日志
const groupedLogs = computed(() => {
  const groups = {}
  dialogLogs.value.forEach(log => {
    if (!log.timestamp) return
    const d = new Date(log.timestamp)
    const minKey = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    if (!groups[minKey]) groups[minKey] = []
    groups[minKey].push(log)
  })
  
  // 按时间倒序
  return Object.keys(groups).sort((a, b) => new Date(b) - new Date(a)).map(k => ({
    time: k,
    logs: groups[k]
  }))
})

const getGroupResultTagType = (group) => {
  const isOccupied = group.logs.some(l => l.raw_payload?.result === 'occupied')
  return isOccupied ? 'success' : 'info'
}

const formatGroupResult = (group) => {
  const isOccupied = group.logs.some(l => l.raw_payload?.result === 'occupied')
  return isOccupied ? '区域有人 (Occupied)' : '区域无人 (Empty)'
}

const openDetail = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  if (logs.length === 0) return
  
  const timeStr = `${hour.toString().padStart(2, '0')}:${(minuteIdx * 10).toString().padStart(2, '0')} - ${hour.toString().padStart(2, '0')}:${(minuteIdx * 10 + 9).toString().padStart(2, '0')}`
  dialogTitle.value = `[${selectedArea.value}] ${dayData.date} ${timeStr} 详情记录`
  dialogLogs.value = [...logs].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
  dialogVisible.value = true
}

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleTimeString()
}

const getEventTypeColor = (event) => {
  if (event === 'Smoking Alert') return 'danger'
  return 'primary'
}

const getResultTagType = (log) => {
  if (log.event === 'Smoking Alert') return 'danger'
  if (log.raw_payload?.result === 'occupied') return 'success'
  return 'info'
}

const formatResult = (log) => {
  if (log.event === 'Smoking Alert') return '确认吸烟 (Confirmed)'
  if (log.raw_payload?.result === 'occupied') return '区域有人 (Occupied)'
  if (log.raw_payload?.result === 'empty') return '区域无人 (Empty)'
  return '未知 (Unknown)'
}

const getImageUrl = (relativePath) => {
  return `http://${window.location.hostname}:10081/${relativePath}`
}

onMounted(() => {
  fetchLogs()
})
</script>

<style scoped>
.heatmap-card {
  min-height: 800px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}
.filter-section {
  display: flex;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #ebeef5;
}
.legend {
  display: flex;
  align-items: center;
  margin-left: auto;
  font-size: 12px;
  color: #606266;
}
.legend-colors {
  display: flex;
  list-style: none;
  padding: 0;
  margin: 0 8px;
  gap: 4px;
}
.legend-colors li {
  width: 14px;
  height: 14px;
  border-radius: 3px;
}

/* GitHub Light Theme Heatmap Colors */
.color-level-0 { background-color: #ebedf0; }
.color-level-1 { background-color: #9be9a8; }
.color-level-2 { background-color: #40c463; }
.color-level-3 { background-color: #30a14e; }
.color-level-4 { background-color: #216e39; }

.heatmaps-wrapper {
  display: flex;
  flex-direction: column;
  gap: 40px;
}
.day-heatmap {
  display: flex;
  flex-direction: column;
}
.day-title {
  margin: 0 0 10px 45px;
  font-size: 14px;
  color: #303133;
  font-weight: 500;
}
.heatmap-container {
  display: flex;
  align-items: flex-start;
}
.y-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-around;
  height: 114px;
  margin-right: 15px;
  padding-bottom: 20px;
}
.y-label {
  font-size: 12px;
  color: #909399;
  line-height: 15px;
  text-align: right;
  width: 30px;
}
.grid-content {
  display: flex;
  flex-direction: column;
}
.grid-columns {
  display: flex;
  gap: 4px;
}
.column {
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.cell {
  width: 15px;
  height: 15px;
  border-radius: 3px;
  cursor: pointer;
  transition: transform 0.1s, box-shadow 0.1s;
}
.cell:hover {
  transform: scale(1.3);
  z-index: 10;
  box-shadow: 0 0 6px rgba(0,0,0,0.2);
}
.x-axis {
  display: flex;
  gap: 4px;
  margin-top: 10px;
}
.x-label {
  width: 15px;
  font-size: 11px;
  color: #909399;
  text-align: left;
  overflow: visible;
  white-space: nowrap;
}
.x-label:nth-child(odd) {
  opacity: 0;
}

/* Dialog inner styles */
.log-detail-card {
  margin-bottom: 5px;
}
.log-header {
  display: flex;
  align-items: center;
  gap: 10px;
}
.log-event {
  font-weight: bold;
}
.log-image {
  width: 100%;
  height: 200px;
  border-radius: 4px;
  background-color: #f5f7fa;
  border: 1px solid #ebeef5;
}
</style>
