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
          <!-- Y-axis (Minutes) -->
          <div class="y-axis">
            <div class="y-label" v-for="m in 6" :key="m">{{ (m - 1) * 10 }}m</div>
          </div>
          
          <!-- Grid -->
          <div class="grid-content">
             <div class="grid-columns">
                <div class="column" v-for="hour in 24" :key="hour">
                   <el-tooltip
                      v-for="minuteIdx in 6" :key="minuteIdx"
                      placement="top"
                      :content="getTooltip(dayData, hour-1, minuteIdx-1)"
                      :show-after="200"
                   >
                     <div 
                        class="cell" 
                        :class="'color-level-' + getCellIntensity(dayData, hour-1, minuteIdx-1)"
                        @click="openDetail(dayData, hour-1, minuteIdx-1)"
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
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="800px" top="5vh">
      <div v-if="dialogLogs.length === 0">
        <el-empty description="该时间段无记录"></el-empty>
      </div>
      <el-timeline v-else>
        <el-timeline-item
          v-for="(log, index) in dialogLogs"
          :key="log.id"
          :timestamp="formatTime(log.timestamp)"
          :type="getEventTypeColor(log.event)"
        >
          <el-card shadow="hover" class="log-detail-card">
            <div class="log-header">
              <el-tag :type="getResultTagType(log)" size="small" effect="dark">
                {{ formatResult(log) }}
              </el-tag>
              <span class="log-event">{{ log.event === 'Smoking Alert' ? '吸烟违规告警' : '人员状态更新' }}</span>
            </div>
            
            <div class="log-meta" style="margin-top: 10px; font-size: 13px; color: #8b949e;">
              <p><strong>数据来源:</strong> {{ log.raw_payload?.source || 'yolo26m+gemma' }}</p>
              <p><strong>最终裁决:</strong> <span style="color: #409EFF">{{ log.threshold_used }}</span></p>
            </div>

            <div v-if="log.images && log.images.length > 0" class="image-gallery" style="margin-top: 15px;">
              <p><strong>现场抓拍与复核截图:</strong></p>
              <el-row :gutter="10">
                <el-col :span="12" v-for="(img, imgIdx) in log.images" :key="imgIdx">
                  <el-image 
                    :src="getImageUrl(img)" 
                    :preview-src-list="log.images.map(i => getImageUrl(i))"
                    fit="contain"
                    class="log-image"
                    :initial-index="imgIdx"
                  />
                </el-col>
              </el-row>
            </div>
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
import { Calendar } from '@element-plus/icons-vue'

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
    // 匹配特定日期的日志
    const dayLogs = areaLogs.filter(log => log.date === dateStr)
    
    // 如果查询的是特定的日期段，或者有数据，或者是默认的最近8天，都展示该日期的图
    if (dayLogs.length > 0 || datesToDisplay.length <= 8) {
      // 初始化 24(列) x 6(行) 的网格，代表 24小时，每小时6个10分钟窗口
      const grid = Array.from({ length: 24 }, () => Array.from({ length: 6 }, () => []))
      
      dayLogs.forEach(log => {
        if (!log.timestamp) return
        const d = new Date(log.timestamp)
        const hour = d.getHours()
        const min = d.getMinutes()
        const minIdx = Math.floor(min / 10)
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
  background-color: #0d1117;
  color: #c9d1d9;
  border: 1px solid #30363d;
  min-height: 800px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  color: #c9d1d9;
  font-weight: bold;
}
.filter-section {
  display: flex;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #30363d;
}
.legend {
  display: flex;
  align-items: center;
  margin-left: auto;
  font-size: 12px;
  color: #8b949e;
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

/* GitHub Dark Theme Heatmap Colors */
.color-level-0 { background-color: #161b22; outline: 1px solid rgba(255, 255, 255, 0.05); outline-offset: -1px; }
.color-level-1 { background-color: #0e4429; }
.color-level-2 { background-color: #006d32; }
.color-level-3 { background-color: #26a641; }
.color-level-4 { background-color: #39d353; }

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
  color: #c9d1d9;
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
  height: 114px; /* 6 * 15px + 5 gaps (4px) = 90 + 20 = 110px. approx 114px */
  margin-right: 15px;
  padding-bottom: 20px; /* Offset for X-axis */
}
.y-label {
  font-size: 12px;
  color: #8b949e;
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
  box-shadow: 0 0 6px rgba(0,0,0,0.8);
}
.x-axis {
  display: flex;
  gap: 4px;
  margin-top: 10px;
}
.x-label {
  width: 15px;
  font-size: 11px;
  color: #8b949e;
  text-align: left;
  overflow: visible;
  white-space: nowrap;
}
/* Hide odd hour labels for cleaner look to prevent overlapping */
.x-label:nth-child(odd) {
  opacity: 0;
}

/* Dialog inner styles */
.log-detail-card {
  margin-bottom: 5px;
  background-color: #161b22;
  color: #c9d1d9;
  border: 1px solid #30363d;
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
  background-color: #000;
  border: 1px solid #30363d;
}
:deep(.el-timeline-item__timestamp) {
  color: #8b949e;
}
</style>
