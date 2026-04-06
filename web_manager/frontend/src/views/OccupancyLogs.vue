<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>场景检测综合结果 (Scene Detection Results)</span>
        <el-button @click="fetchLogs" type="primary" plain size="small" :loading="loading">刷新日志</el-button>
      </div>
    </template>

    <div v-loading="loading">
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="filter-section">
            <el-select v-model="selectedArea" placeholder="选择场景 (Area)" style="width: 100%; margin-bottom: 10px;" clearable>
              <el-option v-for="area in uniqueAreas" :key="area" :label="area" :value="area"></el-option>
            </el-select>
            
            <el-select v-model="selectedDate" placeholder="选择日期" style="width: 100%; margin-bottom: 10px;" clearable>
              <el-option v-for="date in uniqueDates" :key="date" :label="date" :value="date"></el-option>
            </el-select>

            <el-select v-model="selectedEvent" placeholder="事件类型" style="width: 100%; margin-bottom: 20px;" clearable>
              <el-option label="人员感知 (Presence)" value="Presence Update"></el-option>
              <el-option label="吸烟告警 (Smoking)" value="Smoking Alert"></el-option>
            </el-select>
          </div>
        </el-col>
        
        <el-col :span="18">
          <div v-if="filteredLogs.length === 0" class="no-data">
            <el-empty description="暂无检测结果"></el-empty>
          </div>
          <el-timeline v-else>
            <el-timeline-item
              v-for="(log, index) in filteredLogs"
              :key="log.id"
              :timestamp="formatTime(log.timestamp)"
              :type="getEventTypeColor(log.event)"
              placement="top"
            >
              <el-card shadow="hover">
                <h4>
                  <el-icon v-if="log.event === 'Smoking Alert'"><Warning /></el-icon>
                  <el-icon v-else><UserFilled /></el-icon>
                  {{ log.event === 'Smoking Alert' ? '吸烟违规告警' : '人员状态更新' }}
                </h4>
                <p><strong>场景区域 (Area):</strong> {{ log.areaCode }}</p>
                <p>
                  <strong>检测结果:</strong> 
                  <el-tag :type="getResultTagType(log)" size="small" effect="dark">
                    {{ formatResult(log) }}
                  </el-tag>
                </p>
                
                <el-collapse style="margin-top: 10px;">
                  <el-collapse-item title="双轨制决策详情" name="1">
                    <div>
                      <p><strong>数据来源:</strong> <el-tag size="small" type="info">{{ log.raw_payload?.source || 'yolo26m+gemma' }}</el-tag></p>
                      <p v-if="log.event === 'Presence Update'">
                        <strong>时段策略:</strong> {{ formatTimePeriod(log.raw_payload?.timePeriod) }} 
                        <span style="color: #909399; font-size: 12px;">(连续判断窗口: {{ log.raw_payload?.windowMinutes }} 分钟)</span>
                      </p>
                      <p v-if="log.event === 'Smoking Alert'">
                        <strong>抓拍间隔:</strong> {{ log.raw_payload?.sampleIntervalSeconds }} 秒
                        <span style="color: #909399; font-size: 12px;">(分析窗口: {{ log.raw_payload?.windowMinutes }} 分钟)</span>
                      </p>
                      <p style="font-weight: bold; color: #409EFF;">最终裁决: {{ log.threshold_used }}</p>
                    </div>
                  </el-collapse-item>
                </el-collapse>

                <div v-if="log.images && log.images.length > 0" class="image-gallery" style="margin-top: 15px;">
                  <p><strong>现场抓拍与复核截图:</strong></p>
                  <el-row :gutter="10">
                    <el-col :span="12" v-for="(img, imgIdx) in log.images" :key="imgIdx">
                      <el-image 
                        :src="getImageUrl(img)" 
                        :preview-src-list="[getImageUrl(img)]"
                        fit="contain"
                        class="log-image"
                        :initial-index="0"
                      />
                    </el-col>
                  </el-row>
                </div>
              </el-card>
            </el-timeline-item>
          </el-timeline>
        </el-col>
      </el-row>
    </div>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { UserFilled, Warning } from '@element-plus/icons-vue'

const loading = ref(false)
const logs = ref([])
const selectedArea = ref('')
const selectedDate = ref('')
const selectedEvent = ref('')

const fetchLogs = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/occupancy/logs')
    logs.value = res.data || []
  } catch (e) {
    ElMessage.error('获取日志失败')
  }
  loading.value = false
}

const uniqueAreas = computed(() => {
  const areas = new Set(logs.value.map(l => l.areaCode))
  return Array.from(areas)
})

const uniqueDates = computed(() => {
  const dates = new Set(logs.value.map(l => l.date))
  return Array.from(dates).sort((a, b) => b.localeCompare(a))
})

const filteredLogs = computed(() => {
  return logs.value.filter(log => {
    const matchArea = selectedArea.value ? log.areaCode === selectedArea.value : true
    const matchDate = selectedDate.value ? log.date === selectedDate.value : true
    const matchEvent = selectedEvent.value ? log.event === selectedEvent.value : true
    return matchArea && matchDate && matchEvent
  })
})

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleString()
}

const getEventTypeColor = (event) => {
  if (event === 'Smoking Alert') return 'danger'
  return 'primary'
}

const getResultTagType = (log) => {
  if (log.event === 'Smoking Alert') return 'danger'
  if (log.raw_payload?.result === 'occupied') return 'warning'
  return 'info'
}

const formatResult = (log) => {
  if (log.event === 'Smoking Alert') return '确认吸烟 (Confirmed)'
  if (log.raw_payload?.result === 'occupied') return '区域有人 (Occupied)'
  if (log.raw_payload?.result === 'empty') return '区域无人 (Empty)'
  return '未知 (Unknown)'
}

const formatTimePeriod = (period) => {
  if (period === 'worktime') return '工作时段 (防误关灯)'
  if (period === 'overtime') return '加班时段 (防漏检)'
  if (period === 'night') return '深夜时段 (节能优先)'
  return period || '默认'
}

const getImageUrl = (relativePath) => {
  // Use current host and ZLM static port
  return `http://${window.location.hostname}:10081/${relativePath}`
}

onMounted(() => {
  fetchLogs()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.log-image {
  width: 100%;
  height: 240px;
  border-radius: 4px;
  border: 1px solid #EBEEF5;
  background-color: #000;
}
.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}
</style>