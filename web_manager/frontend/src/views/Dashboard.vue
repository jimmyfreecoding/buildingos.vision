<template>
  <div class="dashboard-container">
    <el-row :gutter="20">
      <!-- System Info Card -->
      <el-col :span="12">
        <el-card class="box-card system-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><Monitor /></el-icon> 边缘计算节点状态 (Orin Nano)</span>
            </div>
          </template>
          <div v-loading="loadingSys" class="sys-content">
            <el-row :gutter="20" class="sys-metrics">
              <el-col :span="8" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.cpu.usage" :color="customColors">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(1) }}%</span>
                    <span class="percentage-label">CPU 负载</span>
                  </template>
                </el-progress>
                <div class="metric-desc">{{ sysInfo.cpu.cores }} Cores</div>
              </el-col>
              <el-col :span="8" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.memory.usagePercent" :color="customColors">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(1) }}%</span>
                    <span class="percentage-label">内存使用</span>
                  </template>
                </el-progress>
                <div class="metric-desc">{{ formatBytes(sysInfo.memory.used) }} / {{ formatBytes(sysInfo.memory.total) }}</div>
              </el-col>
              <el-col :span="8" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.gpu.util" :color="customColors">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(1) }}%</span>
                    <span class="percentage-label">GPU (TensorRT)</span>
                  </template>
                </el-progress>
                <div class="metric-desc">Mem: {{ sysInfo.gpu.memUsed }}MB / {{ sysInfo.gpu.memTotal }}MB</div>
              </el-col>
            </el-row>
            <el-divider />
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item label="OS Platform">{{ sysInfo.os.platform }} {{ sysInfo.os.release }}</el-descriptions-item>
              <el-descriptions-item label="Uptime">{{ formatUptime(sysInfo.os.uptime) }}</el-descriptions-item>
              <el-descriptions-item label="CPU Model">{{ sysInfo.cpu.model }}</el-descriptions-item>
            </el-descriptions>
          </div>
        </el-card>
      </el-col>

      <!-- ZLM Media Server Status Card -->
      <el-col :span="12">
        <el-card class="box-card zlm-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><VideoCamera /></el-icon> 流媒体引擎状态 (ZLMediaKit)</span>
              <el-button type="primary" link @click="fetchZlmMetrics">
                <el-icon><RefreshRight /></el-icon> 刷新
              </el-button>
            </div>
          </template>
          <div v-loading="loadingZlm" class="zlm-content">
            <el-row :gutter="20" style="margin-bottom: 20px;">
              <el-col :span="8">
                <el-statistic title="活跃流总数" :value="uniqueStreams.length" />
              </el-col>
              <el-col :span="8">
                <el-statistic title="协议分发总数" :value="zlmData.length" />
              </el-col>
              <el-col :span="8">
                <el-statistic title="当前总带宽" :value="formatBytes(totalBandwidth) + '/s'" />
              </el-col>
            </el-row>

            <el-table :data="uniqueStreamsData" height="250" style="width: 100%" size="small" border>
              <el-table-column prop="stream" label="流 ID (Stream)" width="120" />
              <el-table-column label="可用协议 (Schemas)">
                <template #default="scope">
                  <el-tag 
                    v-for="schema in scope.row.schemas" 
                    :key="schema" 
                    size="small" 
                    style="margin-right: 5px; margin-bottom: 5px;"
                    :type="schema === 'rtsp' ? 'success' : (schema === 'rtmp' ? 'warning' : 'info')"
                  >
                    {{ schema.toUpperCase() }}
                  </el-tag>
                </template>
              </el-table-column>
              <el-table-column label="在线时长" width="100">
                <template #default="scope">
                  {{ formatUptime(scope.row.aliveSecond) }}
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, computed } from 'vue'
import axios from 'axios'
import { Monitor, VideoCamera, RefreshRight } from '@element-plus/icons-vue'

const loadingSys = ref(false)
const loadingZlm = ref(false)
let refreshInterval = null

// Sys Info State
const sysInfo = ref({
  cpu: { usage: 0, cores: 0, model: '' },
  memory: { usagePercent: 0, used: 0, total: 0 },
  gpu: { util: 0, memUsed: 0, memTotal: 0 },
  os: { platform: '', release: '', uptime: 0 }
})

// ZLM State
const zlmData = ref([])

const customColors = [
  { color: '#5cb87a', percentage: 60 },
  { color: '#e6a23c', percentage: 80 },
  { color: '#f56c6c', percentage: 100 },
]

// Computed properties for ZLM
const uniqueStreams = computed(() => {
  return [...new Set(zlmData.value.map(item => item.stream))]
})

const uniqueStreamsData = computed(() => {
  const map = {}
  zlmData.value.forEach(item => {
    if (!map[item.stream]) {
      map[item.stream] = {
        stream: item.stream,
        schemas: [],
        aliveSecond: item.aliveSecond
      }
    }
    map[item.stream].schemas.push(item.schema)
  })
  return Object.values(map)
})

const totalBandwidth = computed(() => {
  // Only sum up the pull origin streams to avoid double counting
  const pulls = zlmData.value.filter(item => item.originTypeStr === 'pull' && item.schema === 'rtsp')
  if (pulls.length > 0) {
      return pulls.reduce((sum, item) => sum + (item.bytesSpeed || 0), 0)
  }
  // Fallback if no specific pull rtsp is found, just sum all (might be inaccurate)
  return zlmData.value.reduce((sum, item) => sum + (item.bytesSpeed || 0), 0) / zlmData.value.length || 0; 
})

// Formatting Helpers
const formatBytes = (bytes, decimals = 2) => {
  if (!+bytes) return '0 Bytes'
  const k = 1024
  const dm = decimals < 0 ? 0 : decimals
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB', 'PB', 'EB', 'ZB', 'YB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(dm))} ${sizes[i]}`
}

const formatUptime = (seconds) => {
  if (!seconds) return '0s'
  const d = Math.floor(seconds / (3600*24*24))
  const h = Math.floor(seconds % (3600*24) / 3600)
  const m = Math.floor(seconds % 3600 / 60)
  const s = Math.floor(seconds % 60)
  
  const dDisplay = d > 0 ? d + "d " : ""
  const hDisplay = h > 0 ? h + "h " : ""
  const mDisplay = m > 0 ? m + "m " : ""
  const sDisplay = s > 0 ? s + "s" : ""
  return dDisplay + hDisplay + mDisplay + sDisplay
}

// Fetchers
const fetchSysInfo = async () => {
  try {
    const res = await axios.get('/api/system/info')
    sysInfo.value = res.data
  } catch (e) {
    console.error('Failed to fetch system info')
  }
}

const fetchZlmMetrics = async () => {
  loadingZlm.value = true
  try {
    const res = await axios.get('/api/zlm/metrics')
    if (res.data.code === 0 && res.data.data) {
      zlmData.value = res.data.data
    } else {
      zlmData.value = []
    }
  } catch (e) {
    console.error('Failed to fetch ZLM metrics')
  }
  loadingZlm.value = false
}

onMounted(() => {
  loadingSys.value = true
  Promise.all([fetchSysInfo(), fetchZlmMetrics()]).finally(() => {
    loadingSys.value = false
  })
  
  // Refresh system info every 5 seconds
  refreshInterval = setInterval(() => {
    fetchSysInfo()
  }, 5000)
})

onBeforeUnmount(() => {
  if (refreshInterval) {
    clearInterval(refreshInterval)
  }
})
</script>

<style scoped>
.dashboard-container {
  padding: 10px;
}
.box-card {
  margin-bottom: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}
.sys-metrics {
  text-align: center;
  margin-bottom: 20px;
}
.metric-item {
  display: flex;
  flex-direction: column;
  align-items: center;
}
.percentage-value {
  display: block;
  font-size: 24px;
  font-weight: bold;
  color: #303133;
}
.percentage-label {
  display: block;
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
.metric-desc {
  margin-top: 10px;
  font-size: 13px;
  color: #606266;
}
</style>
