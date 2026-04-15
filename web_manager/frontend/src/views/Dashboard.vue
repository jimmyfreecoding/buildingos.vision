<template>
  <div class="dashboard-container">
    <el-row :gutter="20">
      <!-- System Info Card -->
      <el-col :span="12">
        <el-card class="box-card system-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><Monitor /></el-icon> {{ $t('dashboard.nodeStatus') }} ({{ sysInfo.board.model }})</span>
            </div>
          </template>
          <div v-loading="loadingSys" class="sys-content">
            <!-- First Row: Main Usage Metrics -->
            <el-row :gutter="10" class="sys-metrics">
              <el-col :span="6" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.cpu.usage" :color="customColors" :width="100">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(0) }}%</span>
                    <span class="percentage-label">{{ $t('dashboard.cpuLoad') }}</span>
                  </template>
                </el-progress>
                <div class="metric-desc">{{ sysInfo.cpu.cores }} Cores</div>
              </el-col>
              <el-col :span="6" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.memory.ram.usagePercent" :color="customColors" :width="100">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(0) }}%</span>
                    <span class="percentage-label">{{ $t('dashboard.unifiedMem') }}</span>
                  </template>
                </el-progress>
                <div class="metric-desc">{{ formatBytes(sysInfo.memory.ram.used) }} / {{ formatBytes(sysInfo.memory.ram.total) }}</div>
              </el-col>
              <el-col :span="6" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.disk?.percent || 0" :color="customColors" :width="100">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(0) }}%</span>
                    <span class="percentage-label">{{ $t('dashboard.diskUsage') }}</span>
                  </template>
                </el-progress>
                <div class="metric-desc">{{ formatBytes(sysInfo.disk?.used || 0) }} / {{ formatBytes(sysInfo.disk?.total || 0) }}</div>
              </el-col>
              <el-col :span="6" class="metric-item">
                <el-progress type="dashboard" :percentage="sysInfo.gpu.util" :color="customColors" :width="100">
                  <template #default="{ percentage }">
                    <span class="percentage-value">{{ percentage.toFixed(0) }}%</span>
                    <span class="percentage-label">{{ $t('dashboard.gpuLoad') }}</span>
                  </template>
                </el-progress>
                <div class="metric-desc">Mem: {{ sysInfo.gpu.memUsed.toFixed(0) }}MB / {{ sysInfo.gpu.memTotal.toFixed(0) }}MB</div>
              </el-col>
            </el-row>
            
            <el-divider style="margin: 10px 0;" />
            
            <!-- Second Row: Hardware Vitals (Power, Temp) -->
            <el-descriptions :column="2" border size="small">
              <el-descriptions-item :label="$t('dashboard.power')">
                <el-tag size="small" type="warning" effect="plain">{{ (sysInfo.power.total / 1000).toFixed(1) }} W</el-tag>
                <span style="font-size: 12px; color: #909399; margin-left: 5px;">(CPU+GPU+CV: {{ (powerComputeRail / 1000).toFixed(1) }} W)</span>
              </el-descriptions-item>
              <el-descriptions-item :label="$t('dashboard.temp')">
                <span style="font-size: 12px;">
                  GPU: <span :style="{ color: gpuTemp > 75 ? 'red' : 'inherit' }">{{ gpuTemp || 'N/A' }}°C</span> | 
                  CPU: <span :style="{ color: cpuTemp > 75 ? 'red' : 'inherit' }">{{ cpuTemp || 'N/A' }}°C</span>
                </span>
              </el-descriptions-item>
              <el-descriptions-item :label="$t('dashboard.nvpModel')">
                <el-tag size="small" type="success">{{ sysInfo.board.nvpmodel }}</el-tag>
              </el-descriptions-item>
              <el-descriptions-item :label="$t('dashboard.uptime')">{{ formatUptime(sysInfo.board.uptime) }}</el-descriptions-item>
            </el-descriptions>

            <el-divider style="margin: 10px 0;" />

            <el-alert
              v-for="(warning, idx) in systemWarnings"
              :key="idx"
              :title="warning"
              type="warning"
              show-icon
              :closable="false"
              class="warning-item"
            />

            <el-card shadow="never" class="engine-panel">
              <div class="engine-header">{{ $t('dashboard.enginePanel') }}</div>
              <div class="engine-tags">
                <el-tag
                  v-for="engine in engineEntries"
                  :key="engine.name"
                  size="small"
                  :type="engine.value > 60 ? 'danger' : (engine.value > 0 ? 'warning' : 'info')"
                >
                  {{ engine.name }}: {{ formatEngineValue(engine.value) }}
                </el-tag>
              </div>
              <div class="engine-note">{{ $t('dashboard.engineNote') }}</div>
              <div v-if="isEngineAllIdle" class="engine-idle-note">{{ $t('dashboard.engineIdleNote') }}</div>
            </el-card>
          </div>
        </el-card>

        <!-- Gemma Local Model Status Card -->
        <el-card class="box-card gemma-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><Cpu /></el-icon> {{ $t('dashboard.gemmaStatus') }} (Gemma 4 E2B)</span>
              <el-tag size="small" :type="gemmaStatus === 'Running' ? 'success' : (gemmaStatus === 'Loading' ? 'warning' : 'danger')">
                <span style="display: flex; align-items: center; gap: 5px;">
                  <span v-if="gemmaStatus === 'Running'" class="status-dot green"></span>
                  <span v-else-if="gemmaStatus === 'Loading'" class="status-dot yellow"></span>
                  <span v-else class="status-dot red"></span>
                  {{ gemmaStatus }}
                </span>
              </el-tag>
            </div>
          </template>
          <div class="gemma-content" v-loading="loadingSys">
            <el-descriptions :column="1" border size="small" v-if="gemmaStatus === 'Running' && gemmaDetails">
              <el-descriptions-item :label="$t('dashboard.modelInstance')">
                {{ gemmaDetails.props?.default_generation_settings?.model || 'llama.cpp GGUF Model' }}
              </el-descriptions-item>
              <el-descriptions-item label="上下文容量 (Context Size)">
                {{ gemmaDetails.props?.default_generation_settings?.n_ctx || 'Unknown' }} Tokens
              </el-descriptions-item>
              <el-descriptions-item :label="$t('dashboard.runStatus')">
                <div v-if="gemmaDetails.slots && gemmaDetails.slots.length > 0">
                  <div v-for="slot in gemmaDetails.slots" :key="slot.id" style="margin-bottom: 5px;">
                    <el-tag size="small" :type="slot.state === 0 ? 'info' : 'primary'">
                      Slot {{ slot.id }}: {{ slot.state === 0 ? 'Idle (空闲)' : 'Running (运行中)' }}
                    </el-tag>
                    <span v-if="slot.state !== 0" style="margin-left: 10px; font-size: 12px; color: #606266;">
                      Prompt: {{ slot.n_prompt_tokens }} | Decoded: {{ slot.n_decoded_tokens }}
                    </span>
                  </div>
                </div>
                <div v-else>
                  <el-tag size="small" type="info">Idle (空闲)</el-tag>
                </div>
              </el-descriptions-item>
            </el-descriptions>
            <div v-else-if="gemmaStatus === 'Loading'">
              <el-alert :title="$t('dashboard.modelLoading')" type="warning" show-icon :closable="false" />
            </div>
            <div v-else>
              <el-alert :title="$t('dashboard.modelOffline')" type="error" show-icon :closable="false" />
            </div>
          </div>
        </el-card>
      </el-col>

      <!-- ZLM Media Server Status Card -->
      <el-col :span="12">
        <el-card class="box-card zlm-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><VideoCamera /></el-icon> {{ $t('dashboard.zlmStatus') }} (ZLMediaKit)</span>
              <el-tag size="small" type="info">{{ $t('dashboard.refreshPerSec') }}</el-tag>
            </div>
          </template>
          <div class="zlm-content">
            <el-row :gutter="20" style="margin-bottom: 20px;">
              <el-col :span="8">
                <el-statistic :title="$t('dashboard.activeStreams')" :value="uniqueStreams.length" />
              </el-col>
              <el-col :span="8">
                <el-statistic :title="$t('dashboard.totalProtocols')" :value="zlmData.length" />
              </el-col>
              <el-col :span="8">
                <el-statistic :title="$t('dashboard.totalBandwidth')" :value="formatBytes(totalBandwidth) + '/s'" />
              </el-col>
            </el-row>

            <el-table :data="uniqueStreamsData" height="150" style="width: 100%" size="small" border>
              <el-table-column prop="stream" :label="$t('dashboard.streamId')" width="120" />
              <el-table-column :label="$t('dashboard.schemas')">
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
              <el-table-column :label="$t('dashboard.uptime')" width="100">
                <template #default="scope">
                  {{ formatUptime(scope.row.aliveSecond) }}
                </template>
              </el-table-column>
            </el-table>
          </div>
        </el-card>

        <!-- AI Engine Status Card -->
        <el-card class="box-card ai-card">
          <template #header>
            <div class="card-header">
              <span><el-icon><Cpu /></el-icon> {{ $t('dashboard.aiTasks') }}</span>
            </div>
          </template>
          <div class="ai-content">
             <el-table :data="aiTasks" height="150" style="width: 100%" size="small" border>
              <el-table-column prop="camId" :label="$t('dashboard.cameraId')" width="100" />
              <el-table-column prop="taskType" :label="$t('dashboard.algoType')" width="120">
                 <template #default="scope">
                    <el-tag size="small" :type="scope.row.taskType === 'smoking' ? 'danger' : 'primary'">
                      {{ scope.row.taskType === 'smoking' ? $t('dashboard.smoking') : $t('dashboard.presence') }}
                    </el-tag>
                 </template>
              </el-table-column>
              <el-table-column prop="status" :label="$t('dashboard.status')">
                 <template #default="scope">
                    <el-tag size="small" :type="scope.row.status === 'Running' ? 'success' : 'warning'">
                      <span style="display: flex; align-items: center; gap: 5px;">
                        <span v-if="scope.row.status === 'Running'" class="status-dot green"></span>
                        <span v-else class="status-dot yellow"></span>
                        {{ scope.row.status }}
                      </span>
                    </el-tag>
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
import { useI18n } from 'vue-i18n'
import { Monitor, VideoCamera, Cpu } from '@element-plus/icons-vue'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
const loadingSys = ref(false)
let refreshInterval = null
let zlmInterval = null
const prevSwapUsed = ref(0)
const prevGpuFreq = ref(0)
const systemWarnings = ref([])

// Sys Info State
const sysInfo = ref({
  cpu: { usage: 0, cores: 0, details: {} },
  memory: { 
    ram: { usagePercent: 0, used: 0, total: 0 },
    swap: { usagePercent: 0, used: 0, total: 0 }
  },
  gpu: { util: 0, memUsed: 0, memTotal: 0, freq: 0 },
  engines: {},
  power: { total: 0, gpu: 0, cpu: 0 },
  temperature: {},
  board: { model: 'Loading...', jetpack: '', nvpmodel: '', uptime: 0 }
})

// ZLM State
const zlmData = ref([])

// AI Tasks State
const aiTasks = ref([])

// Gemma State
const gemmaStatus = ref('Unknown')
const gemmaDetails = ref(null)

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
  return zlmData.value.reduce((sum, item) => sum + (item.bytesSpeed || 0), 0) / zlmData.value.length || 0; 
})

const engineEntries = computed(() => {
  const engines = sysInfo.value.engines || {}
  return Object.entries(engines).map(([name, value]) => ({
    name,
    value: typeof value === 'number' ? value : (value ? 100 : 0)
  }))
})

const isEngineAllIdle = computed(() => {
  if (engineEntries.value.length === 0) return true
  return engineEntries.value.every(item => (item.value || 0) <= 0)
})

const powerComputeRail = computed(() => {
  const power = sysInfo.value.power || {}
  const rail = power.cpu_gpu_cv || 0
  return rail > 0 ? rail : (power.gpu || 0)
})

const getTempByKeys = (keys) => {
  const temps = sysInfo.value.temperature || {}
  const found = Object.entries(temps).find(([name]) => keys.includes(name))
  return found ? found[1] : 0
}

const gpuTemp = computed(() => getTempByKeys(['GPU', 'gpu', 'GPU-therm', 'gpu-therm', 'TGPU']))
const cpuTemp = computed(() => getTempByKeys(['CPU', 'cpu', 'CPU-therm', 'cpu-therm', 'TCPU']))

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
  const d = Math.floor(seconds / (3600*24))
  const h = Math.floor(seconds % (3600*24) / 3600)
  const m = Math.floor(seconds % 3600 / 60)
  const s = Math.floor(seconds % 60)
  
  const dDisplay = d > 0 ? d + "d " : ""
  const hDisplay = h > 0 ? h + "h " : ""
  const mDisplay = m > 0 ? m + "m " : ""
  const sDisplay = s > 0 ? s + "s" : ""
  return dDisplay + hDisplay + mDisplay + sDisplay
}

const formatEngineValue = (value) => {
  if (typeof value !== 'number') return String(value)
  return `${value.toFixed(0)}%`
}

const updateWarnings = (data) => {
  const warnings = []
  const ramUsage = data?.memory?.ram?.usagePercent || 0
  const swapUsage = data?.memory?.swap?.usagePercent || 0
  const swapUsed = data?.memory?.swap?.used || 0
  const gpuUtil = data?.gpu?.util || 0
  const gpuFreq = data?.gpu?.freq || 0

  if (ramUsage >= 90) {
    warnings.push(t('dashboard.warnRam'))
  }

  if (swapUsage >= 50) {
    warnings.push(t('dashboard.warnSwap'))
  }

  if (prevSwapUsed.value > 0) {
    const swapDeltaMb = (swapUsed - prevSwapUsed.value) / (1024 * 1024)
    if (swapDeltaMb > 20) {
      warnings.push(t('dashboard.warnSwapDelta', { delta: swapDeltaMb.toFixed(1) }))
    }
  }

  if (prevGpuFreq.value > 0 && gpuUtil >= 85 && gpuFreq < prevGpuFreq.value * 0.85) {
    warnings.push(t('dashboard.warnGpuFreq'))
  }

  if (gpuTemp.value >= 80 || cpuTemp.value >= 80) {
    warnings.push(t('dashboard.warnTemp'))
  }

  const engines = data?.engines || {}
  const heavyEngines = Object.entries(engines)
    .filter(([, value]) => typeof value === 'number' && value >= 60)
    .map(([name]) => name)
  if (heavyEngines.length > 0) {
    warnings.push(t('dashboard.warnEngines', { engines: heavyEngines.join('、') }))
  }

  systemWarnings.value = warnings
  prevSwapUsed.value = swapUsed
  prevGpuFreq.value = gpuFreq
}

// Fetchers
const fetchSysInfo = async () => {
  try {
    const res = await axios.get('/api/system/info')
    sysInfo.value = res.data
    updateWarnings(res.data)
  } catch (e) {
    systemWarnings.value = [t('dashboard.fetchFailed')]
  }
}

const fetchZlmMetrics = async () => {
  try {
    const res = await axios.get('/api/zlm/metrics')
    if (res.data.code === 0 && res.data.data) {
      zlmData.value = res.data.data
    } else {
      zlmData.value = []
    }
  } catch (e) {
    // console.error('Failed to fetch ZLM metrics')
  }
}

const fetchAiTasks = async () => {
  try {
    const res = await axios.get('/api/ai/status')
    aiTasks.value = res.data
  } catch (e) {
    // console.error('Failed to fetch AI status')
  }
}

const fetchGemmaStatus = async () => {
  try {
    const res = await axios.get('/api/gemma/status')
    gemmaStatus.value = res.data.status
    gemmaDetails.value = res.data.details
  } catch (e) {
    gemmaStatus.value = 'Offline'
    gemmaDetails.value = null
  }
}

onMounted(() => {
  loadingSys.value = true
  Promise.all([fetchSysInfo(), fetchZlmMetrics(), fetchAiTasks(), fetchGemmaStatus()]).finally(() => {
    loadingSys.value = false
  })
  
  refreshInterval = setInterval(() => {
    fetchSysInfo()
    fetchAiTasks()
  }, 1000)

  // Refresh ZLM metrics every 1 second
  zlmInterval = setInterval(() => {
    fetchZlmMetrics()
    fetchGemmaStatus() // fetch Gemma real-time slot state often
  }, 1000)
})

onBeforeUnmount(() => {
  if (refreshInterval) clearInterval(refreshInterval)
  if (zlmInterval) clearInterval(zlmInterval)
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
.metric-note {
  margin-top: 4px;
  font-size: 11px;
  color: #909399;
  text-align: center;
}
.warning-item {
  margin-top: 8px;
}
.engine-panel {
  margin-top: 10px;
  border: 1px solid #ebeef5;
}
.engine-header {
  font-size: 13px;
  color: #303133;
  font-weight: 600;
  margin-bottom: 8px;
}
.engine-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 6px;
}
.engine-note {
  margin-top: 8px;
  font-size: 12px;
  color: #909399;
}
.engine-idle-note {
  margin-top: 6px;
  font-size: 12px;
  color: #67c23a;
}
.status-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  display: inline-block;
}
.status-dot.green {
  background-color: #67C23A;
  box-shadow: 0 0 5px #67C23A;
}
.status-dot.yellow {
  background-color: #E6A23C;
  box-shadow: 0 0 5px #E6A23C;
}
.status-dot.red {
  background-color: #F56C6C;
  box-shadow: 0 0 5px #F56C6C;
}
</style>
