<template>
  <div class="ai-monitor-container">
    <el-row :gutter="20" style="height: 100%;">
      <!-- Left: Camera Grid -->
      <el-col :span="16" style="height: 100%;">
        <el-card class="box-card video-card" body-style="height: calc(100% - 60px); padding: 10px;">
          <template #header>
            <div class="card-header">
              <span><el-icon><VideoCamera /></el-icon> {{ $t('monitor.videoMatrix') }}</span>
              <el-tag type="success" size="small" v-if="cameras.length > 0">{{ $t('monitor.onlineCount', { count: cameras.length }) }}</el-tag>
            </div>
          </template>
          <CameraGrid v-if="cameras.length > 0" :cameras="cameras" />
          <el-empty v-else :description="$t('monitor.noConfig')" />
        </el-card>
      </el-col>

      <!-- Right: AI Logs & Events -->
      <el-col :span="8" style="height: 100%;">
        <el-card class="box-card logs-card" body-style="height: calc(100% - 60px); padding: 0; display: flex; flex-direction: column;">
          <template #header>
            <div class="card-header">
              <span><el-icon><DataLine /></el-icon> {{ $t('monitor.aiLogs') }}</span>
              <el-switch v-model="autoScroll" :active-text="$t('monitor.autoScroll')" size="small" />
            </div>
          </template>
          
          <div class="log-terminal" ref="logContainer">
            <div v-for="(log, index) in logs" :key="index" class="log-entry" :class="getLogClass(log.message)">
              <span class="log-time">[{{ formatTime(log.timestamp) }}]</span>
              <span class="log-cam" v-if="log.camId && log.camId !== 'system'">[{{ log.camId }}]</span>
              <span class="log-msg">{{ log.message }}</span>
            </div>
          </div>
        </el-card>
      </el-col>
    </el-row>
  </div>
</template>

<script setup>
import { ref, onMounted, onBeforeUnmount, nextTick, watch } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'
import { VideoCamera, DataLine } from '@element-plus/icons-vue'
import { io } from 'socket.io-client'
import CameraGrid from './CameraGrid.vue'

const { t } = useI18n()
const cameras = ref([])
const logs = ref([])
const autoScroll = ref(true)
const logContainer = ref(null)
let socket = null

const fetchConfig = async () => {
  try {
    const res = await axios.get('/api/config')
    const config = res.data
    const allStreams = []
    
    // Combine smoking and occupancy streams for the grid
    if (config.streams) {
      if (config.streams.smoking) {
        config.streams.smoking.forEach(s => {
            if (!allStreams.find(ex => ex.name === s.id)) {
                allStreams.push({ name: s.id, type: 'smoking' })
            }
        })
      }
      if (config.streams.occupancy) {
        config.streams.occupancy.forEach(s => {
            if (!allStreams.find(ex => ex.name === s.id)) {
                allStreams.push({ name: s.id, type: 'occupancy' })
            }
        })
      }
    }
    cameras.value = allStreams
  } catch (e) {
    console.error('Failed to fetch camera config:', e)
  }
}

const setupWebSocket = () => {
  // Connect to the backend server
  const serverUrl = window.location.hostname === 'localhost' ? 'http://localhost:3000' : ''
  socket = io(serverUrl)

  socket.on('connect', () => {
    logs.value.push({ timestamp: new Date(), message: t('monitor.logConnected'), camId: 'system' })
  })

  socket.on('log', (data) => {
    logs.value.push(data)
    // Keep only the last 1000 logs to prevent memory leaks
    if (logs.value.length > 1000) {
      logs.value.shift()
    }
  })

  socket.on('disconnect', () => {
    logs.value.push({ timestamp: new Date(), message: t('monitor.logDisconnected'), camId: 'system' })
  })
}

const formatTime = (isoString) => {
  const date = new Date(isoString)
  return `${date.getHours().toString().padStart(2, '0')}:${date.getMinutes().toString().padStart(2, '0')}:${date.getSeconds().toString().padStart(2, '0')}`
}

const getLogClass = (message) => {
  if (!message) return ''
  if (message.includes('[ERROR]') || message.includes('Failed')) return 'log-error'
  if (message.includes('Warning') || message.includes('POTENTIAL')) return 'log-warning'
  if (message.includes('Triggered') || message.includes('DETECTED') || message.includes('ACTIVE')) return 'log-highlight'
  return 'log-info'
}

watch(logs, () => {
  if (autoScroll.value && logContainer.value) {
    nextTick(() => {
      logContainer.value.scrollTop = logContainer.value.scrollHeight
    })
  }
}, { deep: true })

onMounted(() => {
  fetchConfig()
  setupWebSocket()
})

onBeforeUnmount(() => {
  if (socket) {
    socket.disconnect()
  }
})
</script>

<style scoped>
.ai-monitor-container {
  padding: 10px;
  height: calc(100vh - 100px); /* Adjust based on your layout header/footer */
}

.box-card {
  height: 100%;
}

.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}

.log-terminal {
  flex: 1;
  background-color: #1e1e1e;
  color: #d4d4d4;
  padding: 10px;
  overflow-y: auto;
  font-family: 'Consolas', 'Courier New', monospace;
  font-size: 13px;
  line-height: 1.5;
}

.log-entry {
  word-break: break-all;
  border-bottom: 1px solid #333;
  padding: 2px 0;
}

.log-time {
  color: #569cd6;
  margin-right: 5px;
}

.log-cam {
  color: #4ec9b0;
  margin-right: 5px;
  font-weight: bold;
}

.log-error {
  color: #f14c4c;
}

.log-warning {
  color: #cca700;
}

.log-highlight {
  color: #b5cea8;
  font-weight: bold;
  background-color: rgba(255, 255, 255, 0.1);
}

.log-info {
  color: #cccccc;
}
</style>