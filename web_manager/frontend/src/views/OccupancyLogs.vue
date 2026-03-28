<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>人存在算法过程观测 (Occupancy Algorithm Dashboard)</span>
        <el-button @click="fetchLogs" type="primary" plain size="small" :loading="loading">刷新</el-button>
      </div>
    </template>

    <div v-loading="loading">
      <el-row :gutter="20">
        <el-col :span="6">
          <div class="filter-section">
            <el-select v-model="selectedArea" placeholder="选择区域" style="width: 100%; margin-bottom: 10px;" clearable>
              <el-option v-for="area in uniqueAreas" :key="area" :label="area" :value="area"></el-option>
            </el-select>
            
            <el-select v-model="selectedDate" placeholder="选择日期" style="width: 100%; margin-bottom: 20px;" clearable>
              <el-option v-for="date in uniqueDates" :key="date" :label="date" :value="date"></el-option>
            </el-select>
          </div>
        </el-col>
        
        <el-col :span="18">
          <div v-if="filteredLogs.length === 0" class="no-data">
            <el-empty description="暂无日志数据"></el-empty>
          </div>
          <el-timeline v-else>
            <el-timeline-item
              v-for="(log, index) in filteredLogs"
              :key="log.id"
              :timestamp="formatTime(log.timestamp)"
              :type="getEventTypeColor(log.event)"
              placement="top"
            >
              <el-card>
                <h4>{{ log.event }}</h4>
                <p><strong>区域:</strong> {{ log.areaCode }}</p>
                <p>
                  <strong>状态:</strong> 
                  <el-tag :type="log.is_occupied ? 'success' : 'info'" size="small">
                    {{ log.is_occupied ? '有人' : '无人' }}
                  </el-tag>
                </p>
                <p><strong>检测人数:</strong> {{ log.person_count || 0 }}</p>
                
                <el-collapse style="margin-top: 10px;">
                  <el-collapse-item title="查看多维得分详情" name="1">
                    <div v-if="log.scores">
                      <p>视觉基础分 (60%): {{ log.scores.visual?.toFixed(2) || 0 }}</p>
                      <p>微动变化分 (20%): {{ log.scores.motion?.toFixed(2) || 0 }}</p>
                      <p>时间偏置分 (20%): {{ log.scores.time_bias?.toFixed(2) || 0 }}</p>
                      <p style="font-weight: bold; color: #409EFF;">加权总分: {{ log.scores.total?.toFixed(2) || 0 }} (阈值: {{ log.threshold_used }})</p>
                    </div>
                  </el-collapse-item>
                </el-collapse>

                <div v-if="log.images && log.images.length > 0" class="image-gallery" style="margin-top: 15px;">
                  <p><strong>算法截图:</strong></p>
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

const loading = ref(false)
const logs = ref([])
const selectedArea = ref('')
const selectedDate = ref('')

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
    return matchArea && matchDate
  })
})

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleString()
}

const getEventTypeColor = (event) => {
  if (event === 'LEVEL_1_DECISION') return 'success'
  if (event === 'LEVEL_2_TRIGGER') return 'warning'
  if (event === 'LEVEL_3_TRIGGER') return 'danger'
  return 'info'
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
  height: 200px;
  border-radius: 4px;
  border: 1px solid #EBEEF5;
  background-color: #f5f7fa;
}
.no-data {
  display: flex;
  justify-content: center;
  align-items: center;
  height: 300px;
}
</style>