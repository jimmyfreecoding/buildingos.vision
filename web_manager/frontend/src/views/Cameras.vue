<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>{{ $t('cameras.title') }}</span>
        <el-button type="primary" @click="saveConfig">{{ $t('cameras.save') }}</el-button>
      </div>
    </template>

    <div v-loading="loading">
      <el-alert 
        :title="$t('cameras.architectureTip')" 
        type="info" 
        :description="$t('cameras.architectureDesc')"
        show-icon
        style="margin-bottom: 20px;"
      />

      <div v-for="(area, areaIndex) in areas" :key="area.areaCode" class="area-card">
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <div class="area-header">
              <span style="font-weight: bold; font-size: 16px;">
                <el-icon><Location /></el-icon> {{ $t('cameras.areaRegion') }}: {{ area.areaCode === 'UNKNOWN' ? $t('cameras.unassignedArea') : area.areaCode }}
              </span>
              <div>
                <el-button size="small" type="success" plain @click="openAddCameraDialog(area.areaCode)">{{ $t('cameras.addCamera') }}</el-button>
                <el-button size="small" type="danger" plain @click="removeArea(areaIndex)" v-if="area.areaCode !== 'UNKNOWN'">{{ $t('cameras.deleteArea') }}</el-button>
              </div>
            </div>
          </template>

          <el-table :data="area.cameras" style="width: 100%" border size="small">
            <el-table-column prop="id" :label="$t('cameras.deviceId')" width="150" />
            <el-table-column prop="name" :label="$t('cameras.locationName')" width="180" />
            <el-table-column prop="source_url" :label="$t('cameras.rtspUrl')" />
            <el-table-column :label="$t('cameras.enabledAlgorithms')" width="220">
              <template #default="scope">
                <el-checkbox-group v-model="scope.row.tasks">
                  <el-checkbox label="presence">{{ $t('cameras.presence') }}</el-checkbox>
                  <el-checkbox label="smoking">{{ $t('cameras.smoking') }}</el-checkbox>
                </el-checkbox-group>
              </template>
            </el-table-column>
            <el-table-column :label="$t('cameras.actions')" width="100">
              <template #default="scope">
                <el-button size="small" type="danger" @click="removeCamera(areaIndex, scope.$index)">{{ $t('cameras.remove') }}</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <el-button type="primary" plain @click="dialogAreaVisible = true" style="width: 100%; border-style: dashed;">
        <el-icon><Plus /></el-icon> {{ $t('cameras.addNewArea') }}
      </el-button>
    </div>

    <!-- 添加场景弹窗 -->
    <el-dialog v-model="dialogAreaVisible" :title="$t('cameras.addAreaDialogTitle')" width="400px">
      <el-form label-width="100px">
        <el-form-item :label="$t('cameras.areaCode')">
          <el-input v-model="newAreaCode" :placeholder="$t('cameras.areaCodePlaceholder')"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogAreaVisible = false">{{ $t('cameras.cancel') }}</el-button>
        <el-button type="primary" @click="confirmAddArea">{{ $t('cameras.confirm') }}</el-button>
      </template>
    </el-dialog>

    <!-- 添加摄像头弹窗 -->
    <el-dialog v-model="dialogCamVisible" :title="$t('cameras.addCamera')" width="600px">
      <el-form :model="newCamera" label-width="120px">
        <el-form-item :label="$t('cameras.belongingArea')">
          <el-input v-model="currentAddAreaCode" disabled></el-input>
        </el-form-item>
        <el-form-item :label="$t('cameras.deviceId')">
          <el-input v-model="newCamera.id" :placeholder="$t('cameras.deviceIdPlaceholder')"></el-input>
        </el-form-item>
        <el-form-item :label="$t('cameras.locationName')">
          <el-input v-model="newCamera.name" :placeholder="$t('cameras.locationNamePlaceholder')"></el-input>
        </el-form-item>
        <el-form-item :label="$t('cameras.rtspUrl')">
          <el-input v-model="newCamera.source_url" placeholder="rtsp://admin:pwd@ip:554/..."></el-input>
        </el-form-item>
        <el-form-item :label="$t('cameras.initialAlgorithms')">
          <el-checkbox-group v-model="newCamera.tasks">
            <el-checkbox label="presence">{{ $t('cameras.presence') }}</el-checkbox>
            <el-checkbox label="smoking">{{ $t('cameras.smoking') }}</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogCamVisible = false">{{ $t('cameras.cancel') }}</el-button>
        <el-button type="primary" @click="confirmAddCamera">{{ $t('cameras.confirm') }}</el-button>
      </template>
    </el-dialog>

  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { Location, Plus } from '@element-plus/icons-vue'

const { t } = useI18n()
const loading = ref(false)
const fullConfig = ref({})
const areas = ref([])

// Dialog states
const dialogAreaVisible = ref(false)
const newAreaCode = ref('')

const dialogCamVisible = ref(false)
const currentAddAreaCode = ref('')
const newCamera = ref({ id: '', name: '', source_url: '', tasks: ['presence'] })

const fetchConfig = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/config')
    fullConfig.value = res.data
    
    const streams = res.data.streams || { smoking: [], occupancy: [] }
    const areaMap = {}

    // 解析 Presence 摄像头
    streams.occupancy.forEach(cam => {
      const code = cam.areaCode || 'UNKNOWN'
      if (!areaMap[code]) areaMap[code] = { areaCode: code, cameras: [] }
      areaMap[code].cameras.push({
        id: cam.id,
        name: cam.name,
        source_url: cam.source_url,
        tasks: ['presence']
      })
    })

    // 解析 Smoking 摄像头并合并
    streams.smoking.forEach(cam => {
      const code = cam.areaCode || 'UNKNOWN'
      if (!areaMap[code]) areaMap[code] = { areaCode: code, cameras: [] }
      
      const existing = areaMap[code].cameras.find(c => c.id === cam.id)
      if (existing) {
        if (!existing.tasks.includes('smoking')) {
          existing.tasks.push('smoking')
        }
      } else {
        areaMap[code].cameras.push({
          id: cam.id,
          name: cam.name,
          source_url: cam.source_url,
          tasks: ['smoking']
        })
      }
    })

    areas.value = Object.values(areaMap)
  } catch (e) {
    ElMessage.error(t('cameras.fetchFailed'))
  }
  loading.value = false
}

const confirmAddArea = () => {
  if (!newAreaCode.value.trim()) {
    return ElMessage.warning(t('cameras.areaCodeEmpty'))
  }
  if (areas.value.find(a => a.areaCode === newAreaCode.value)) {
    return ElMessage.warning(t('cameras.areaExists'))
  }
  areas.value.push({ areaCode: newAreaCode.value, cameras: [] })
  dialogAreaVisible.value = false
  newAreaCode.value = ''
}

const removeArea = (index) => {
  areas.value.splice(index, 1)
}

const openAddCameraDialog = (areaCode) => {
  currentAddAreaCode.value = areaCode
  newCamera.value = { id: '', name: '', source_url: '', tasks: ['presence', 'smoking'] }
  dialogCamVisible.value = true
}

const confirmAddCamera = () => {
  if (!newCamera.value.id || !newCamera.value.source_url) {
    return ElMessage.warning(t('cameras.idUrlRequired'))
  }
  if (newCamera.value.tasks.length === 0) {
    return ElMessage.warning(t('cameras.taskRequired'))
  }
  
  const targetArea = areas.value.find(a => a.areaCode === currentAddAreaCode.value)
  if (targetArea) {
    // 检查 ID 是否全局冲突
    let isConflict = false
    areas.value.forEach(a => {
      if (a.cameras.find(c => c.id === newCamera.value.id)) isConflict = true
    })
    if (isConflict) return ElMessage.warning(t('cameras.idConflict'))

    targetArea.cameras.push({ ...newCamera.value })
  }
  dialogCamVisible.value = false
}

const removeCamera = (areaIndex, camIndex) => {
  areas.value[areaIndex].cameras.splice(camIndex, 1)
}

const saveConfig = async () => {
  loading.value = true
  try {
    const newStreams = { smoking: [], occupancy: [] }
    
    // 拍平回底层的 streams 结构
    areas.value.forEach(area => {
      area.cameras.forEach(cam => {
        if (cam.tasks.length === 0) return

        const streamData = {
          id: cam.id,
          name: cam.name,
          areaCode: area.areaCode,
          source_url: cam.source_url,
          zlm_stream_id: cam.id,
          url: `rtsp://zlm:554/live/${cam.id}`
        }
        
        if (cam.tasks.includes('presence')) {
          newStreams.occupancy.push({ ...streamData })
        }
        if (cam.tasks.includes('smoking')) {
          // Smoking 也携带 areaCode 以便 AI 引擎统一处理
          newStreams.smoking.push({ ...streamData })
        }
      })
    })

    fullConfig.value.streams = newStreams
    await axios.post('/api/config', fullConfig.value)
    ElMessage.success(t('cameras.saveSuccess'))
  } catch (e) {
    ElMessage.error(t('cameras.saveFailed'))
  }
  loading.value = false
}

onMounted(() => {
  fetchConfig()
})
</script>

<style scoped>
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.area-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
</style>