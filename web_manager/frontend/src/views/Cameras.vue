<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>场景与摄像头管理 (双轨制架构)</span>
        <el-button type="primary" @click="saveConfig">保存并应用</el-button>
      </div>
    </template>

    <div v-loading="loading">
      <el-alert 
        title="双轨制架构提示" 
        type="info" 
        description="您可以将多个摄像头绑定到同一个场景 (Area) 中。吸烟检测仅在人员感知触发后才会激活小窗口。"
        show-icon
        style="margin-bottom: 20px;"
      />

      <div v-for="(area, areaIndex) in areas" :key="area.areaCode" class="area-card">
        <el-card shadow="hover" style="margin-bottom: 20px;">
          <template #header>
            <div class="area-header">
              <span style="font-weight: bold; font-size: 16px;">
                <el-icon><Location /></el-icon> 场景区域: {{ area.areaCode === 'UNKNOWN' ? '未分配区域' : area.areaCode }}
              </span>
              <div>
                <el-button size="small" type="success" plain @click="openAddCameraDialog(area.areaCode)">添加摄像头</el-button>
                <el-button size="small" type="danger" plain @click="removeArea(areaIndex)" v-if="area.areaCode !== 'UNKNOWN'">删除场景</el-button>
              </div>
            </div>
          </template>

          <el-table :data="area.cameras" style="width: 100%" border size="small">
            <el-table-column prop="id" label="设备 ID" width="150" />
            <el-table-column prop="name" label="位置名称" width="180" />
            <el-table-column prop="source_url" label="RTSP 源地址" />
            <el-table-column label="启用算法" width="220">
              <template #default="scope">
                <el-checkbox-group v-model="scope.row.tasks">
                  <el-checkbox label="presence">人员感知</el-checkbox>
                  <el-checkbox label="smoking">吸烟检测</el-checkbox>
                </el-checkbox-group>
              </template>
            </el-table-column>
            <el-table-column label="操作" width="100">
              <template #default="scope">
                <el-button size="small" type="danger" @click="removeCamera(areaIndex, scope.$index)">移除</el-button>
              </template>
            </el-table-column>
          </el-table>
        </el-card>
      </div>

      <el-button type="primary" plain @click="dialogAreaVisible = true" style="width: 100%; border-style: dashed;">
        <el-icon><Plus /></el-icon> 添加新场景 (Area)
      </el-button>
    </div>

    <!-- 添加场景弹窗 -->
    <el-dialog v-model="dialogAreaVisible" title="添加新场景" width="400px">
      <el-form label-width="100px">
        <el-form-item label="场景编号">
          <el-input v-model="newAreaCode" placeholder="如: Floor01/AreaA/Office01"></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogAreaVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmAddArea">确定</el-button>
      </template>
    </el-dialog>

    <!-- 添加摄像头弹窗 -->
    <el-dialog v-model="dialogCamVisible" title="添加摄像头" width="600px">
      <el-form :model="newCamera" label-width="120px">
        <el-form-item label="所属场景">
          <el-input v-model="currentAddAreaCode" disabled></el-input>
        </el-form-item>
        <el-form-item label="设备 ID">
          <el-input v-model="newCamera.id" placeholder="如: cam_01 (需全局唯一)"></el-input>
        </el-form-item>
        <el-form-item label="位置名称">
          <el-input v-model="newCamera.name" placeholder="如: 1楼大厅"></el-input>
        </el-form-item>
        <el-form-item label="RTSP 源地址">
          <el-input v-model="newCamera.source_url" placeholder="rtsp://admin:pwd@ip:554/..."></el-input>
        </el-form-item>
        <el-form-item label="初始算法">
          <el-checkbox-group v-model="newCamera.tasks">
            <el-checkbox label="presence">人员感知</el-checkbox>
            <el-checkbox label="smoking">吸烟检测</el-checkbox>
          </el-checkbox-group>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogCamVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmAddCamera">确定</el-button>
      </template>
    </el-dialog>

  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'
import { Location, Plus } from '@element-plus/icons-vue'

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
    ElMessage.error('获取配置失败')
  }
  loading.value = false
}

const confirmAddArea = () => {
  if (!newAreaCode.value.trim()) {
    return ElMessage.warning('场景编号不能为空')
  }
  if (areas.value.find(a => a.areaCode === newAreaCode.value)) {
    return ElMessage.warning('该场景已存在')
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
    return ElMessage.warning('设备 ID 和 源地址必填')
  }
  if (newCamera.value.tasks.length === 0) {
    return ElMessage.warning('至少需要选择一项算法任务')
  }
  
  const targetArea = areas.value.find(a => a.areaCode === currentAddAreaCode.value)
  if (targetArea) {
    // 检查 ID 是否全局冲突
    let isConflict = false
    areas.value.forEach(a => {
      if (a.cameras.find(c => c.id === newCamera.value.id)) isConflict = true
    })
    if (isConflict) return ElMessage.warning('设备 ID 已存在，请确保全局唯一')

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
    ElMessage.success('保存成功，配置将在 AI 引擎重启后生效')
  } catch (e) {
    ElMessage.error('保存失败')
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