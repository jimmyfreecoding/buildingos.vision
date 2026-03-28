<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>摄像头视频流管理</span>
        <el-button type="primary" @click="saveConfig">保存并应用</el-button>
      </div>
    </template>

    <div v-loading="loading">
      <el-tabs v-model="activeTab">
        <!-- 吸烟检测流 -->
        <el-tab-pane label="吸烟检测 (Smoking)" name="smoking">
          <el-table :data="streams.smoking" style="width: 100%; margin-bottom: 20px;" border>
            <el-table-column prop="id" label="设备 ID" width="150" />
            <el-table-column prop="name" label="位置名称" width="200" />
            <el-table-column prop="source_url" label="RTSP 源地址" />
            <el-table-column label="操作" width="120">
              <template #default="scope">
                <el-button size="small" type="danger" @click="removeStream('smoking', scope.$index)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-button type="success" plain @click="addStream('smoking')">添加新流</el-button>
        </el-tab-pane>

        <!-- 人员感知流 -->
        <el-tab-pane label="人员感知 (Occupancy)" name="occupancy">
          <el-table :data="streams.occupancy" style="width: 100%; margin-bottom: 20px;" border>
            <el-table-column prop="id" label="设备 ID" width="150" />
            <el-table-column prop="name" label="位置名称" width="200" />
            <el-table-column prop="source_url" label="RTSP 源地址" />
            <el-table-column label="操作" width="120">
              <template #default="scope">
                <el-button size="small" type="danger" @click="removeStream('occupancy', scope.$index)">删除</el-button>
              </template>
            </el-table-column>
          </el-table>
          <el-button type="success" plain @click="addStream('occupancy')">添加新流</el-button>
        </el-tab-pane>
      </el-tabs>
    </div>

    <!-- 添加弹窗 -->
    <el-dialog v-model="dialogVisible" title="添加视频流">
      <el-form :model="newStream" label-width="120px">
        <el-form-item label="设备 ID">
          <el-input v-model="newStream.id" placeholder="如: cam_01 (需唯一)"></el-input>
        </el-form-item>
        <el-form-item label="位置名称">
          <el-input v-model="newStream.name" placeholder="如: 1楼大厅"></el-input>
        </el-form-item>
        <el-form-item label="RTSP 源地址">
          <el-input v-model="newStream.source_url" placeholder="rtsp://admin:pwd@ip:554/..."></el-input>
        </el-form-item>
      </el-form>
      <template #footer>
        <el-button @click="dialogVisible = false">取消</el-button>
        <el-button type="primary" @click="confirmAdd">确定</el-button>
      </template>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const loading = ref(false)
const activeTab = ref('smoking')
const fullConfig = ref({})
const streams = ref({ smoking: [], occupancy: [] })

const dialogVisible = ref(false)
const currentTarget = ref('')
const newStream = ref({ id: '', name: '', source_url: '' })

const fetchConfig = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/config')
    fullConfig.value = res.data
    if (res.data.streams) {
      streams.value = res.data.streams
    }
  } catch (e) {
    ElMessage.error('获取配置失败')
  }
  loading.value = false
}

const addStream = (type) => {
  currentTarget.value = type
  newStream.value = { id: '', name: '', source_url: '' }
  dialogVisible.value = true
}

const confirmAdd = () => {
  if (!newStream.value.id || !newStream.value.source_url) {
    return ElMessage.warning('ID 和 源地址必填')
  }
  // 自动补全 ZLM 代理相关的字段
  const streamData = {
    ...newStream.value,
    zlm_stream_id: newStream.value.id,
    url: `rtsp://zlm:554/live/${newStream.value.id}`
  }
  streams.value[currentTarget.value].push(streamData)
  dialogVisible.value = false
}

const removeStream = (type, index) => {
  streams.value[type].splice(index, 1)
}

const saveConfig = async () => {
  loading.value = true
  try {
    fullConfig.value.streams = streams.value
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
</style>
