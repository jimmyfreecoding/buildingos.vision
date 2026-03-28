<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>系统参数与 AI 配置</span>
        <el-button type="primary" @click="saveParams">保存配置</el-button>
      </div>
    </template>

    <el-form :model="params" label-width="250px" v-loading="loading">
      
      <el-divider content-position="left">MQTT 物联网配置</el-divider>
      <el-form-item label="MQTT Broker 地址">
        <el-input v-model="mqttParams.broker" placeholder="例如: 10.0.0.100 或 emqx.io"></el-input>
      </el-form-item>
      <el-form-item label="MQTT 端口">
        <el-input-number v-model="mqttParams.port" :min="1" :max="65535"></el-input-number>
      </el-form-item>
      <el-form-item label="Keepalive (秒)">
        <el-input-number v-model="mqttParams.keepalive" :min="10" :max="300"></el-input-number>
      </el-form-item>

      <el-divider content-position="left">吸烟检测参数 (Smoking Detection)</el-divider>
      <el-form-item label="阶段一：姿态阈值 (Pose Conf)">
        <el-slider v-model="params.smoking_conf" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>
      <el-form-item label="阶段一：手靠近脸距离系数">
        <el-slider v-model="params.pose_heuristic_threshold" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>
      <el-form-item label="阶段二：烟雾识别置信度">
        <el-slider v-model="params.smoking_specialist_conf" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>

      <el-divider content-position="left">人存在算法全局配置 (Occupancy Algorithm)</el-divider>
      <el-form-item label="算法日志存储空间上限 (MB)">
        <el-input-number v-model="storageQuota.max_size_mb" :min="100" :max="10240" :step="100"></el-input-number>
        <div class="form-tip">当历史截图和日志超过此大小，系统将自动清理最老的数据以释放空间（例如: 1024MB = 1GB）</div>
      </el-form-item>

      <el-divider content-position="left">人存在算法区域策略配置 (Occupancy Areas)</el-divider>
      <div v-for="(area, index) in areas" :key="index" class="area-card">
        <div class="area-header">
          <span style="font-weight: bold;">区域标识 (areaCode): </span>
          <el-input v-model="area.areaCode" style="width: 200px; margin-right: 10px;" placeholder="例如: Floor01/AreaA/Office01"></el-input>
          <el-button type="danger" size="small" @click="removeArea(index)" v-if="areas.length > 1">删除此区域</el-button>
        </div>
        <div class="area-body">
          <el-form-item label="多维加权总分阈值">
            <el-slider v-model="area.score_threshold" :min="0" :max="1" :step="0.05" show-input style="width: 300px;"></el-slider>
          </el-form-item>
          <el-form-item label="缓冲期 (分钟)">
            <el-input-number v-model="area.buffer_minutes" :min="1" :max="60"></el-input-number>
            <div class="form-tip">判断为无人的观察期 (默认: 2)</div>
          </el-form-item>
          <el-form-item label="Level 2 触发时间 (分钟)">
            <el-input-number v-model="area.level2_minutes" :min="1" :max="120"></el-input-number>
            <div class="form-tip">持续无人，下发 Level 2 通知 (如: 调暗灯光, 默认: 5)</div>
          </el-form-item>
          <el-form-item label="Level 3 触发时间 (分钟)">
            <el-input-number v-model="area.level3_minutes" :min="1" :max="120"></el-input-number>
            <div class="form-tip">持续无人，下发 Level 3 通知 (如: 关闭设备, 默认: 10)</div>
          </el-form-item>
        </div>
      </div>
      <el-button type="primary" plain @click="addArea" style="margin-left: 250px; margin-bottom: 20px;">+ 添加区域策略</el-button>
    </el-form>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const loading = ref(false)
const fullConfig = ref({})
const params = ref({
  smoking_conf: 0.5,
  occupancy_conf: 0.4,
  state_patience: 120,
  smoking_specialist_conf: 0.25,
  pose_heuristic_threshold: 0.40
})

const mqttParams = ref({
  broker: "buildingos-emqx-prod",
  port: 1883,
  keepalive: 60
})

const storageQuota = ref({
  max_size_mb: 1024
})

const areas = ref([
  {
    areaCode: "Floor01/AreaA/Office01",
    score_threshold: 0.6,
    buffer_minutes: 2,
    level2_minutes: 5,
    level3_minutes: 10
  }
])

const addArea = () => {
  areas.value.push({
    areaCode: "NewArea",
    score_threshold: 0.6,
    buffer_minutes: 2,
    level2_minutes: 5,
    level3_minutes: 10
  })
}

const removeArea = (index) => {
  areas.value.splice(index, 1)
}

const fetchConfig = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/config')
    fullConfig.value = res.data
    if (res.data.model_params) {
      params.value = res.data.model_params
    }
    if (res.data.mqtt) {
      mqttParams.value = res.data.mqtt
    }
    if (res.data.storage_quota) {
      storageQuota.value.max_size_mb = res.data.storage_quota.max_size_mb || 1024
    }
    if (res.data.areas && Array.isArray(res.data.areas)) {
      areas.value = res.data.areas
    }
  } catch (e) {
    ElMessage.error('获取配置失败')
  }
  loading.value = false
}

const saveParams = async () => {
  loading.value = true
  try {
    fullConfig.value.model_params = params.value
    fullConfig.value.mqtt = mqttParams.value
    
    if (!fullConfig.value.storage_quota) fullConfig.value.storage_quota = {}
    fullConfig.value.storage_quota.max_size_mb = storageQuota.value.max_size_mb
    
    fullConfig.value.areas = areas.value

    await axios.post('/api/config', fullConfig.value)
    ElMessage.success('保存成功，AI引擎已自动重启生效')
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
.form-tip {
  font-size: 12px;
  color: #909399;
  line-height: 1.2;
  margin-top: 4px;
}
.area-card {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 15px;
  margin-bottom: 20px;
  margin-left: 50px;
  background-color: #fcfcfc;
}
.area-header {
  margin-bottom: 15px;
  padding-bottom: 10px;
  border-bottom: 1px solid #ebeef5;
}
</style>
