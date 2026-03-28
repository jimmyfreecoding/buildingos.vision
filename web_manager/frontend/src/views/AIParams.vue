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

      <el-divider content-position="left">人员感知参数 (Occupancy)</el-divider>
      <el-form-item label="人员识别置信度">
        <el-slider v-model="params.occupancy_conf" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>
      <el-form-item label="无人判定延迟时间 (秒)">
        <el-input-number v-model="params.state_patience" :min="10" :max="600" :step="10"></el-input-number>
      </el-form-item>
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
