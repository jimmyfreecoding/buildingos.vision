<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>{{ $t('params.title') }}</span>
        <el-button type="primary" @click="saveParams">{{ $t('params.save') }}</el-button>
      </div>
    </template>

    <el-form :model="params" label-width="250px" v-loading="loading">
      
      <el-divider content-position="left">{{ $t('params.mqttConfig') }}</el-divider>
      <el-form-item :label="$t('params.mqttBroker')">
        <el-input v-model="mqttParams.broker" placeholder="例如: 10.0.0.100 或 emqx.io"></el-input>
      </el-form-item>
      <el-form-item :label="$t('params.mqttPort')">
        <el-input-number v-model="mqttParams.port" :min="1" :max="65535"></el-input-number>
      </el-form-item>
      <el-form-item :label="$t('params.mqttKeepalive')">
        <el-input-number v-model="mqttParams.keepalive" :min="10" :max="300"></el-input-number>
      </el-form-item>

      <el-divider content-position="left">{{ $t('params.smokingTitle') }}</el-divider>
      <el-form-item :label="$t('params.poseConf')">
        <el-slider v-model="params.smoking_conf" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>
      <el-form-item :label="$t('params.poseHeuristic')">
        <el-slider v-model="params.pose_heuristic_threshold" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>
      <el-form-item :label="$t('params.smokingSpecConf')">
        <el-slider v-model="params.smoking_specialist_conf" :min="0" :max="1" :step="0.05" show-input></el-slider>
      </el-form-item>

      <el-divider content-position="left">{{ $t('params.occGlobalTitle') }}</el-divider>
      <el-form-item :label="$t('params.maxLogSize')">
        <el-input-number v-model="storageQuota.max_size_mb" :min="100" :max="10240" :step="100"></el-input-number>
        <div class="form-tip">{{ $t('params.maxLogSizeTip') }}</div>
      </el-form-item>

      <el-divider content-position="left">{{ $t('params.occAreasTitle') }}</el-divider>
      <div v-for="(area, index) in areas" :key="index" class="area-card">
        <div class="area-header">
          <span style="font-weight: bold;">{{ $t('params.areaCode') }}: </span>
          <el-input v-model="area.areaCode" style="width: 200px; margin-right: 10px;" placeholder="例如: Floor01/AreaA/Office01"></el-input>
          <el-button type="danger" size="small" @click="removeArea(index)" v-if="areas.length > 1">{{ $t('params.deleteArea') }}</el-button>
        </div>
        <div class="area-body">
          <el-form-item :label="$t('params.scoreThreshold')">
            <el-slider v-model="area.score_threshold" :min="0" :max="1" :step="0.05" show-input style="width: 300px;"></el-slider>
          </el-form-item>
          <el-form-item :label="$t('params.bufferMinutes')">
            <el-input-number v-model="area.buffer_minutes" :min="1" :max="60"></el-input-number>
            <div class="form-tip">{{ $t('params.bufferMinutesTip') }}</div>
          </el-form-item>
          <el-form-item :label="$t('params.level2Minutes')">
            <el-input-number v-model="area.level2_minutes" :min="1" :max="120"></el-input-number>
            <div class="form-tip">{{ $t('params.level2MinutesTip') }}</div>
          </el-form-item>
          <el-form-item :label="$t('params.level3Minutes')">
            <el-input-number v-model="area.level3_minutes" :min="1" :max="120"></el-input-number>
            <div class="form-tip">{{ $t('params.level3MinutesTip') }}</div>
          </el-form-item>
        </div>
      </div>
      <el-button type="primary" plain @click="addArea" style="margin-left: 250px; margin-bottom: 20px;">{{ $t('params.addArea') }}</el-button>
    </el-form>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
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
    ElMessage.error(t('params.fetchFailed'))
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
    ElMessage.success(t('params.saveSuccess'))
  } catch (e) {
    ElMessage.error(t('params.saveFailed'))
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
