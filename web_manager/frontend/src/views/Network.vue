<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>{{ $t('network.title') }}</span>
      </div>
    </template>
    
    <el-form :model="form" label-width="120px" v-loading="loading">
      <el-form-item :label="$t('network.mode')">
        <el-radio-group v-model="form.mode">
          <el-radio label="dhcp">{{ $t('network.dhcp') }}</el-radio>
          <el-radio label="static">{{ $t('network.static') }}</el-radio>
        </el-radio-group>
      </el-form-item>
      
      <div v-if="form.mode === 'static'">
        <el-form-item :label="$t('network.ip')">
          <el-input v-model="form.ip" placeholder="192.168.1.100"></el-input>
        </el-form-item>
        <el-form-item :label="$t('network.netmask')">
          <el-input v-model="form.netmask" placeholder="255.255.255.0"></el-input>
        </el-form-item>
        <el-form-item :label="$t('network.gateway')">
          <el-input v-model="form.gateway" placeholder="192.168.1.1"></el-input>
        </el-form-item>
        <el-form-item :label="$t('network.dns')">
          <el-input v-model="form.dns" placeholder="8.8.8.8, 114.114.114.114"></el-input>
        </el-form-item>
      </div>

      <el-form-item>
        <el-button type="primary" @click="saveNetwork">{{ $t('network.save') }}</el-button>
        <p style="color: #e6a23c; margin-left: 15px; font-size: 12px;">{{ $t('network.saveTip') }}</p>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { useI18n } from 'vue-i18n'
import { ElMessage, ElMessageBox } from 'element-plus'

const { t } = useI18n()
const loading = ref(false)
const form = ref({
  mode: 'dhcp',
  ip: '',
  netmask: '',
  gateway: '',
  dns: ''
})

const fetchNetwork = async () => {
  loading.value = true
  try {
    const res = await axios.get('/api/network')
    form.value = res.data
  } catch (e) {
    ElMessage.error(t('network.fetchFailed'))
  }
  loading.value = false
}

const saveNetwork = () => {
  ElMessageBox.confirm(
    t('network.confirmSave'),
    t('network.confirmTitle'),
    { confirmButtonText: t('network.confirmButton'), cancelButtonText: t('network.cancelButton'), type: 'warning' }
  ).then(async () => {
    loading.value = true
    try {
      await axios.post('/api/network', form.value)
      ElMessage.success(t('network.saveSuccess'))
    } catch (e) {
      ElMessage.error(t('network.saveFailed'))
    }
    loading.value = false
  }).catch(() => {})
}

onMounted(() => {
  fetchNetwork()
})
</script>
