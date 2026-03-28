<template>
  <el-card class="box-card">
    <template #header>
      <div class="card-header">
        <span>系统网络配置 (Jetson)</span>
      </div>
    </template>
    
    <el-form :model="form" label-width="120px" v-loading="loading">
      <el-form-item label="网络模式">
        <el-radio-group v-model="form.mode">
          <el-radio label="dhcp">自动获取 (DHCP)</el-radio>
          <el-radio label="static">静态 IP</el-radio>
        </el-radio-group>
      </el-form-item>
      
      <div v-if="form.mode === 'static'">
        <el-form-item label="IP 地址">
          <el-input v-model="form.ip" placeholder="192.168.1.100"></el-input>
        </el-form-item>
        <el-form-item label="子网掩码">
          <el-input v-model="form.netmask" placeholder="255.255.255.0"></el-input>
        </el-form-item>
        <el-form-item label="默认网关">
          <el-input v-model="form.gateway" placeholder="192.168.1.1"></el-input>
        </el-form-item>
        <el-form-item label="DNS 服务器">
          <el-input v-model="form.dns" placeholder="8.8.8.8, 114.114.114.114"></el-input>
        </el-form-item>
      </div>

      <el-form-item>
        <el-button type="primary" @click="saveNetwork">保存并应用</el-button>
        <p style="color: #e6a23c; margin-left: 15px; font-size: 12px;">注：修改网络设置可能会导致当前连接断开，需要使用新IP重新访问此页面。</p>
      </el-form-item>
    </el-form>
  </el-card>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

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
    ElMessage.error('无法获取网络配置')
  }
  loading.value = false
}

const saveNetwork = () => {
  ElMessageBox.confirm(
    '应用网络设置可能导致当前连接断开，是否继续？',
    '提示',
    { confirmButtonText: '确定', cancelButtonText: '取消', type: 'warning' }
  ).then(async () => {
    loading.value = true
    try {
      await axios.post('/api/network', form.value)
      ElMessage.success('配置已下发，请在设备重启或网络重启后使用新IP访问。')
    } catch (e) {
      ElMessage.error('保存失败')
    }
    loading.value = false
  }).catch(() => {})
}

onMounted(() => {
  fetchNetwork()
})
</script>
