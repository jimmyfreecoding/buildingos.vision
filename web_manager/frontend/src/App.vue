<template>
  <el-container class="layout-container" v-loading.fullscreen.lock="isRebooting" element-loading-text="系统重启中，请等待设备重新连接...">
    <el-aside width="200px" style="background-color: #304156;">
      <div class="logo">
        <h3 style="color: white; text-align: center;">边缘计算网关</h3>
      </div>
      <el-menu
        active-text-color="#409eff"
        background-color="#304156"
        text-color="#bfcbd9"
        router
        :default-active="$route.path"
      >
        <el-menu-item index="/cameras">
          <el-icon><VideoCamera /></el-icon>
          <span>流媒体配置</span>
        </el-menu-item>
        <el-menu-item index="/ai-params">
          <el-icon><Setting /></el-icon>
          <span>AI 算法参数</span>
        </el-menu-item>
        <el-menu-item index="/network">
          <el-icon><Connection /></el-icon>
          <span>网络设置</span>
        </el-menu-item>
        <el-menu-item @click="handleReboot">
          <el-icon><SwitchButton /></el-icon>
          <span>重启设备</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header style="text-align: right; font-size: 12px; border-bottom: 1px solid #eee;">
        <span style="line-height: 60px;">Admin</span>
      </el-header>
      
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref } from 'vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

const isRebooting = ref(false)

const handleReboot = () => {
  ElMessageBox.confirm(
    '设备即将重启，视频流和 AI 分析将暂时中断，是否继续？',
    '警告',
    {
      confirmButtonText: '确认重启',
      cancelButtonText: '取消',
      type: 'warning',
    }
  ).then(async () => {
    try {
      await axios.post('/api/system/reboot')
      isRebooting.value = true
      startPingLoop()
    } catch (e) {
      ElMessage.error('发送重启指令失败')
    }
  }).catch(() => {})
}

const startPingLoop = () => {
  // 等待10秒后开始轮询，因为刚下发指令时可能还能ping通
  setTimeout(() => {
    const timer = setInterval(async () => {
      try {
        await axios.get('/api/ping', { timeout: 2000 })
        // 如果能通，说明重启完成
        clearInterval(timer)
        isRebooting.value = false
        ElMessage.success('设备已重新连接！')
        window.location.reload()
      } catch (e) {
        // 报错是正常的，说明还在重启中
        console.log('Waiting for device to boot...')
      }
    }, 5000)
  }, 10000)
}
</script>

<style scoped>
.layout-container {
  height: 100vh;
}
</style>
