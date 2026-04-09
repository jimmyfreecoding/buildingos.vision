<template>
  <div v-if="$route.name === 'Login'">
    <router-view></router-view>
  </div>
  <el-container v-else class="layout-container" v-loading.fullscreen.lock="isRebooting" :element-loading-text="$t('common.rebooting')">
    <el-aside :width="isCollapse ? '64px' : '220px'" class="sidebar-container">
      <div class="logo-container" :class="{ 'collapsed': isCollapse }">
        <img src="/image/logo.png" alt="Logo" class="logo-img" />
        <h3 v-if="!isCollapse" class="logo-text">{{ $t('header.title') }}</h3>
      </div>
      <el-menu
        active-text-color="#409eff"
        background-color="#304156"
        text-color="#bfcbd9"
        router
        :default-active="$route.path"
        :collapse="isCollapse"
        class="el-menu-vertical"
      >
        <el-menu-item index="/dashboard">
          <el-icon><Monitor /></el-icon>
          <template #title>{{ $t('menu.dashboard') }}</template>
        </el-menu-item>
        <el-menu-item index="/ai-monitor">
          <el-icon><DataLine /></el-icon>
          <template #title>{{ $t('menu.aiMonitor') }}</template>
        </el-menu-item>
        <el-menu-item index="/cameras">
          <el-icon><VideoCamera /></el-icon>
          <template #title>{{ $t('menu.cameras') }}</template>
        </el-menu-item>
        <el-menu-item index="/ai-params">
          <el-icon><Setting /></el-icon>
          <template #title>{{ $t('menu.aiParams') }}</template>
        </el-menu-item>
        <el-menu-item index="/occupancy-logs">
          <el-icon><Document /></el-icon>
          <template #title>{{ $t('menu.occupancyLogs') }}</template>
        </el-menu-item>
        <el-menu-item index="/network">
          <el-icon><Connection /></el-icon>
          <template #title>{{ $t('menu.network') }}</template>
        </el-menu-item>
        <el-menu-item index="/local-model">
          <el-icon><Cpu /></el-icon>
          <template #title>{{ $t('menu.localModel') }}</template>
        </el-menu-item>
        <el-menu-item @click="handleReboot">
          <el-icon><SwitchButton /></el-icon>
          <template #title>{{ $t('menu.reboot') }}</template>
        </el-menu-item>
      </el-menu>
    </el-aside>
    
    <el-container>
      <el-header class="header-container">
        <div class="header-left">
          <el-button type="text" @click="isCollapse = !isCollapse">
            <el-icon :size="20">
              <Expand v-if="isCollapse" />
              <Fold v-else />
            </el-icon>
          </el-button>
        </div>
        <div class="header-right">
          <!-- Theme Switch -->
          <div class="header-item">
            <el-switch
              v-model="isDark"
              inline-prompt
              :active-icon="Moon"
              :inactive-icon="Sunny"
              @change="toggleTheme"
            />
          </div>
          <!-- Language Switch -->
          <div class="header-item">
            <el-dropdown @command="handleLangCommand">
              <span class="lang-dropdown">
                <el-icon :size="18"><MagicStick /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="zh">中文</el-dropdown-item>
                  <el-dropdown-item command="en">English</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
          <!-- User Profile -->
          <div class="header-item">
            <el-dropdown @command="handleUserCommand">
              <span class="user-dropdown">
                {{ username }}
                <el-icon class="el-icon--right"><arrow-down /></el-icon>
              </span>
              <template #dropdown>
                <el-dropdown-menu>
                  <el-dropdown-item command="logout">{{ $t('header.logout') }}</el-dropdown-item>
                </el-dropdown-menu>
              </template>
            </el-dropdown>
          </div>
        </div>
      </el-header>
      
      <el-main>
        <router-view></router-view>
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { 
  VideoCamera, Setting, Connection, SwitchButton, Monitor, 
  DataLine, Document, Cpu, Expand, Fold, Moon, Sunny, MagicStick, ArrowDown 
} from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage, ElMessageBox } from 'element-plus'

const { t, locale } = useI18n()
const router = useRouter()

const isRebooting = ref(false)
const isCollapse = ref(false)
const isDark = ref(localStorage.getItem('theme') === 'dark')
const username = ref(localStorage.getItem('username') || 'Admin')

const toggleTheme = (val) => {
  const html = document.documentElement
  if (val) {
    html.classList.add('dark')
    localStorage.setItem('theme', 'dark')
  } else {
    html.classList.remove('dark')
    localStorage.setItem('theme', 'light')
  }
}

const handleLangCommand = (lang) => {
  locale.value = lang
  localStorage.setItem('lang', lang)
  ElMessage.success(lang === 'zh' ? '语言已切换为中文' : 'Language switched to English')
}

const handleUserCommand = (command) => {
  if (command === 'logout') {
    localStorage.removeItem('token')
    localStorage.removeItem('username')
    router.push('/login')
  }
}

const handleReboot = () => {
  ElMessageBox.confirm(
    t('common.rebootConfirm'),
    t('common.warning'),
    {
      confirmButtonText: t('common.confirm'),
      cancelButtonText: t('common.cancel'),
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
  setTimeout(() => {
    const timer = setInterval(async () => {
      try {
        await axios.get('/api/ping', { timeout: 2000 })
        clearInterval(timer)
        isRebooting.value = false
        ElMessage.success('设备已重新连接！')
        window.location.reload()
      } catch (e) {
        console.log('Waiting for device to boot...')
      }
    }, 5000)
  }, 10000)
}

onMounted(() => {
  // Initialize theme
  toggleTheme(isDark.value)
})
</script>

<style scoped>
.layout-container {
  height: 100vh;
}

.sidebar-container {
  background-color: #304156;
  transition: width 0.3s;
  display: flex;
  flex-direction: column;
}

.logo-container {
  height: 60px;
  display: flex;
  align-items: center;
  padding: 0 16px;
  background-color: #2b2f3a;
  overflow: hidden;
  transition: all 0.3s;
}

.logo-container.collapsed {
  padding: 0;
  justify-content: center;
}

.logo-img {
  width: 32px;
  height: 32px;
  flex-shrink: 0;
}

.logo-text {
  margin: 0 0 0 12px;
  color: white;
  font-weight: 600;
  font-size: 16px;
  white-space: nowrap;
}

.el-menu-vertical {
  border-right: none;
  flex: 1;
}

.header-container {
  display: flex;
  justify-content: space-between;
  align-items: center;
  border-bottom: 1px solid var(--el-border-color-lighter);
  background-color: var(--el-bg-color);
  padding: 0 20px;
}

.header-right {
  display: flex;
  align-items: center;
}

.header-item {
  margin-left: 20px;
  display: flex;
  align-items: center;
}

.lang-dropdown, .user-dropdown {
  cursor: pointer;
  display: flex;
  align-items: center;
  color: var(--el-text-color-primary);
}

.lang-dropdown:hover, .user-dropdown:hover {
  color: var(--el-color-primary);
}
</style>

<style>
/* Global dark mode overrides if needed */
.dark body {
  background-color: #1a1a1a;
}
</style>
