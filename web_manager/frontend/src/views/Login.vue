<template>
  <div class="login-container">
    <div class="lang-switch-login">
      <el-dropdown @command="handleLangCommand">
        <span class="lang-link">
          <el-icon><MagicStick /></el-icon>
          {{ currentLangName }}
        </span>
        <template #dropdown>
          <el-dropdown-menu>
            <el-dropdown-item command="zh">中文</el-dropdown-item>
            <el-dropdown-item command="en">English</el-dropdown-item>
          </el-dropdown-menu>
        </template>
      </el-dropdown>
    </div>
    <el-card class="login-card">
      <template #header>
        <div class="login-header">
          <img src="/image/logo.png" alt="Logo" class="login-logo" />
          <h2>{{ $t('login.title') }}</h2>
        </div>
      </template>
      <el-form :model="loginForm" :rules="loginRules" ref="loginFormRef" label-position="top">
        <el-form-item :label="$t('login.username')" prop="username">
          <el-input v-model="loginForm.username" :placeholder="$t('login.usernamePlaceholder')" prefix-icon="User" />
        </el-form-item>
        <el-form-item :label="$t('login.password')" prop="password">
          <el-input v-model="loginForm.password" type="password" :placeholder="$t('login.passwordPlaceholder')" prefix-icon="Lock" show-password @keyup.enter="handleLogin" />
        </el-form-item>
        <el-form-item>
          <el-button type="primary" :loading="loading" style="width: 100%" @click="handleLogin">{{ $t('login.loginButton') }}</el-button>
        </el-form-item>
      </el-form>
    </el-card>
  </div>
</template>

<script setup>
import { ref, reactive, computed } from 'vue'
import { useRouter } from 'vue-router'
import { useI18n } from 'vue-i18n'
import { ElMessage } from 'element-plus'
import { User, Lock, MagicStick } from '@element-plus/icons-vue'
import axios from 'axios'

const { t, locale } = useI18n()
const router = useRouter()
const loginFormRef = ref(null)
const loading = ref(false)

const currentLangName = computed(() => {
  return locale.value === 'zh' ? '中文' : 'English'
})

const handleLangCommand = (lang) => {
  locale.value = lang
  localStorage.setItem('lang', lang)
}

const loginForm = reactive({
  username: '',
  password: ''
})

const loginRules = computed(() => ({
  username: [{ required: true, message: t('login.usernamePlaceholder'), trigger: 'blur' }],
  password: [{ required: true, message: t('login.passwordPlaceholder'), trigger: 'blur' }]
}))

const handleLogin = async () => {
  if (!loginFormRef.value) return
  
  await loginFormRef.value.validate(async (valid) => {
    if (valid) {
      loading.value = true
      try {
        const res = await axios.post('/api/login', loginForm)
        if (res.data.token) {
          localStorage.setItem('token', res.data.token)
          localStorage.setItem('username', res.data.username)
          ElMessage.success(t('login.loginSuccess'))
          router.push('/dashboard')
        }
      } catch (err) {
        ElMessage.error(err.response?.data?.message || t('login.loginFailed'))
      } finally {
        loading.value = false
      }
    }
  })
}
</script>

<style scoped>
.login-container {
  height: 100vh;
  display: flex;
  justify-content: center;
  align-items: center;
  background-color: #f5f7fa;
  position: relative;
}

.lang-switch-login {
  position: absolute;
  top: 20px;
  right: 20px;
}

.lang-link {
  cursor: pointer;
  display: flex;
  align-items: center;
  gap: 5px;
  color: #606266;
  font-size: 14px;
}

.lang-link:hover {
  color: #409eff;
}

.login-card {
  width: 400px;
}

.login-header {
  display: flex;
  flex-direction: column;
  align-items: center;
}

.login-logo {
  width: 80px;
  height: 80px;
  margin-bottom: 20px;
}

.login-header h2 {
  margin: 0;
  color: #409eff;
}
</style>
