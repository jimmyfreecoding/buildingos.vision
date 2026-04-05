<template>
  <div class="local-model-container">
    <el-card class="box-card status-card">
      <template #header>
        <div class="card-header">
          <span>Gemma 服务状态</span>
          <el-tag :type="statusType" effect="dark" class="status-tag">
            {{ gemmaStatus }}
          </el-tag>
        </div>
      </template>
      <div class="status-details">
        <p>本地大语言模型与多模态模型服务（Gemma 4 E2B）。当前通过 llama.cpp 提供推理服务接口。</p>
        <el-button type="primary" :icon="Refresh" @click="fetchStatus" :loading="loadingStatus">
          刷新状态
        </el-button>
      </div>
    </el-card>

    <el-card class="box-card inference-card">
      <template #header>
        <div class="card-header">
          <span>图像推理验证 (Image Inference)</span>
        </div>
      </template>
      <el-form :model="inferForm" label-width="120px" @submit.prevent>
        <el-form-item label="推理提示词模版">
          <el-select v-model="inferForm.promptTemplate" placeholder="请选择提示词模版" style="width: 100%" @change="onTemplateChange">
            <el-option label="基础描述 (Describe this image)" value="Describe this image in detail." />
            <el-option label="抽烟检测验证" value="Is there a person smoking in this image? Answer 'Yes' or 'No' and provide reasons." />
            <el-option label="安全帽佩戴验证" value="Are all people in the image wearing safety helmets? Detail any violations." />
            <el-option label="自定义 (Custom)" value="custom" />
          </el-select>
        </el-form-item>
        
        <el-form-item label="提示词内容">
          <el-input 
            type="textarea" 
            v-model="inferForm.prompt" 
            :rows="3" 
            placeholder="输入您要询问的提示词内容..."
            :disabled="inferForm.promptTemplate !== 'custom'"
          />
        </el-form-item>

        <el-form-item label="上传图像">
          <el-upload
            class="avatar-uploader"
            action="#"
            :show-file-list="false"
            :auto-upload="false"
            :on-change="handleImageChange"
            accept="image/*"
          >
            <img v-if="inferForm.imageUrl" :src="inferForm.imageUrl" class="avatar" />
            <el-icon v-else class="avatar-uploader-icon"><Plus /></el-icon>
          </el-upload>
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="submitInference" :loading="inferring" :disabled="!inferForm.imageUrl">
            开始推理
          </el-button>
        </el-form-item>
      </el-form>
      
      <div class="result-section" v-if="inferResult || inferError">
        <el-divider>推理结果</el-divider>
        <el-alert v-if="inferError" :title="inferError" type="error" show-icon :closable="false" />
        <div v-else class="result-box">
          <pre>{{ inferResult }}</pre>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Refresh, Plus } from '@element-plus/icons-vue'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const gemmaStatus = ref('Unknown')
const loadingStatus = ref(false)

const inferring = ref(false)
const inferResult = ref('')
const inferError = ref('')

const inferForm = ref({
  promptTemplate: 'Describe this image in detail.',
  prompt: 'Describe this image in detail.',
  imageUrl: '',
  imageBase64: ''
})

const statusType = ref('info')

const fetchStatus = async () => {
  loadingStatus.value = true
  try {
    const res = await axios.get('/api/gemma/status')
    gemmaStatus.value = res.data.status
    if (gemmaStatus.value === 'Running') {
      statusType.value = 'success'
    } else if (gemmaStatus.value === 'Error') {
      statusType.value = 'danger'
    } else {
      statusType.value = 'warning'
    }
  } catch (err) {
    gemmaStatus.value = 'Offline'
    statusType.value = 'danger'
  } finally {
    loadingStatus.value = false
  }
}

const onTemplateChange = (val) => {
  if (val !== 'custom') {
    inferForm.value.prompt = val
  } else {
    inferForm.value.prompt = ''
  }
}

const handleImageChange = (file) => {
  const rawFile = file.raw
  if (!rawFile.type.startsWith('image/')) {
    ElMessage.error('只能上传图片文件!')
    return false
  }
  
  // Create a local URL for preview
  inferForm.value.imageUrl = URL.createObjectURL(rawFile)
  
  // Convert to Base64 for API
  const reader = new FileReader()
  reader.readAsDataURL(rawFile)
  reader.onload = () => {
    inferForm.value.imageBase64 = reader.result
  }
}

const submitInference = async () => {
  if (!inferForm.value.imageBase64) {
    ElMessage.warning('请先上传图片')
    return
  }
  if (!inferForm.value.prompt) {
    ElMessage.warning('提示词不能为空')
    return
  }

  inferring.value = true
  inferResult.value = ''
  inferError.value = ''

  try {
    const res = await axios.post('/api/gemma/infer', {
      image: inferForm.value.imageBase64,
      prompt: inferForm.value.prompt
    })
    
    if (res.data.error) {
      inferError.value = res.data.error + (res.data.details ? `: ${res.data.details}` : '')
    } else {
      inferResult.value = res.data.result
    }
  } catch (err) {
    inferError.value = err.response?.data?.error || err.message || '推理请求失败'
  } finally {
    inferring.value = false
  }
}

onMounted(() => {
  fetchStatus()
})
</script>

<style scoped>
.local-model-container {
  padding: 20px;
}
.box-card {
  margin-bottom: 20px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.status-tag {
  font-size: 14px;
  padding: 4px 10px;
}
.status-details {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.avatar-uploader .el-upload {
  border: 1px dashed var(--el-border-color);
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  transition: var(--el-transition-duration-fast);
}

.avatar-uploader .el-upload:hover {
  border-color: var(--el-color-primary);
}

.el-icon.avatar-uploader-icon {
  font-size: 28px;
  color: #8c939d;
  width: 178px;
  height: 178px;
  text-align: center;
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
}

.avatar {
  width: 178px;
  height: 178px;
  display: block;
  object-fit: cover;
  border-radius: 6px;
}

.result-section {
  margin-top: 30px;
}

.result-box {
  background-color: #f4f4f5;
  padding: 15px;
  border-radius: 4px;
  border: 1px solid #e9e9eb;
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: monospace;
  line-height: 1.5;
}
</style>
