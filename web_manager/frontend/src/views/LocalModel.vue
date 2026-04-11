<template>
  <div class="local-model-container">
    <el-card class="box-card inference-card">
      <template #header>
        <div class="card-header">
          <span>{{ $t('localModel.title') }}</span>
        </div>
      </template>
      <el-form :model="inferForm" label-width="120px" @submit.prevent>
        <el-form-item :label="$t('localModel.template')">
          <el-select v-model="inferForm.promptTemplate" :placeholder="$t('localModel.templatePlaceholder')" style="width: 100%" @change="onTemplateChange">
            <el-option :label="$t('localModel.descTemplate')" value="Describe this image in detail. Please reply in Chinese. (请详细描述这张图片，用中文回复)" />
            <el-option :label="$t('localModel.smokingTemplate')" value="Is there a person smoking in this image? Answer 'Yes' or 'No' and provide reasons. Please reply in Chinese. (图片中是否有人在抽烟？用中文回复是或否，并说明理由。)" />
            <el-option :label="$t('localModel.helmetTemplate')" value="Are all people in the image wearing safety helmets? Detail any violations. Please reply in Chinese. (图片中所有人是否都佩戴了安全帽？请用中文详细说明违规情况。)" />
            <el-option :label="$t('localModel.presenceTemplate')" value="检测图片中是否有活人存在，仔细鉴别头肩和肢体等人体要输，如果有人回答YES，并且告知在什么位置。没有则回答NO" />
            <el-option :label="$t('localModel.customTemplate')" value="custom" />
          </el-select>
        </el-form-item>
        
        <el-form-item :label="$t('localModel.promptLabel')">
          <el-input 
            type="textarea" 
            v-model="inferForm.prompt" 
            :rows="3" 
            :placeholder="$t('localModel.promptPlaceholder')"
            :disabled="inferForm.promptTemplate !== 'custom'"
          />
        </el-form-item>

        <el-form-item :label="$t('localModel.uploadImage')">
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

        <el-form-item :label="$t('localModel.enableThinking')">
          <el-switch
            v-model="inferForm.enableThinking"
            :active-text="$t('localModel.thinkingYes')"
            :inactive-text="$t('localModel.thinkingNo')"
          />
        </el-form-item>

        <el-form-item>
          <el-button type="primary" @click="submitInference" :loading="inferring" :disabled="!inferForm.imageUrl">
            {{ $t('localModel.startInference') }}
          </el-button>
        </el-form-item>
      </el-form>
      
      <div class="result-section" v-if="inferResult || inferError || inferring">
        <el-divider>{{ $t('localModel.resultDivider') }}</el-divider>
        <el-alert v-if="inferError" :title="inferError" type="error" show-icon :closable="false" />
        <div v-else-if="inferring" class="loading-state">
          <el-skeleton :rows="5" animated />
        </div>
        <div v-else class="result-container">
          <!-- Final Result (Categorized) -->
          <div :class="['result-box', inferResult === 'YES' ? 'occupied' : inferResult === 'NO' ? 'empty' : '']">
            <div class="result-label">RESULT: {{ inferResult }}</div>
            <div class="analysis-content">{{ inferReasoning }}</div>
          </div>

          <!-- Raw LLM Response -->
          <el-collapse class="raw-collapse">
            <el-collapse-item name="1">
              <template #title>
                <el-icon class="header-icon"><Cpu /></el-icon> LLM Raw JSON Response
              </template>
              <div class="raw-content">
                <pre>{{ inferRawResponse }}</pre>
              </div>
            </el-collapse-item>
          </el-collapse>

          <!-- Metrics Footer -->
          <div v-if="inferMetrics" class="metrics-footer">
            <div class="metric-item">
              <el-icon><Timer /></el-icon>
              <span>{{ inferMetrics.durationStr }}</span>
            </div>
            <div class="metric-item">
              <el-icon><Document /></el-icon>
              <span>{{ inferMetrics.contextStr }}</span>
            </div>
            <div class="metric-item">
              <el-icon><Aim /></el-icon>
              <span>{{ inferMetrics.outputStr }}</span>
            </div>
            <div class="metric-item">
              <el-icon><Odometer /></el-icon>
              <span>{{ inferMetrics.speedStr }}</span>
            </div>
          </div>
        </div>
      </div>
    </el-card>
  </div>
</template>

<script setup>
import { ref, onMounted } from 'vue'
import { Plus, Cpu, Aim, Odometer, Document, Timer } from '@element-plus/icons-vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'
import { ElMessage } from 'element-plus'

const { t } = useI18n()
const inferring = ref(false)
const inferResult = ref('')
const inferReasoning = ref('')
const inferRawResponse = ref('')
const inferMetrics = ref(null)
const inferError = ref('')

const inferForm = ref({
  promptTemplate: 'Describe this image in detail. Please reply in Chinese. (请详细描述这张图片，用中文回复)',
  prompt: 'Describe this image in detail. Please reply in Chinese. (请详细描述这张图片，用中文回复)',
  imageUrl: '',
  imageBase64: '',
  enableThinking: false
})

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
    ElMessage.error(t('localModel.onlyImageError'))
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
    ElMessage.warning(t('localModel.uploadRequired'))
    return
  }
  if (!inferForm.value.prompt) {
    ElMessage.warning(t('localModel.promptRequired'))
    return
  }

  inferring.value = true
  inferResult.value = ''
  inferReasoning.value = ''
  inferRawResponse.value = ''
  inferMetrics.value = null
  inferError.value = ''

  try {
    const res = await axios.post('/api/gemma/infer', {
      image: inferForm.value.imageBase64,
      prompt: inferForm.value.prompt,
      enableThinking: inferForm.value.enableThinking
    })
    
    if (res.data.error) {
      inferError.value = res.data.error + (res.data.details ? `: ${res.data.details}` : '')
    } else {
      inferResult.value = res.data.result
      inferReasoning.value = res.data.reasoning
      inferRawResponse.value = res.data.llm_response
      
      // Calculate metrics if available
      if (res.data.usage && res.data.timings) {
        const promptTokens = res.data.usage.prompt_tokens || 0
        const totalContext = 4096 // Typical context size, adjust if needed
        const contextPercent = Math.round((promptTokens / totalContext) * 100)
        
        const predictedTokens = res.data.timings.predicted_n || 0
        const tokensPerSecond = (res.data.timings.predicted_per_second || 0).toFixed(1)
        const durationSec = ((res.data.durationMs || 0) / 1000).toFixed(2)
        
        inferMetrics.value = {
          contextStr: `Context: ${promptTokens}/${totalContext} (${contextPercent}%)`,
          outputStr: `Output: ${predictedTokens}/∞`,
          speedStr: `${tokensPerSecond} t/s`,
          durationStr: `Time: ${durationSec} s`
        }
      }
    }
  } catch (err) {
    inferError.value = err.response?.data?.error || err.message || '推理请求失败'
  } finally {
    inferring.value = false
  }
}

onMounted(() => {
  // fetchStatus() removed
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

.result-container {
  display: flex;
  flex-direction: column;
  gap: 15px;
}

.reasoning-collapse {
  border: 1px solid var(--el-border-color-light);
  border-radius: 4px;
}

.reasoning-collapse :deep(.el-collapse-item__header) {
  padding-left: 15px;
  background-color: #fafafa;
  color: #606266;
  font-weight: bold;
}

.header-icon {
  margin-right: 8px;
  font-size: 16px;
}

.reasoning-content {
  padding: 15px;
  background-color: #fff;
  border-top: 1px solid var(--el-border-color-lighter);
}

.reasoning-content pre {
  margin: 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  color: #909399;
  font-family: monospace;
  line-height: 1.5;
  font-size: 13px;
}

.result-box {
  background-color: #f0f9eb;
  padding: 15px;
  border-radius: 4px;
  border: 1px solid #e1f3d8;
  color: #303133;
  line-height: 1.6;
}

.result-box.occupied {
  background-color: #fef0f0;
  border-color: #fde2e2;
  color: #f56c6c;
}

.result-box.empty {
  background-color: #f0f9eb;
  border-color: #e1f3d8;
  color: #67c23a;
}

.result-label {
  font-weight: bold;
  font-size: 16px;
  margin-bottom: 8px;
  border-bottom: 1px dashed rgba(0,0,0,0.1);
  padding-bottom: 5px;
}

.analysis-content {
  font-size: 14px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.raw-collapse {
  margin-top: 10px;
}

.raw-content {
  padding: 10px;
  background-color: #303133;
  color: #fff;
  border-radius: 4px;
}

.raw-content pre {
  margin: 0;
  font-size: 11px;
  white-space: pre-wrap;
  word-wrap: break-word;
}

.metrics-footer {
  display: flex;
  justify-content: flex-start;
  gap: 30px;
  margin-top: 5px;
  padding-top: 10px;
  border-top: 1px solid #ebeef5;
  color: #909399;
  font-size: 12px;
}

.metric-item {
  display: flex;
  align-items: center;
  gap: 5px;
}

.loading-state {
  padding: 20px;
}
</style>
