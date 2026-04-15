<template>
  <el-card class="box-card heatmap-card">
    <template #header>
      <div class="card-header">
        <span><el-icon><Calendar /></el-icon> {{ $t('logs.heatmapTitle') }}</span>
        <div class="header-buttons">
          <el-button @click="dialogTestVisible = true" type="warning" plain size="small" :icon="Picture">{{ $t('logs.testButton') }}</el-button>
          <el-button @click="fetchLogs" type="primary" plain size="small" :icon="Search" :loading="loading">{{ $t('logs.refreshButton') }}</el-button>
        </div>
      </div>
    </template>

    <div class="filter-section">
      <el-select v-model="selectedArea" :placeholder="$t('logs.selectAreaPlaceholder')" style="width: 250px; margin-right: 15px;" @change="handleFilterChange" filterable>
        <el-option v-for="area in uniqueAreas" :key="area" :label="area" :value="area"></el-option>
      </el-select>
      
      <el-date-picker
        v-model="dateRange"
        type="daterange"
        :range-separator="$t('logs.dateRangeSeparator')"
        :start-placeholder="$t('logs.startDatePlaceholder')"
        :end-placeholder="$t('logs.endDatePlaceholder')"
        value-format="YYYY-MM-DD"
        @change="handleFilterChange"
        style="width: 300px; margin-right: 15px;"
      />
      
      <!-- Auto refresh switch -->
      <el-switch
        v-model="autoRefresh"
        :active-text="$t('logs.autoRefreshOn')"
        :inactive-text="$t('logs.autoRefreshOff')"
        @change="toggleAutoRefresh"
        style="margin-right: 15px;"
      />

      <div class="legend">
        <span style="margin-right: 10px;">{{ $t('logs.legendNoRecord') }}</span>
        <ul class="legend-colors">
          <li class="color-level-null"></li>
        </ul>
        <span style="margin-right: 10px;">{{ $t('logs.legendEmpty') }}</span>
        <ul class="legend-colors">
          <li class="color-level-0"></li>
          <li class="color-level-1"></li>
          <li class="color-level-2"></li>
          <li class="color-level-3"></li>
          <li class="color-level-4"></li>
        </ul>
        <span>{{ $t('logs.legendOccupied') }}</span>
      </div>
    </div>

    <div v-loading="loading" class="heatmaps-wrapper">
      <div v-if="!selectedArea" class="no-data">
        <el-empty :description="$t('logs.selectAreaTip')"></el-empty>
      </div>
      <div v-else-if="displayDays.length === 0" class="no-data">
        <el-empty :description="$t('logs.noDataTip')"></el-empty>
      </div>
      
      <!-- 左右两块布局，一行两天 -->
      <el-row :gutter="40" v-else>
        <el-col :span="12" v-for="dayData in displayDays" :key="dayData.date" style="margin-bottom: 40px;">
          <div class="day-heatmap">
            <div class="day-header">
              <h4 class="day-title">{{ dayData.date }}</h4>
              <el-button 
                v-if="summaries[dayData.date]" 
                type="primary" 
                link 
                size="small" 
                class="summary-link"
                @click="openSummary(dayData.date)"
              >
                <el-icon><Document /></el-icon> {{ $t('logs.viewDailySummary') }}
              </el-button>
            </div>
            <div class="heatmap-container">
              <!-- Y-axis (Minutes): 50m at top, 0m at bottom -->
              <div class="y-axis-wrapper">
                <div class="y-axis">
                  <div class="y-label" v-for="m in 6" :key="m">
                    <span>{{ (6 - m) * 10 }}m</span>
                  </div>
                </div>
                <div class="x-axis-placeholder"></div>
              </div>
              
              <!-- Grid -->
              <div class="grid-content">
                 <div class="grid-columns">
                    <div class="column" v-for="hour in 24" :key="hour">
                       <!-- 反转 minuteIdx 渲染顺序，使 DOM 节点也自下而上排列 0m, 10m...50m -->
                       <el-tooltip
                          v-for="minuteIdx in 6" :key="minuteIdx"
                          placement="top"
                          :content="getTooltip(dayData, hour-1, 6 - minuteIdx)"
                          :show-after="200"
                       >
                         <div 
                            class="cell" 
                            :class="getCellIntensityClass(dayData, hour-1, 6 - minuteIdx)"
                            @click="openDetail(dayData, hour-1, 6 - minuteIdx)"
                         ></div>
                       </el-tooltip>
                    </div>
                 </div>
                 <!-- X-axis (Hours) -->
                 <div class="x-axis">
                   <div class="x-label" v-for="hour in 24" :key="hour">{{ (hour - 1).toString().padStart(2, '0') }}:00</div>
                 </div>
              </div>
            </div>
          </div>
        </el-col>
      </el-row>
    </div>

    <!-- AI 日报总结弹窗 -->
    <el-dialog v-model="summaryDialogVisible" :title="$t('logs.summaryDialogTitle', { date: selectedDate })" width="700px">
      <div v-if="selectedSummary" class="summary-content">
        <el-descriptions :column="1" border size="small" class="summary-stats">
          <el-descriptions-item :label="$t('logs.summaryGeneratedAt')">
            {{ formatTime(selectedSummary.generated_at) }}
          </el-descriptions-item>
          <el-descriptions-item :label="$t('logs.summaryOverview')">
            <el-tag size="small">{{ $t('logs.summaryTotalSamples') }}: {{ selectedSummary.stats.summary_stats.total_samples }}</el-tag>
            <el-tag size="small" type="success" style="margin-left: 5px;">{{ $t('logs.summaryLvl1Direct') }}: {{ selectedSummary.stats.summary_stats.lvl1_direct_confirm }}</el-tag>
            <el-tag size="small" type="warning" style="margin-left: 5px;">{{ $t('logs.summaryLvl2Reviews') }}: {{ selectedSummary.stats.summary_stats.lvl2_gemma_reviews }}</el-tag>
          </el-descriptions-item>
          <el-descriptions-item :label="$t('logs.summaryReviewResult')">
            <span style="color: #67C23A">{{ $t('logs.summaryOccupiedConfirmed') }}: {{ selectedSummary.stats.summary_stats.lvl2_gemma_confirmed }}</span>
            <span style="color: #F56C6C; margin-left: 15px;">{{ $t('logs.summaryFalseAlarmDenied') }}: {{ selectedSummary.stats.summary_stats.lvl2_gemma_denied }}</span>
          </el-descriptions-item>
        </el-descriptions>

        <div class="summary-text-box">
          <div class="summary-label">{{ $t('logs.summaryGemmaReport') }}</div>
          <div class="summary-markdown" v-html="renderMarkdown(selectedSummary.summary)"></div>
        </div>

        <div v-if="selectedSummary.stats.areas[selectedArea]?.lvl2_details?.length > 0" class="summary-details">
          <div class="summary-label">{{ $t('logs.summaryTimelineTitle', { area: selectedArea }) }}</div>
          <el-table :data="selectedSummary.stats.areas[selectedArea].lvl2_details" size="small" border stripe style="margin-top: 10px;">
            <el-table-column prop="time" :label="$t('logs.summaryTableTime')" width="180">
              <template #default="scope">
                {{ formatTime(scope.row.time) }}
              </template>
            </el-table-column>
            <el-table-column prop="res" :label="$t('logs.summaryTableResult')" width="100">
              <template #default="scope">
                <el-tag :type="scope.row.res === 'YES' ? 'success' : 'danger'" size="small">
                  {{ scope.row.res === 'YES' ? $t('logs.statusOccupied') : $t('logs.statusEmpty') }}
                </el-tag>
              </template>
            </el-table-column>
            <el-table-column prop="reason" :label="$t('logs.summaryTableChain')">
              <template #default="scope">
                <span style="font-size: 11px; color: #909399;">{{ scope.row.reason.join(' → ') }}</span>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </el-dialog>

    <!-- 详情弹窗 -->
    <el-dialog v-model="dialogVisible" :title="dialogTitle" width="1000px" top="5vh">
      <div v-if="dialogLogs.length === 0">
        <el-empty :description="$t('logs.noRecordTip')"></el-empty>
      </div>
      <el-timeline v-else>
        <el-timeline-item
          v-for="group in groupedLogs"
          :key="group.time"
          :timestamp="group.time"
          :type="group.logs.some(l => l.raw_payload?.result === 'occupied') ? 'success' : 'info'"
        >
          <el-card shadow="hover" class="log-detail-card">
            <div class="log-header">
              <el-tag :type="getGroupResultTagType(group)" size="small" effect="dark">
                {{ formatGroupResult(group) }}
              </el-tag>
              <span class="log-event">{{ $t('logs.statusUpdate') }}</span>
            </div>
            
            <div class="log-meta" style="margin-top: 10px; font-size: 13px; color: #606266;">
              <p><strong>{{ $t('logs.strategyChain') }}</strong> Object detection+Gemma</p>
              <p style="margin-top: 5px;">
                <strong>{{ $t('logs.finalDecision') }}</strong> 
                <el-popover placement="bottom" :title="$t('logs.decisionProcessTitle')" width="400" trigger="click">
                  <template #reference>
                    <el-link type="primary" :underline="false">1-minute sample ({{ $t('logs.clickToView') }})</el-link>
                  </template>
                  <div style="font-size: 13px;">
                    <p v-for="log in group.logs" :key="log.id" style="margin-bottom: 5px;">
                      <b><el-icon><VideoCamera /></el-icon> {{ log.camera_id }}:</b> 
                      <span v-if="log.raw_payload?.result === 'occupied'" style="color: #67C23A; margin-left: 5px;">{{ $t('logs.decidedOccupied') }}</span>
                      <span v-else style="color: #909399; margin-left: 5px;">{{ $t('logs.decidedEmpty') }}</span>
                      <span style="margin-left: 5px; color: #E6A23C;" v-if="log.raw_payload?.yolo_count > 0">({{ $t('logs.detectedCount', { count: log.raw_payload?.yolo_count }) }})</span>
                    </p>
                    <el-divider style="margin: 10px 0;"></el-divider>
                    <p><b>{{ $t('logs.areaSummaryResult') }}</b> 
                      <span v-if="group.logs.some(l => l.raw_payload?.result === 'occupied')" style="color: #67C23A; font-weight: bold;">{{ $t('logs.areaOccupied') }}</span>
                      <span v-else style="color: #909399; font-weight: bold;">{{ $t('logs.areaEmpty') }}</span>
                    </p>
                  </div>
                </el-popover>
              </p>
            </div>

            <div style="margin-top: 15px; font-size: 13px; font-weight: bold; color: #303133; margin-bottom: 10px;">
              {{ $t('logs.evidenceTitle') }}
            </div>
            
            <!-- 多摄像头左右排列 -->
            <el-row :gutter="15">
              <el-col :span="24 / Math.min(group.logs.length, 4)" v-for="log in group.logs" :key="log.id">
                <div class="camera-evidence">
                  <p style="font-weight: bold; margin-bottom: 5px; color: #409EFF; font-size: 13px;">
                    <el-icon><VideoCamera /></el-icon> {{ log.camera_id }}
                  </p>
                  <!-- 显示第一张图（也就是带有时间戳和红框的 annotated_frame） -->
                  <el-image 
                    v-if="log.images && log.images.length > 0"
                    :src="`${getImageUrl(log.images[0])}?t=${new Date().getTime()}`" 
                    :preview-src-list="log.images.map(i => getImageUrl(i))"
                    :initial-index="0"
                    fit="contain"
                    class="log-image"
                  />
                  <div class="evidence-chain" style="margin-top: 10px; font-size: 12px; color: #606266; background: #f5f7fa; padding: 8px; border-radius: 4px; min-height: 80px;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                      <b style="color: #303133;">{{ $t('logs.evidenceChainTitle') }}</b>
                      <el-link type="info" :underline="false" style="font-size: 11px;" @click="viewRawJson(log)">
                        [{{ $t('logs.viewRawJson') }}]
                      </el-link>
                    </div>
                    <ul style="padding-left: 20px; margin-top: 5px; margin-bottom: 0;">
                      <li v-for="(step, idx) in (log.raw_payload?.decision_chain || [$t('logs.noLogChain')])" :key="idx" style="margin-bottom: 3px;">
                        <span v-if="step.includes('Gemma 复核')">
                          {{ translateChainStep(step) }}
                          <el-link type="primary" size="small" @click="handleManualGemmaReview(log)" :loading="manualReviewing === log.id" style="margin-left: 5px; font-size: 11px;">
                            [{{ $t('logs.manualReviewButton') }}]
                          </el-link>
                        </span>
                        <span v-else>{{ translateChainStep(step) }}</span>
                      </li>
                    </ul>
                  </div>
                </div>
              </el-col>
            </el-row>
          </el-card>
        </el-timeline-item>
      </el-timeline>
    </el-dialog>

    <!-- 算法验证单图测试弹窗 -->
    <el-dialog v-model="dialogTestVisible" :title="$t('logs.testDialogTitle')" width="800px" destroy-on-close>
      <div class="test-container">
        <el-row :gutter="20">
          <el-col :span="10">
            <el-form :model="testForm" label-position="top">
              <el-form-item :label="$t('logs.uploadTestImage')">
                <el-upload
                  class="test-uploader"
                  action="#"
                  :show-file-list="false"
                  :auto-upload="false"
                  :on-change="handleTestImageChange"
                  accept="image/*"
                >
                  <img v-if="testForm.imageUrl" :src="testForm.imageUrl" class="test-preview-img" />
                  <div v-else class="test-uploader-placeholder">
                    <el-icon class="test-uploader-icon"><Plus /></el-icon>
                    <span>{{ $t('logs.clickToUpload') }}</span>
                  </div>
                </el-upload>
              </el-form-item>
              
              <el-form-item :label="$t('logs.confThres')">
                <el-slider v-model="testForm.conf_thres" :min="0.01" :max="0.99" :step="0.01" show-input />
                <div class="form-tip">{{ $t('logs.confThresTip') }}</div>
              </el-form-item>

              <el-form-item>
                <el-button type="primary" @click="submitTest" :loading="testing" :disabled="!testForm.imageBase64" style="width: 100%">
                  {{ $t('logs.startTestInference') }}
                </el-button>
              </el-form-item>
            </el-form>
          </el-col>
          
          <el-col :span="14">
            <div class="test-result-section">
              <div v-if="testing" class="test-loading">
                <el-skeleton :rows="8" animated />
              </div>
              <div v-else-if="testResult" class="test-result-content">
                <div class="result-title">{{ $t('logs.testVisualTitle') }}</div>
                <el-image 
                  :src="testResult.annotated_image" 
                  :preview-src-list="[testResult.annotated_image]"
                  fit="contain" 
                  class="test-result-img"
                />
                
                <div class="result-stats">
                  <el-tag size="small" type="info">{{ $t('logs.testDetector') }} {{ testResult.detector_source }}</el-tag>
                  <el-tag size="small" type="success" style="margin-left: 10px;">{{ $t('logs.testDetectedTargets', { count: testResult.results.length }) }}</el-tag>
                </div>

                <div class="result-list" style="margin-top: 15px;">
                  <el-table :data="testResult.results" size="small" border height="150">
                    <el-table-column prop="class_name" :label="$t('logs.testTableClass')" width="100" />
                    <el-table-column prop="conf" :label="$t('logs.testTableConf')" width="100">
                      <template #default="scope">
                        {{ (scope.row.conf * 100).toFixed(1) }}%
                      </template>
                    </el-table-column>
                    <el-table-column :label="$t('logs.testTableBbox')">
                      <template #default="scope">
                        {{ scope.row.bbox.join(', ') }}
                      </template>
                    </el-table-column>
                  </el-table>
                </div>
              </div>
              <el-empty v-else :description="$t('logs.testWaitingTip')"></el-empty>
            </div>
          </el-col>
        </el-row>
      </div>
    </el-dialog>
  </el-card>
</template>

<script setup>
import { ref, computed, onMounted } from 'vue'
import { useI18n } from 'vue-i18n'
import axios from 'axios'

const { t } = useI18n()
import { ElMessage, ElMessageBox, ElLoading } from 'element-plus'
import { Calendar, VideoCamera, Plus, Picture, Search, Document } from '@element-plus/icons-vue'
import { marked } from 'marked'

const loading = ref(false)
const allLogs = ref([])
const areaList = ref([]) // 存储场景列表
const selectedArea = ref('')
const dateRange = ref([])
const autoRefresh = ref(false)
let refreshInterval = null

const dialogVisible = ref(false)
const dialogTitle = ref('')
const dialogLogs = ref([])

const viewRawJson = (log) => {
  ElMessageBox.alert(
    `<pre style="background: #303133; color: #fff; padding: 15px; border-radius: 4px; font-size: 12px; overflow: auto; max-height: 500px;">${JSON.stringify(log, null, 2)}</pre>`,
    t('logs.rawJsonTitle'),
    {
      dangerouslyUseHTMLString: true,
      confirmButtonText: t('logs.close'),
      width: '700px'
    }
  )
}

// --- Summary State ---
const summaryDialogVisible = ref(false)
const summaries = ref({})
const selectedDate = ref('')
const selectedSummary = ref(null)

const openSummary = (date) => {
  selectedDate.value = date
  selectedSummary.value = summaries.value[date]
  summaryDialogVisible.value = true
}

const renderMarkdown = (text) => {
  if (!text) return ''
  return marked(text)
}

const fetchSummary = async (date) => {
  try {
    const res = await axios.get(`/api/occupancy/summary/${date}`)
    summaries.value[date] = res.data
  } catch (e) {
    // If not found, ignore
  }
}
// ----------------------

// --- Test Image State ---
const dialogTestVisible = ref(false)
const testing = ref(false)
const testResult = ref(null)
const testForm = ref({
  imageUrl: '',
  imageBase64: '',
  conf_thres: 0.25
})

const handleTestImageChange = (file) => {
  const rawFile = file.raw
  if (!rawFile.type.startsWith('image/')) {
    ElMessage.error('只能上传图片文件!')
    return false
  }
  
  testForm.value.imageUrl = URL.createObjectURL(rawFile)
  
  const reader = new FileReader()
  reader.readAsDataURL(rawFile)
  reader.onload = () => {
    testForm.value.imageBase64 = reader.result
  }
}

const submitTest = async () => {
  if (!testForm.value.imageBase64) return
  
  testing.value = true
  testResult.value = null
  
  try {
    const res = await axios.post('/api/ai/test', {
      image: testForm.value.imageBase64,
      conf_thres: testForm.value.conf_thres
    })
    testResult.value = res.data
  } catch (e) {
    ElMessage.error(e.response?.data?.error || t('logs.testRequestFailed'))
  } finally {
    testing.value = false
  }
}
// -------------------------

const manualReviewing = ref('')

const translateChainStep = (step) => {
  if (!step) return step
  
  // 1. Detector 检测到 X 个候选人员
  let match = step.match(/Detector 检测到 (\d+) 个候选人员/)
  if (match) return t('chain.detectorDetected', { count: match[1] })

  // 2. Detector 高置信度(X)直接确认有人
  match = step.match(/Detector 高置信度\(([\d.]+)\)直接确认有人/)
  if (match) return t('chain.detectorHighConf', { conf: match[1] })

  // 3. Gemma 二级裁决结果: YES/NO
  match = step.match(/Gemma 二级裁决结果: (\w+)/)
  if (match) return t('chain.gemmaL2Result', { res: match[1] })

  // 4. 固定短语匹配
  const directMap = {
    "Gemma 复核: 确认图中存在真实人员": "chain.gemmaConfirmed",
    "Gemma 复核: Detector漏报，但Gemma在全图中发现了人员": "chain.gemmaMissedButFound",
    "Gemma 复核: 否决 (认定疑似目标为误报/假人)": "chain.gemmaDenied",
    "Gemma 复核: 确认全图确实无人": "chain.gemmaConfirmedEmpty",
    "Gemma 响应异常，降级采信 Detector 结果: YES": "chain.gemmaExceptionYes",
    "Gemma 响应异常，降级采信 Detector 结果: NO": "chain.gemmaExceptionNo",
    "图像编码失败，降级采信 Detector": "chain.encodingFailed",
    "AI 引擎默认状态更新": "chain.defaultUpdate",
    "直接采信无日志": "logs.noLogChain"
  }

  return directMap[step] ? t(directMap[step]) : step
}

const handleManualGemmaReview = async (log) => {
  if (!log.images || log.images.length === 0) return
  
  manualReviewing.value = log.id
  const loadingInstance = ElLoading.service({
    lock: true,
    text: t('logs.manualReviewingLoading'),
    background: 'rgba(0, 0, 0, 0.7)',
  })

  try {
    // 1. 获取原始图片 (如果是[annotated, original]，则选第二个；否则选第一个)
    const imageUrl = getImageUrl(log.images[1] || log.images[0])
    
    // 2. 将图片转换为 Base64
    const response = await fetch(imageUrl)
    const blob = await response.blob()
    const reader = new FileReader()
    const base64Promise = new Promise((resolve) => {
      reader.onloadend = () => resolve(reader.result)
      reader.readAsDataURL(blob)
    })
    const imageBase64 = await base64Promise

    // 3. 调用后端复核接口
    const prompt = "检测图片中是否有活人存在，仔细鉴别头肩和肢体等人体要输，如果有人回答YES，并且告知在什么位置。没有则回答NO"
    
    const res = await axios.post('/api/gemma/infer', {
      image: imageBase64,
      prompt: prompt,
      enableThinking: false // 自动 JSON 模式不需要思维链显示
    })

    const { result, reasoning, prompt: sentPrompt, llm_response } = res.data
    
    ElMessageBox.alert(
      `<div style="font-size: 14px;">
        <p><b>${t('logs.manualReviewResultTitle')}</b> <span style="color: ${result === 'YES' ? '#67C23A' : '#F56C6C'}; font-weight: bold;">${result}</span></p>
        <p style="margin-top: 10px;"><b>${t('logs.manualReviewReasoningTitle')}</b></p>
        <div style="background: #f5f7fa; padding: 10px; border-radius: 4px; font-size: 12px; color: #606266; max-height: 150px; overflow-y: auto; margin-bottom: 10px;">
          ${reasoning || t('logs.none')}
        </div>
        <p><b>LLM Raw Response:</b></p>
        <pre style="background: #303133; color: #fff; padding: 10px; border-radius: 4px; font-size: 11px; overflow-x: auto;">${llm_response}</pre>
      </div>`,
      t('logs.manualReviewDialogTitle'),
      {
        dangerouslyUseHTMLString: true,
        confirmButtonText: t('logs.close'),
        width: '600px'
      }
    )
  } catch (e) {
    ElMessage.error(t('logs.manualReviewFailed') + (e.response?.data?.error || e.message))
  } finally {
    manualReviewing.value = ''
    loadingInstance.close()
  }
}

const toggleAutoRefresh = (val) => {
  if (val) {
    ElMessage.success(t('logs.autoRefreshStarted'))
    fetchLogs(true)
    refreshInterval = setInterval(() => fetchLogs(true), 60000)
  } else {
    if (refreshInterval) clearInterval(refreshInterval)
    ElMessage.info(t('logs.autoRefreshStopped'))
  }
}

// 默认显示最近4天（包含今天）
const getDefaultDateRange = () => {
  const dates = []
  for (let i = 0; i < 4; i++) {
    const d = new Date()
    d.setDate(d.getDate() - i)
    const year = d.getFullYear()
    const month = String(d.getMonth() + 1).padStart(2, '0')
    const day = String(d.getDate()).padStart(2, '0')
    dates.push(`${year}-${month}-${day}`)
  }
  return dates
}

const defaultDates = ref(getDefaultDateRange())

const fetchAreas = async () => {
  try {
    const res = await axios.get('/api/occupancy/areas')
    areaList.value = res.data || []
  } catch (e) {
    console.error("Failed to fetch areas:", e)
  }
}

const fetchLogs = async (silent = false) => {
  if (!silent) loading.value = true
  try {
    // 如果没有选定场景，不请求日志，只请求场景列表
    if (!selectedArea.value) {
      await fetchAreas()
      if (!silent) loading.value = false
      return
    }

    const params = { days: 4, areaCode: selectedArea.value }
    const res = await axios.get('/api/occupancy/logs', { params })
    allLogs.value = (res.data || []).filter(l => l.camera_id && l.areaCode && l.areaCode !== 'UNKNOWN')
    
    defaultDates.value.forEach(date => {
      fetchSummary(date)
    })
  } catch (e) {
    if (!silent) ElMessage.error(t('logs.fetchLogsFailed'))
  }
  if (!silent) loading.value = false
}

// 监听选择场景的变化，自动重新加载数据
watch(selectedArea, () => {
  fetchLogs()
})

const uniqueAreas = computed(() => {
  // 优先使用从后端获取的场景列表，如果没有则回退到日志中解析
  if (areaList.value.length > 0) return areaList.value
  const areas = new Set(allLogs.value.map(l => l.areaCode))
  return Array.from(areas).sort()
})

const handleFilterChange = () => {
  fetchLogs()
}

// 核心：按选定日期和场景，生成二维热力图数据
const displayDays = computed(() => {
  if (!selectedArea.value) return []

  let datesToDisplay = []
  if (dateRange.value && dateRange.value.length === 2) {
    const start = new Date(dateRange.value[0])
    const end = new Date(dateRange.value[1])
    for (let d = new Date(end); d >= start; d.setDate(d.getDate() - 1)) {
      const year = d.getFullYear()
      const month = String(d.getMonth() + 1).padStart(2, '0')
      const day = String(d.getDate()).padStart(2, '0')
      datesToDisplay.push(`${year}-${month}-${day}`)
    }
  } else {
    datesToDisplay = defaultDates.value
  }

  const result = []
  const areaLogs = allLogs.value.filter(log => log.areaCode === selectedArea.value)

  datesToDisplay.forEach(dateStr => {
    // 根据 dateStr 获取该日期的边界，过滤掉未来的时间点
    const todayStr = new Date().toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' }).replace(/\//g, '-')
    const isToday = dateStr === todayStr
    const currentHour = new Date().getHours()
    const currentMin = new Date().getMinutes()

    // 匹配特定日期的日志 (使用 log.date 而非 log.timestamp 解析，避免跨时区导致日期漂移)
    const dayLogs = areaLogs.filter(log => log.date === dateStr)
    
    if (dayLogs.length > 0 || datesToDisplay.length <= 4) {
      const grid = Array.from({ length: 24 }, () => Array.from({ length: 6 }, () => []))
      
      dayLogs.forEach(log => {
        if (!log.timestamp) return
        
        // 修复：解析ISO字符串时，确保时区一致。如果是本地记录的timestamp且没有带Z，
        // 解析出来可能会因为浏览器本地时区有偏差。
        // 为了安全，如果timestamp中没有T或者Z，可以假设它是本地时间
        let d = new Date(log.timestamp)
        // 如果后端返回的是 UTC (带 Z) 但你希望按照设备本地时间显示，需要确保时区
        
        const hour = d.getHours()
        const min = d.getMinutes()
        const minIdx = Math.floor(min / 10)
        
        // 过滤掉超过当前时间点的未来数据 (可能是时区偏差导致的数据漂移)
        if (isToday) {
            if (hour > currentHour || (hour === currentHour && min > currentMin)) {
                return // 跳过未来的记录
            }
        }

        if (hour >= 0 && hour < 24 && minIdx >= 0 && minIdx < 6) {
          grid[hour][minIdx].push(log)
        }
      })
      
      result.push({
        date: dateStr,
        grid: grid
      })
    }
  })
  
  return result
})

const getCellLogs = (dayData, hour, minuteIdx) => {
  return dayData.grid[hour][minuteIdx] || []
}

const getCellIntensityClass = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  if (logs.length === 0) return 'color-level-null' // 无记录（比如未来时间，或者断网没收到记录），使用浅灰色背景
  
  const occupiedLogs = logs.filter(l => l.raw_payload?.result === 'occupied')
  if (occupiedLogs.length === 0) {
    return 'color-level-0' // 有日志但无人，使用稍微深一点点的灰色，或者更暗的灰色与无记录区分
  }
  
  const count = occupiedLogs.length
  if (count === 1) return 'color-level-1'
  if (count === 2) return 'color-level-2'
  if (count === 3) return 'color-level-3'
  return 'color-level-4'
}

const getTooltip = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  const timeStr = `${hour.toString().padStart(2, '0')}:${(minuteIdx * 10).toString().padStart(2, '0')} - ${hour.toString().padStart(2, '0')}:${(minuteIdx * 10 + 9).toString().padStart(2, '0')}`
  if (logs.length === 0) return `${timeStr} ${t('logs.noDetectionRecord')}`
  
  const occupiedLogs = logs.filter(l => l.raw_payload?.result === 'occupied')
  return `${timeStr} | ${t('logs.statusOccupied')}: ${occupiedLogs.length}次, ${t('logs.summaryTotalSamples')}: ${logs.length}次`
}

// 提取并聚合多摄像头的按分钟日志
const groupedLogs = computed(() => {
  const groups = {}
  dialogLogs.value.forEach(log => {
    if (!log.timestamp) return
    const d = new Date(log.timestamp)
    const minKey = `${d.getFullYear()}-${String(d.getMonth()+1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`
    if (!groups[minKey]) groups[minKey] = []
    groups[minKey].push(log)
  })
  
  // 按时间倒序
  return Object.keys(groups).sort((a, b) => new Date(b) - new Date(a)).map(k => ({
    time: k,
    logs: groups[k]
  }))
})

const getGroupResultTagType = (group) => {
  const isOccupied = group.logs.some(l => l.raw_payload?.result === 'occupied')
  return isOccupied ? 'success' : 'info'
}

const formatGroupResult = (group) => {
  const isOccupied = group.logs.some(l => l.raw_payload?.result === 'occupied')
  return isOccupied ? t('logs.areaOccupied') : t('logs.areaEmpty')
}

const openDetail = (dayData, hour, minuteIdx) => {
  const logs = getCellLogs(dayData, hour, minuteIdx)
  if (logs.length === 0) return
  
  const timeStr = `${hour.toString().padStart(2, '0')}:${(minuteIdx * 10).toString().padStart(2, '0')} - ${hour.toString().padStart(2, '0')}:${(minuteIdx * 10 + 9).toString().padStart(2, '0')}`
  dialogTitle.value = `[${selectedArea.value}] ${dayData.date} ${timeStr} ${t('logs.detailRecordTitle')}`
  dialogLogs.value = [...logs].sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp))
  dialogVisible.value = true
}

const formatTime = (isoString) => {
  if (!isoString) return ''
  const date = new Date(isoString)
  return date.toLocaleTimeString()
}

const getEventTypeColor = (event) => {
  if (event === 'Smoking Alert') return 'danger'
  return 'primary'
}

const getResultTagType = (log) => {
  if (log.event === 'Smoking Alert') return 'danger'
  if (log.raw_payload?.result === 'occupied') return 'success'
  return 'info'
}

const formatResult = (log) => {
  if (log.event === 'Smoking Alert') return t('logs.smokingConfirmed')
  if (log.raw_payload?.result === 'occupied') return t('logs.areaOccupied')
  if (log.raw_payload?.result === 'empty') return t('logs.areaEmpty')
  return t('logs.unknown')
}

const getImageUrl = (relativePath) => {
  return `http://${window.location.hostname}:10081/${relativePath}`
}

onMounted(() => {
  fetchLogs()
})
</script>

<style scoped>
.heatmap-card {
  min-height: 800px;
}
.card-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-weight: bold;
}
.filter-section {
  display: flex;
  align-items: center;
  margin-bottom: 30px;
  padding-bottom: 20px;
  border-bottom: 1px solid #ebeef5;
}
.legend {
  display: flex;
  align-items: center;
  margin-left: auto;
  font-size: 12px;
  color: #606266;
}
.legend-colors {
  display: flex;
  list-style: none;
  padding: 0;
  margin: 0 8px;
  gap: 4px;
}
.legend-colors li {
  width: 14px;
  height: 14px;
  border-radius: 3px;
}

/* Element Plus Primary Blue Theme Heatmap Colors */
.color-level-null { background-color: #ebedf0; } /* 根本没日志（未来时间，断网等） */
.color-level-0 { background-color: #f4f4f5; }    /* 有判断日志，但是判定为无人 */
.color-level-1 { background-color: #c6e2ff; }
.color-level-2 { background-color: #79bbff; }
.color-level-3 { background-color: #409eff; }
.color-level-4 { background-color: #337ecc; }

.heatmaps-wrapper {
  display: flex;
  flex-direction: column;
  gap: 40px;
}
.day-heatmap {
  display: flex;
  flex-direction: column;
}
.day-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 10px;
  padding-left: 45px;
}
.day-title {
  margin: 0;
  font-size: 14px;
  color: #303133;
  font-weight: bold;
}
.summary-link {
  font-size: 12px;
}
.summary-content {
  padding: 10px;
}
.summary-stats {
  margin-bottom: 20px;
}
.summary-text-box {
  background: #f4f4f5;
  padding: 15px;
  border-radius: 4px;
  border-left: 4px solid #909399;
  margin-bottom: 20px;
}
.summary-label {
  font-weight: bold;
  margin-bottom: 12px;
  color: #303133;
  font-size: 15px;
}
.summary-markdown {
  line-height: 1.6;
  color: #606266;
  font-size: 14px;
}
.summary-markdown :deep(h1), 
.summary-markdown :deep(h2), 
.summary-markdown :deep(h3) {
  margin-top: 15px;
  margin-bottom: 10px;
  color: #303133;
}
.summary-markdown :deep(ul), 
.summary-markdown :deep(ol) {
  padding-left: 20px;
  margin-bottom: 10px;
}
.summary-markdown :deep(li) {
  margin-bottom: 5px;
}
.summary-markdown :deep(p) {
  margin-bottom: 10px;
}
.summary-markdown :deep(strong) {
  color: #303133;
}
.summary-details {
  margin-top: 20px;
}
.heatmap-container {
  display: flex;
  align-items: flex-start;
  width: 100%;
}
.y-axis-wrapper {
  display: flex;
  flex-direction: column;
  margin-right: 15px;
  width: 30px;
}
.y-axis {
  display: flex;
  flex-direction: column;
  justify-content: space-between;
  gap: 4px; /* Matches grid gap */
}
.y-label {
  font-size: 10px;
  color: #909399;
  line-height: 1;
  text-align: right;
  /* Make label height match cell height proportionally */
  display: flex;
  align-items: center;
  justify-content: flex-end;
  /* Use aspect-ratio to match square cells if needed, 
     but since they are in flex-column, we just need them to share the space */
  flex: 1;
  padding-bottom: 100%; /* Force labels to have same aspect ratio as cells */
  position: relative;
}
.y-label span {
  position: absolute;
  top: 50%;
  transform: translateY(-50%);
  right: 0;
}
.x-axis-placeholder {
  height: 21px; /* Matches x-axis height + margin */
}
.grid-content {
  display: flex;
  flex-direction: column;
  flex: 1;
}
.grid-columns {
  display: flex;
  gap: 4px;
  justify-content: space-between;
  width: 100%;
}
.column {
  display: flex;
  flex-direction: column;
  gap: 4px;
  flex: 1;
}
.cell {
  width: 100%;
  padding-bottom: 100%; /* Keep it square */
  height: 0;
  border-radius: 3px;
  cursor: pointer;
  transition: transform 0.1s, box-shadow 0.1s;
}
.cell:hover {
  transform: scale(1.3);
  z-index: 10;
  box-shadow: 0 0 6px rgba(0,0,0,0.2);
}
.x-axis {
  display: flex;
  justify-content: space-between;
  width: 100%;
  margin-top: 10px;
}
.x-label {
  flex: 1;
  font-size: 10px;
  color: #909399;
  text-align: center;
  white-space: nowrap;
}
.x-label:nth-child(even) {
  opacity: 0;
}

/* Dialog inner styles */
.log-detail-card {
  margin-bottom: 5px;
}
.log-header {
  display: flex;
  align-items: center;
  gap: 10px;
}
.log-event {
  font-weight: bold;
}
.log-image {
  width: 100%;
  height: 200px;
  border-radius: 4px;
  background-color: #f5f7fa;
  border: 1px solid #ebeef5;
}

/* Test Dialog Styles */
.test-uploader .el-upload {
  border: 1px dashed #d9d9d9;
  border-radius: 6px;
  cursor: pointer;
  position: relative;
  overflow: hidden;
  width: 100%;
}
.test-uploader .el-upload:hover {
  border-color: #409eff;
}
.test-uploader-placeholder {
  height: 200px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  color: #8c939d;
  background: #fbfdff;
}
.test-uploader-icon {
  font-size: 28px;
  margin-bottom: 10px;
}
.test-preview-img {
  width: 100%;
  height: 200px;
  object-fit: contain;
  display: block;
}
.test-result-section {
  border: 1px solid #ebeef5;
  border-radius: 4px;
  padding: 15px;
  min-height: 400px;
  background: #fcfcfc;
}
.result-title {
  font-size: 14px;
  font-weight: bold;
  margin-bottom: 10px;
  color: #303133;
}
.test-result-img {
  width: 100%;
  height: 250px;
  border-radius: 4px;
  margin-bottom: 15px;
  background: #000;
}
.form-tip {
  font-size: 12px;
  color: #909399;
  margin-top: 5px;
}
</style>
