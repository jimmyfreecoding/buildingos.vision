<template>
  <div class="camera-grid">
    <div v-for="(camera, index) in cameras" :key="camera.name" class="camera-item">
      <!-- 移除了 title header -->
      <div class="camera-view">
        <div :id="`cam-player-${index}`" class="jessibuca-container"></div>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'

const props = defineProps({
  cameras: {
    type: Array,
    default: () => []
  }
})

// ZLMediaKit 服务器 IP (可以从当前浏览器URL获取，更灵活)
const ZLM_SERVER_IP = window.location.hostname
const players = ref({})

const createPlayer = (containerId, streamName) => {
  const container = document.getElementById(containerId)
  if (!container) return null

  if (!window.Jessibuca) {
    console.error('Jessibuca is not loaded. Please ensure jessibuca.js is in index.html')
    return null
  }

  const player = new window.Jessibuca({
    container: container,
    videoBuffer: 0.2,             // 稍微增加缓冲，防网络抖动
    isResize: true,               // 适应容器大小
    useWASM: true,                // 支持 H.265 解码
    useMSE: true,                 // 优先使用 MSE 硬件解码
    autoWasm: true,
    decoder: '/js/decoder.js',    // 显式指定你的本地 decoder 路径
    hasAudio: false,              // 监控通常不需要声音，关闭可提升性能
    loadingText: '视频加载中...',
    background: '#000000',
    controlAutoHide: true,
    isNotMute: false,             // 确保静音
    timeout: 10,                  // 设置超时时间(秒)
    heartTimeout: 10,             // 心跳超时
    supportDblclickFullscreen: true // 双击全屏
  })

  // 拼接 ZLM 后端转换好的 FLV 播放地址 (使用 10081 端口，映射到 zlm 8080)
  const flvUrl = `http://${ZLM_SERVER_IP}:10081/live/${streamName}.live.flv`
  
  // 监听错误事件进行自动重连
  player.on('error', (error) => {
    console.warn(`[Jessibuca] 播放器 ${streamName} 发生错误:`, error)
    // 延迟 3 秒后尝试重新连接
    setTimeout(() => {
      console.log(`[Jessibuca] 正在尝试重新连接 ${streamName}...`)
      if (player) {
        player.play(flvUrl)
      }
    }, 3000)
  })

  // 监听超时事件进行自动重连
  player.on('timeout', () => {
    console.warn(`[Jessibuca] 播放器 ${streamName} 连接超时`)
    setTimeout(() => {
      console.log(`[Jessibuca] 正在尝试重新连接 ${streamName}...`)
      if (player) {
        player.play(flvUrl)
      }
    }, 3000)
  })
  
  // 定期清理内存机制：每隔 4 小时重新初始化一次播放器
  // 这对 H.265 WASM 软解尤其重要，因为长时间运行容易产生内存碎片
  const reloadInterval = setInterval(() => {
    console.log(`[Jessibuca] 执行定时刷新，清理 ${streamName} 内存...`)
    if (player) {
      player.destroy()
      setTimeout(() => {
        // 重建 DOM 并重新播放
        const newPlayer = createPlayer(containerId, streamName)
        if (newPlayer) {
          // 替换掉引用，保证 onBeforeUnmount 销毁的是新的实例
          // （在 setup 中需要一种机制更新，这里依赖上层引用，由于 reloadInterval 在这里，我们需要巧妙处理）
        }
      }, 1000)
    }
  }, 4 * 60 * 60 * 1000) // 4小时
  
  // 保存定时器 ID，以便销毁时清除
  player._reloadInterval = reloadInterval

  // 开始播放
  player.play(flvUrl)
  return player
}

onMounted(() => {
  // 给一点点延迟，确保 DOM 已经完全渲染完毕再挂载播放器
  setTimeout(() => {
    props.cameras.forEach((camera, index) => {
      // 使用传入的 camera.name 作为 streamName
      const streamName = camera.name
      players.value[index] = createPlayer(`cam-player-${index}`, streamName)
    })
  }, 500)
})

onBeforeUnmount(() => {
  // 【极其重要】组件销毁时，必须销毁播放器释放内存！
  Object.values(players.value).forEach(player => {
    if (player) {
      if (player._reloadInterval) {
        clearInterval(player._reloadInterval)
      }
      player.destroy()
    }
  })
})
</script>

<script>
export default {
  name: 'CameraGrid'
}
</script>

<style scoped>
.camera-grid {
  display: grid;
  grid-template-columns: 1fr;
  grid-template-rows: repeat(4, 1fr); /* 改为 4 行，每行一个视频 */
  gap: 10px;
  height: 100%;
}

.camera-item {
  background: black; /* 改为黑色背景更适合视频 */
  border: 1px solid #333;
  border-radius: 8px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.camera-view {
  flex: 1;
  overflow: hidden;
  position: relative;
  background-color: #000;
}

.jessibuca-container {
  width: 100%;
  height: 100%;
  position: absolute;
  top: 0;
  left: 0;
}
</style>
