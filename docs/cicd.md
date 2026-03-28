# 边缘网关 CI/CD 与 OTA 升级方案

本文档描述了如何在国内受限网络环境下，通过部署在 Jetson Orin Nano 上的 `web_manager`，实现对 AI 推理引擎 (`ai_engine`) 等核心服务的 OTA（Over-The-Air）热更新与 CI/CD 流水线落地。

---

## 1. 架构设计：基于 Web Manager 的轻量级 OTA

在边缘计算场景中，由于设备通常部署在内网且缺乏公网独立 IP，无法被动接收云端 CI/CD（如 GitHub Actions, Jenkins）的 Webhook 推送。

因此，我们采用 **“边缘主动拉取 (Pull) + 本地重构”** 的策略。让 `web_manager` 后端充当本地的运维机器人。

### 1.1 核心特权挂载
为了让 `web_manager` (Node.js) 拥有控制其他业务容器的权限，在 `docker-compose.yml` 中必须为其挂载特权：

```yaml
  web-manager:
    image: node:18-alpine
    # ... 其他配置
    volumes:
      # 1. 挂载 docker.sock，赋予它执行 docker 命令、控制其他容器的权力
      - /var/run/docker.sock:/var/run/docker.sock
      # 2. 将宿主机的项目根目录挂载进去，使其能执行 docker-compose 命令
      - /home/jetson/buildingos.vision:/host_project
```

---

## 2. 应对国内网络环境的升级策略

在国内直接在边缘端执行 `git pull` 和 `docker build` 极易因网络问题卡死，且编译过程会消耗大量边缘算力。

工业级的最佳实践是：**“云端编译，边缘拉包” (Registry 模式)**。

### 2.1 云端持续集成 (CI)
当代码或模型（`.pt` / `.engine`）更新推送到代码库后，由云端服务器触发构建，将最新的 `ai-engine` 打包成 Docker 镜像，并推送到国内的私有镜像仓库（如阿里云/腾讯云容器镜像服务）。

*示例镜像标签：`registry.cn-hangzhou.aliyuncs.com/buildingos/ai-engine:latest`*

### 2.2 边缘端持续部署 (CD / OTA)
当实施人员在 Web 界面点击“检查更新”，或设备收到云端的 MQTT 更新指令时，`web_manager` 后端将执行以下流程：

1. **登录私有仓库** (若为私有镜像)。
2. **拉取最新镜像**。
3. **重启相关容器** (Docker 会自动使用新镜像替换旧容器，实现无缝升级)。

---

## 3. 代码实现参考

在 `web_manager/backend/server.js` 中增加 OTA 升级的 API 接口：

```javascript
const { exec } = require('child_process');

app.post('/api/system/update', (req, res) => {
    // 挂载到容器内的项目根目录
    const projectDir = '/host_project'; 
    
    // 立即返回响应，避免前端 HTTP 请求超时
    res.json({ message: 'Update started. System will pull latest code and rebuild containers.' });

    // 组合命令：拉取最新镜像 -> 重启服务
    const updateCommand = `
        cd ${projectDir} && \
        docker compose -f buildingos.vision.yml pull ai-engine && \
        docker compose -f buildingos.vision.yml up -d ai-engine
    `;

    console.log('Executing OTA update...');
    exec(updateCommand, (error, stdout, stderr) => {
        if (error) {
            console.error(`OTA Update failed: ${error}`);
            // 可在此处添加 MQTT 告警，通知云平台更新失败
        } else {
            console.log(`OTA Update success: ${stdout}`);
            // 可在此处添加 MQTT 消息，通知云平台更新成功及最新版本号
        }
    });
});
```

---

## 4. 前端交互体验设计

由于拉取镜像和重启容器可能需要 1~5 分钟的时间，前端交互需进行妥善处理以防用户误操作：

1. **发起请求**：用户点击“系统更新”。
2. **全屏锁定**：前端调用 `/api/system/update` 成功后，立刻弹出 `v-loading` 全屏遮罩，提示文案：“系统正在从云端拉取最新算法并部署，请等待 3-5 分钟...”。
3. **心跳轮询**：前端等待 10 秒后，开始每 5 秒向后端的 `/api/ping` 接口发送请求。
4. **恢复界面**：当 `ping` 接口成功返回 200 OK，说明 `web_manager` 及依赖服务已全部重启完毕，此时前端取消遮罩并自动 `reload` 刷新页面。