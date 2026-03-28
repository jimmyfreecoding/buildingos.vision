# BuildingOS Vision - Web Manager (Jetson 边缘网关管理系统)

这是一个用于管理 Jetson Orin Nano 边缘计算设备的 Web 界面脚手架。
包含前后端分离架构，专门用于管理 `ai_engine` 的配置以及设备的网络/重启状态。

## 技术栈
*   **后端**: Node.js + Express
*   **前端**: Vue 3 + Vite + Element Plus

## 目录结构
*   `/backend`: Node.js API 服务 (监听 3000 端口)
*   `/frontend`: Vue 3 界面应用 (Vite 默认 8080 端口)

## 快速运行指南 (开发环境)

### 1. 启动后端
```bash
cd web_manager/backend
npm install
npm run dev
```

### 2. 启动前端
```bash
# 另开一个终端
cd web_manager/frontend
npm install
npm run dev
```
然后在浏览器访问: `http://localhost:8080`

## 功能清单
1.  **流媒体配置 (Cameras)**: 增删改查海康/大华等 RTSP 摄像头流地址，保存后自动同步给 `config.json`。
2.  **AI 算法参数 (AIParams)**: 滑动调节吸烟置信度、人员识别延时等核心参数。
3.  **网络设置 (Network)**: 模拟路由器界面，可切换 DHCP / 静态 IP (需配合真实 `nmcli` 或 `netplan` 脚本落地)。
4.  **重启设备 (Reboot)**: 一键下发 `sudo reboot` 指令，前端通过 `/api/ping` 轮询展示“等待重连”的遮罩层动画。

## 后期生产部署建议
*   在 Jetson 上，建议将这个前端 Build 出的 `dist` 文件夹，直接扔给已有的 ZLMediaKit 的 `www` 目录进行静态托管。
*   后端可以直接使用 `pm2` 或打包进一个新的 Docker 容器 (`web-manager-api`) 中运行，通过挂载 `/var/run/docker.sock` 来实现对其他容器的重启控制。