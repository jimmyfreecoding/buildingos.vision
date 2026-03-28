const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

const app = express();
app.use(cors());
app.use(express.json());

// Docker 容器内的挂载路径
const CONFIG_PATH = '/app/ai_engine/config/config.json';
const PROJECT_DIR = '/host_project';

// --- 1. 系统状态与重启 API ---
app.get('/api/ping', (req, res) => {
    res.json({ status: 'ok', message: 'System is running' });
});

app.post('/api/system/reboot', (req, res) => {
    res.json({ message: 'System is rebooting in 3 seconds...' });
    console.log('Reboot command received. Rebooting soon...');
    
    // 实际的重启通常由底层宿主机执行，在容器内如果没有 privileged 权限，重启会失败。
    // 这里我们先模拟，后期可以通过 docker.sock 或特定的主机脚本实现硬重启
    setTimeout(() => {
        exec('reboot', (error, stdout, stderr) => {
            if (error) console.error(`Reboot error: ${error}`);
        });
    }, 3000);
});

// --- 2. OTA 升级 API ---
app.post('/api/system/update', (req, res) => {
    res.json({ message: 'Update started. System will pull latest code and rebuild containers.' });

    const updateCommand = `
        cd ${PROJECT_DIR} && \
        git reset --hard HEAD && \
        git pull origin main && \
        docker compose -f buildingos.vision.yml up -d --build
    `;

    console.log('Executing OTA update (Git Pull + Docker Compose Build)...');
    exec(updateCommand, (error, stdout, stderr) => {
        if (error) {
            console.error(`OTA Update failed: ${error}`);
        } else {
            console.log(`OTA Update success: ${stdout}`);
        }
    });
});

// --- 3. 业务配置 (AI Engine Config) API ---
app.get('/api/config', (req, res) => {
    try {
        if (!fs.existsSync(CONFIG_PATH)) {
            return res.status(404).json({ error: 'Config file not found' });
        }
        const data = fs.readFileSync(CONFIG_PATH, 'utf8');
        res.json(JSON.parse(data));
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.post('/api/config', (req, res) => {
    try {
        const oldConfig = fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : { streams: { smoking: [], occupancy: [] } };
        const newConfig = req.body;
        
        // Save new config
        fs.writeFileSync(CONFIG_PATH, JSON.stringify(newConfig, null, 4), 'utf8');
        
        // Find deleted streams
        const oldStreams = [
            ...(oldConfig.streams?.smoking || []),
            ...(oldConfig.streams?.occupancy || [])
        ];
        const newStreams = [
            ...(newConfig.streams?.smoking || []),
            ...(newConfig.streams?.occupancy || [])
        ];
        
        const newStreamIds = new Set(newStreams.map(s => s.id));
        const zlmSecret = newConfig.zlm?.secret || "buildingos_edge_secret_2026";
        
        // --- Full Synchronization with ZLM ---
        // Instead of just relying on what was deleted in this specific save,
        // we query ZLM for ALL currently active streams, and if any stream in ZLM
        // is NOT in our newConfig, we force close it.
        const getMediaListUrl = `http://zlm:80/index/api/getMediaList?secret=${zlmSecret}`;
        
        exec(`curl -s "${getMediaListUrl}"`, (err, stdout) => {
            if (err) {
                console.error(`Failed to fetch media list from ZLM for sync:`, err);
            } else {
                try {
                    const zlmResponse = JSON.parse(stdout);
                    if (zlmResponse.code === 0 && zlmResponse.data) {
                        // We only care about unique stream IDs (app=live)
                        // ZLM returns multiple entries for the same stream (rtsp, rtmp, hls, etc.),
                        // but closing one by app and stream name closes the source.
                        const activeStreamIds = new Set(zlmResponse.data.map(item => item.stream));
                        
                        activeStreamIds.forEach(streamId => {
                            if (!newStreamIds.has(streamId)) {
                                // This stream exists in ZLM but NOT in our new config! Kill it.
                                const closeUrl = `http://zlm:80/index/api/close_streams?secret=${zlmSecret}&app=live&stream=${streamId}&vhost=__defaultVhost__&force=1`;
                                console.log(`[SYNC] Found orphaned stream proxy in ZLM (${streamId}). Closing: ${closeUrl}`);
                                exec(`curl -s "${closeUrl}"`, (closeErr, closeStdout) => {
                                    if (closeErr) console.error(`[SYNC] Failed to close orphaned stream ${streamId}:`, closeErr);
                                    else console.log(`[SYNC] ZLM close response for ${streamId}:`, closeStdout);
                                });
                            }
                        });
                    }
                } catch (parseErr) {
                    console.error("Failed to parse ZLM media list response:", parseErr);
                }
            }
        });

        // 配置保存后，重启 ai-engine 容器使其生效
        // 由于在容器内执行，我们直接重启指定的容器名
        exec('docker restart buildingos-vision-ai-engine-1', (err) => {
             if (err) console.error("Failed to restart ai-engine container:", err);
        });
        res.json({ message: 'Config saved successfully, deleted streams cleared from ZLM, and AI Engine restarted.' });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

// --- 4. 网络配置 API (模拟) ---
app.get('/api/network', (req, res) => {
    res.json({
        mode: 'static',
        ip: '192.168.1.100',
        netmask: '255.255.255.0',
        gateway: '192.168.1.1',
        dns: '8.8.8.8'
    });
});

app.post('/api/network', (req, res) => {
    console.log('Applying new network settings:', req.body);
    res.json({ message: 'Network settings applied. Please reboot for changes to take full effect.' });
});

const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
    console.log(`Backend server running on http://localhost:${PORT}`);
});
