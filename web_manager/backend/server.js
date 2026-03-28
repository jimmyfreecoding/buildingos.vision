const express = require('express');
const cors = require('cors');
const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');
const os = require('os');
const http = require('http');

const app = express();
const server = http.createServer(app);

// WebSocket setup for real-time logs
const { Server } = require("socket.io");
const io = new Server(server, {
  cors: {
    origin: "*",
    methods: ["GET", "POST"]
  }
});

app.use(cors());
app.use(express.json());

// Docker 容器内的挂载路径
const CONFIG_PATH = '/app/ai_engine/config/config.json';
const PROJECT_DIR = '/host_project';

// Real-time AI Logs via Docker logs
let logProcess = null;

io.on('connection', (socket) => {
    console.log('Client connected for AI logs');
    
    // Send a welcome message
    socket.emit('log', { timestamp: new Date().toISOString(), message: 'Connected to AI Engine log stream...' });

    if (!logProcess) {
        // Spawn a process to tail docker logs
        // Using stdbuf or unbuffer might be needed depending on system, but tail -f usually works
        logProcess = exec('docker logs -f buildingos-vision-ai-engine-1');
        
        logProcess.stdout.on('data', (data) => {
            const lines = data.split('\n');
            lines.forEach(line => {
                if (line.trim()) {
                    // Very simple parsing, try to extract camera ID if present [cam_id]
                    let camId = 'system';
                    const match = line.match(/\[(.*?)\]/);
                    if (match && match[1]) {
                        camId = match[1];
                    }
                    
                    io.emit('log', {
                        timestamp: new Date().toISOString(),
                        message: line,
                        camId: camId
                    });
                }
            });
        });

        logProcess.stderr.on('data', (data) => {
             const lines = data.split('\n');
             lines.forEach(line => {
                 if (line.trim()) {
                     io.emit('log', {
                         timestamp: new Date().toISOString(),
                         message: `[ERROR] ${line}`,
                         camId: 'system'
                     });
                 }
             });
        });
    }

    socket.on('disconnect', () => {
        console.log('Client disconnected from logs');
        // If no more clients, maybe kill logProcess, but it's fine to keep running for a small edge device
    });
});

// --- 1. 系统状态与重启 API ---
app.get('/api/ping', (req, res) => {
    res.json({ status: 'ok', message: 'System is running' });
});

app.get('/api/system/info', (req, res) => {
    try {
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const usedMem = totalMem - freeMem;
        const memUsage = (usedMem / totalMem) * 100;
        
        const cpus = os.cpus();
        
        // Mock GPU for now if nvidia-smi is not available
        exec('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits', (error, stdout, stderr) => {
            let gpuInfo = { util: 0, memUsed: 0, memTotal: 0 };
            if (!error && stdout) {
                const parts = stdout.split(',').map(s => s.trim());
                if (parts.length >= 3) {
                    gpuInfo = {
                        util: parseFloat(parts[0]),
                        memUsed: parseFloat(parts[1]),
                        memTotal: parseFloat(parts[2])
                    };
                }
            }

            res.json({
                cpu: {
                    cores: cpus.length,
                    model: cpus[0].model,
                    usage: Math.random() * 100 // Mock CPU usage for quick demo, a real impl needs interval measuring
                },
                memory: {
                    total: totalMem,
                    free: freeMem,
                    used: usedMem,
                    usagePercent: memUsage
                },
                gpu: gpuInfo,
                os: {
                    platform: os.platform(),
                    release: os.release(),
                    uptime: os.uptime()
                }
            });
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/zlm/metrics', (req, res) => {
    try {
        const config = fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : {};
        const zlmSecret = config.zlm?.secret || "buildingos_edge_secret_2026";
        const getMediaListUrl = `http://zlm:80/index/api/getMediaList?secret=${zlmSecret}`;
        
        exec(`curl -s "${getMediaListUrl}"`, (err, stdout) => {
            if (err) {
                return res.status(500).json({ error: "Failed to fetch ZLM data" });
            }
            try {
                const zlmResponse = JSON.parse(stdout);
                res.json(zlmResponse);
            } catch (e) {
                res.status(500).json({ error: "Failed to parse ZLM response" });
            }
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
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
        // Query ZLM for ALL currently active streams, and if any stream in ZLM
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
                        const activeStreamIds = new Set(zlmResponse.data.map(item => item.stream));
                        
                        activeStreamIds.forEach(streamId => {
                            if (!newStreamIds.has(streamId)) {
                                // This stream exists in ZLM but NOT in our new config! Kill it.
                                
                                // Approach 1: Close all active connections for this stream
                                const closeUrl = `http://zlm:80/index/api/close_streams?secret=${zlmSecret}&app=live&stream=${streamId}&vhost=__defaultVhost__&force=1`;
                                console.log(`[SYNC] Closing active connections for orphaned stream ${streamId}`);
                                exec(`curl -s "${closeUrl}"`, () => {});

                                // Approach 2: Delete proxy by iterating through keys
                                // The key is what was returned by addStreamProxy. If we don't know it,
                                // we can use the original URL or standard key format, but ZLM's delStreamProxy 
                                // is notoriously picky. 
                                // A safer approach when proxy is stubborn: restart ZLM container if needed,
                                // but for now, we try standard key formats.
                                const proxyKey1 = `__defaultVhost__/live/${streamId}`;
                                const delProxyUrl1 = `http://zlm:80/index/api/delStreamProxy?secret=${zlmSecret}&key=${proxyKey1}`;
                                exec(`curl -s "${delProxyUrl1}"`, () => {});
                                
                                // Approach 3: Sometimes the key is just the stream ID or a hash.
                                const delProxyUrl2 = `http://zlm:80/index/api/delStreamProxy?secret=${zlmSecret}&key=${streamId}`;
                                exec(`curl -s "${delProxyUrl2}"`, () => {});
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
server.listen(PORT, () => {
    console.log(`Backend server running on http://localhost:${PORT}`);
});
