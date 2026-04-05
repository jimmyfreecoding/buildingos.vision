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
        
        // Jetson Orin Nano doesn't use nvidia-smi, it uses tegrastats.
        // Reading /sys/devices/gpu.0/load is a common way to get GPU load on Jetson without parsing tegrastats continuously.
        exec('cat /sys/devices/gpu.0/load', (error, stdout, stderr) => {
            let gpuInfo = { util: 0, memUsed: 0, memTotal: 0 };
            
            if (!error && stdout) {
                // Jetson GPU load is 0-1000, so we divide by 10
                const load = parseInt(stdout.trim(), 10);
                if (!isNaN(load)) {
                    gpuInfo.util = load / 10;
                }
                
                // For unified memory on Jetson, GPU memory is essentially system memory.
                // We can just mirror system memory for the "GPU" card to avoid confusion, 
                // or leave it as N/A. Let's just show system memory usage for the unified architecture.
                gpuInfo.memTotal = Math.round(totalMem / (1024 * 1024));
                gpuInfo.memUsed = Math.round(usedMem / (1024 * 1024));
            } else {
                // Fallback to nvidia-smi if not on Jetson
                exec('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits', (smiErr, smiOut) => {
                    if (!smiErr && smiOut) {
                        const parts = smiOut.split(',').map(s => s.trim());
                        if (parts.length >= 3) {
                            gpuInfo = {
                                util: parseFloat(parts[0]),
                                memUsed: parseFloat(parts[1]),
                                memTotal: parseFloat(parts[2])
                            };
                        }
                    }
                });
            }

            // Small delay to allow nvidia-smi to complete if it was called
            setTimeout(() => {
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
            }, 100);
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
        
        http.get(getMediaListUrl, (zlmRes) => {
            let data = '';
            zlmRes.on('data', (chunk) => {
                data += chunk;
            });
            zlmRes.on('end', () => {
                try {
                    const zlmResponse = JSON.parse(data);
                    res.json(zlmResponse);
                } catch (e) {
                    console.error("Parse ZLM response failed:", e, data);
                    res.status(500).json({ error: "Failed to parse ZLM response" });
                }
            });
        }).on('error', (err) => {
            console.error("HTTP GET to ZLM failed:", err);
            res.status(500).json({ error: "Failed to fetch ZLM data" });
        });
    } catch (e) {
        console.error("ZLM Metrics API Error:", e);
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
app.get('/api/ai/status', (req, res) => {
    try {
        const config = fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : { streams: { smoking: [], occupancy: [] } };
        
        // 我们通过检查 docker logs 中最后几百行来判断某个线程是否成功启动并输出过日志
        // 或者简单点，如果有配置，且 AI 引擎容器正在运行，我们假定它们在 Running，否则在 Waiting/Error
        exec('docker ps --filter "name=buildingos-vision-ai-engine-1" --format "{{.Status}}"', (err, stdout) => {
            const isAiEngineUp = stdout && stdout.includes('Up');
            
            let tasks = [];
            if (config.streams) {
                if (config.streams.smoking) {
                    config.streams.smoking.forEach(s => {
                        tasks.push({
                            camId: s.id,
                            taskType: 'smoking',
                            status: isAiEngineUp ? 'Running' : 'Offline'
                        });
                    });
                }
                if (config.streams.occupancy) {
                    config.streams.occupancy.forEach(s => {
                        tasks.push({
                            camId: s.id,
                            taskType: 'occupancy',
                            status: isAiEngineUp ? 'Running' : 'Offline'
                        });
                    });
                }
            }
            res.json(tasks);
        });
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});
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
        
        http.get(getMediaListUrl, (zlmRes) => {
            let data = '';
            zlmRes.on('data', (chunk) => {
                data += chunk;
            });
            zlmRes.on('end', () => {
                try {
                    const zlmResponse = JSON.parse(data);
                    if (zlmResponse.code === 0 && zlmResponse.data) {
                        // We only care about unique stream IDs (app=live)
                        const activeStreamIds = new Set(zlmResponse.data.map(item => item.stream));
                        
                        activeStreamIds.forEach(streamId => {
                            if (!newStreamIds.has(streamId)) {
                                // This stream exists in ZLM but NOT in our new config! Kill it.
                                
                                // Helper to execute HTTP GET for cleanup
                                const sendZlmCleanup = (url, logMsg) => {
                                    http.get(url, () => {
                                        if (logMsg) console.log(logMsg);
                                    }).on('error', (err) => {
                                        console.error(`Cleanup failed for ${url}:`, err);
                                    });
                                };

                                // Approach 1: Close all active connections for this stream
                                const closeUrl = `http://zlm:80/index/api/close_streams?secret=${zlmSecret}&app=live&stream=${streamId}&vhost=__defaultVhost__&force=1`;
                                sendZlmCleanup(closeUrl, `[SYNC] Closing active connections for orphaned stream ${streamId}`);

                                // Approach 2: Delete proxy by iterating through keys
                                const proxyKey1 = `__defaultVhost__/live/${streamId}`;
                                const delProxyUrl1 = `http://zlm:80/index/api/delStreamProxy?secret=${zlmSecret}&key=${proxyKey1}`;
                                sendZlmCleanup(delProxyUrl1);
                                
                                // Approach 3: Sometimes the key is just the stream ID or a hash.
                                const delProxyUrl2 = `http://zlm:80/index/api/delStreamProxy?secret=${zlmSecret}&key=${streamId}`;
                                sendZlmCleanup(delProxyUrl2);
                            }
                        });
                    }
                } catch (parseErr) {
                    console.error("Failed to parse ZLM media list response:", parseErr);
                }
            });
        }).on('error', (err) => {
            console.error(`Failed to fetch media list from ZLM for sync:`, err);
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

// --- 5. Occupancy Logs API ---
app.get('/api/occupancy/logs', (req, res) => {
    const logsDir = '/app/www/occupancy_logs';
    try {
        if (!fs.existsSync(logsDir)) {
            return res.json([]);
        }

        let results = [];
        const dates = fs.readdirSync(logsDir).filter(f => fs.statSync(path.join(logsDir, f)).isDirectory());
        
        dates.forEach(date => {
            const dateDir = path.join(logsDir, date);
            const areas = fs.readdirSync(dateDir).filter(f => fs.statSync(path.join(dateDir, f)).isDirectory());
            
            areas.forEach(area => {
                const areaDir = path.join(dateDir, area);
                const files = fs.readdirSync(areaDir);
                
                // Only look for JSON files
                const jsonFiles = files.filter(f => f.endsWith('.json'));
                jsonFiles.forEach(jf => {
                    try {
                        const content = fs.readFileSync(path.join(areaDir, jf), 'utf8');
                        const data = JSON.parse(content);
                        // Add some helper fields
                        data.date = date;
                        data.id = `${date}_${area}_${jf}`;
                        results.push(data);
                    } catch (e) {
                        console.error(`Error reading json log ${jf}:`, e);
                    }
                });
            });
        });
        
        // Sort by timestamp descending
        results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        res.json(results);
    } catch (e) {
        console.error("Occupancy Logs API Error:", e);
        res.status(500).json({ error: e.message });
    }
});

// --- 6. Gemma Local Model API ---
const GEMMA_HOST = process.env.GEMMA_HOST || '172.17.0.1'; // Default docker bridge to host
const GEMMA_PORT = process.env.GEMMA_PORT || 8080;

app.get('/api/gemma/status', (req, res) => {
    http.get(`http://${GEMMA_HOST}:${GEMMA_PORT}/health`, (gemmaRes) => {
        if (gemmaRes.statusCode === 200) {
            res.json({ status: 'Running' });
        } else {
            res.json({ status: 'Error' });
        }
    }).on('error', (err) => {
        res.json({ status: 'Offline', error: err.message });
    });
});

const clearGemmaCache = () => {
    const deleteOptions = {
        hostname: GEMMA_HOST,
        port: GEMMA_PORT,
        path: '/slots/0',
        method: 'DELETE'
    };
    const delReq = http.request(deleteOptions, (delRes) => {
        console.log(`Gemma context cache cleared, status: ${delRes.statusCode}`);
    });
    delReq.on('error', (err) => {
        console.error('Failed to clear Gemma context cache:', err.message);
    });
    delReq.end();
};

app.post('/api/gemma/infer', (req, res) => {
    const { image, prompt } = req.body; // image should be base64 string, prompt is text

    const payload = JSON.stringify({
        model: "gemma",
        messages: [
            {
                role: "user",
                content: [
                    { type: "text", text: prompt || "Describe this image in detail." },
                    { type: "image_url", image_url: { url: image } }
                ]
            }
        ],
        temperature: 0.1,
        max_tokens: 512
    });

    const options = {
        hostname: GEMMA_HOST,
        port: GEMMA_PORT,
        path: '/v1/chat/completions',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload)
        }
    };

    const gemmaReq = http.request(options, (gemmaRes) => {
        let data = '';
        gemmaRes.on('data', (chunk) => { data += chunk; });
        gemmaRes.on('end', () => {
            try {
                const response = JSON.parse(data);
                res.json({ result: response.choices?.[0]?.message?.content || 'No result', raw: response });
            } catch (e) {
                res.status(500).json({ error: 'Failed to parse Gemma response', raw: data });
            } finally {
                // 主动释放 Cache
                clearGemmaCache();
            }
        });
    });

    gemmaReq.on('error', (err) => {
        res.status(500).json({ error: 'Failed to connect to Gemma server', details: err.message });
        clearGemmaCache();
    });

    gemmaReq.write(payload);
    gemmaReq.end();
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Backend server running on http://localhost:${PORT}`);
});
