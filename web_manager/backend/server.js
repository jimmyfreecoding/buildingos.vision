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
app.use(express.json({ limit: '20mb' }));
app.use(express.urlencoded({ limit: '20mb', extended: true }));

// Docker 容器内的挂载路径
const CONFIG_PATH = '/app/ai_engine/config/config.json';
const DEFAULT_CONFIG_PATH = '/app/ai_engine/config/config.default.json';
const PROJECT_DIR = '/host_project';
const HOST_NSENTER = 'nsenter -t 1 -m -u -i -n -p --';

// --- 自动初始化配置文件机制 ---
if (!fs.existsSync(CONFIG_PATH)) {
    console.log(`Warning: ${CONFIG_PATH} not found. Initializing from default config...`);
    try {
        if (fs.existsSync(DEFAULT_CONFIG_PATH)) {
            fs.copyFileSync(DEFAULT_CONFIG_PATH, CONFIG_PATH);
            console.log(`Successfully copied ${DEFAULT_CONFIG_PATH} to ${CONFIG_PATH}`);
        } else {
            // 保底方案
            const emptyConfig = { streams: { occupancy: [], smoking: [] } };
            fs.writeFileSync(CONFIG_PATH, JSON.stringify(emptyConfig, null, 4), 'utf8');
            console.log(`Created empty config at ${CONFIG_PATH}`);
        }
    } catch (e) {
        console.error(`Error initializing config file: ${e.message}`);
    }
}

// Real-time AI Logs via Docker logs
let logProcess = null;

io.on('connection', (socket) => {
    console.log('Client connected for AI logs');
    
    // Send a welcome message
    socket.emit('log', { timestamp: new Date().toISOString(), message: 'Connected to AI Engine log stream...' });

    if (!logProcess) {
        // Spawn a process to tail docker logs
        // Using stdbuf or unbuffer might be needed depending on system, but tail -f usually works
        logProcess = exec(`${HOST_NSENTER} journalctl -u ai-engine -f -n 200 --no-pager`);
        
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

// --- 1. 系统状态与登录 API ---
app.post('/api/login', (req, res) => {
    const { username, password } = req.body;
    try {
        const config = fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : {};
        const auth = config.auth || { admin: 'admin', password: 'admin' };
        
        if (username === auth.admin && password === auth.password) {
            // Simple mock token
            res.json({ status: 'ok', token: 'buildingos_token_2026', username: auth.admin });
        } else {
            res.status(401).json({ status: 'error', message: 'Invalid username or password' });
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/ping', (req, res) => {
    res.json({ status: 'ok', message: 'System is running' });
});

app.get('/api/system/info', (req, res) => {
    try {
        // 1. 获取存储空间信息 (始终执行)
        let diskInfo = { used: 0, total: 0, percent: 0 };
        try {
            // 优先检测 /app/www (容器内存储路径)
            const checkPaths = ['/app/www', '/host_project', '/'];
            let found = false;
            for (const p of checkPaths) {
                if (fs.existsSync(p)) {
                    try {
                        // 尝试使用 statfsSync
                        const stats = fs.statfsSync(p);
                        diskInfo.total = Number(stats.bsize) * Number(stats.blocks);
                        diskInfo.used = diskInfo.total - (Number(stats.bsize) * Number(stats.bfree));
                        if (diskInfo.total > 0) {
                            diskInfo.percent = (diskInfo.used / diskInfo.total) * 100;
                            found = true;
                            break;
                        }
                    } catch (e) {
                        // 如果 statfsSync 失败，尝试 df 命令
                        const dfOut = require('child_process').execSync(`df -B1 ${p} | tail -1`).toString();
                        const parts = dfOut.trim().split(/\s+/);
                        if (parts.length >= 4) {
                            diskInfo.total = parseInt(parts[1]);
                            diskInfo.used = parseInt(parts[2]);
                            diskInfo.percent = (diskInfo.used / diskInfo.total) * 100;
                            found = true;
                            break;
                        }
                    }
                }
            }
        } catch (diskErr) {
            console.warn("Failed to fetch disk info:", diskErr);
        }

        const jtopFiles = ['/host_tmp/jtop_status.json', '/tmp/jtop_status.json'];
        const jtopFile = jtopFiles.find(file => fs.existsSync(file));
        if (jtopFile) {
            try {
                const jtopData = JSON.parse(fs.readFileSync(jtopFile, 'utf8'));
                if (!jtopData.error) {
                    // 将磁盘信息合并到 jtop 数据中返回
                    jtopData.disk = diskInfo;
                    return res.json(jtopData);
                }
            } catch (e) {
                console.warn("Failed to read jtop file:", e);
            }
        }
        
        // Fallback 逻辑
        const totalMem = os.totalmem();
        const freeMem = os.freemem();
        const usedMem = totalMem - freeMem;
        const memUsage = (usedMem / totalMem) * 100;
        
        const cpus = os.cpus();

        exec('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits', (smiErr, smiOut) => {
            let gpuInfo = { util: 0, memUsed: 0, memTotal: 0 };
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
            
            res.json({
                cpu: {
                    usage: Math.random() * 100, // Mock for fallback
                    cores: cpus.length,
                    details: {}
                },
                memory: {
                    ram: {
                        usagePercent: memUsage,
                        used: usedMem,
                        total: totalMem
                    },
                    swap: { usagePercent: 0, used: 0, total: 0 }
                },
                gpu: gpuInfo,
                disk: diskInfo,
                engines: {},
                power: { total: 0, gpu: 0, cpu: 0 },
                temperature: {},
                board: {
                    model: 'Fallback System',
                    jetpack: 'N/A',
                    nvpmodel: 'N/A',
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
        const zlmSecret = process.env.ZLM_API_SECRET || config.zlm?.secret || "buildingos_edge_secret_2026";
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
    res.json({ message: 'Update started. System will pull latest code and restart host ai-engine service.' });

    const updateCommand = `
        cd ${PROJECT_DIR} && \
        git reset --hard HEAD && \
        git pull origin main && \
        ${HOST_NSENTER} systemctl daemon-reload && \
        ${HOST_NSENTER} systemctl restart ai-engine && \
        ${HOST_NSENTER} systemctl status ai-engine --no-pager -n 50
    `;

    console.log('Executing OTA update (Git Pull + Host systemd restart)...');
    exec(updateCommand, (error, stdout, stderr) => {
        if (error) {
            console.error(`OTA Update failed: ${error}`);
            if (stderr) console.error(stderr);
        } else {
            console.log(`OTA Update success: ${stdout}`);
        }
    });
});

// --- 3. 业务配置 (AI Engine Config) API ---
app.get('/api/ai/status', (req, res) => {
    try {
        const config = fs.existsSync(CONFIG_PATH) ? JSON.parse(fs.readFileSync(CONFIG_PATH, 'utf8')) : { streams: { smoking: [], occupancy: [] } };
        
        exec(`${HOST_NSENTER} systemctl is-active ai-engine`, (err, stdout) => {
            const isAiEngineUp = (stdout || '').trim() === 'active';
            
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
        
        // Restart AI Engine host service
        exec(`${HOST_NSENTER} systemctl restart ai-engine`, (err) => {
             if (err) console.error("Failed to restart ai-engine host service:", err);
        });
        res.json({ message: 'Config saved successfully and ai-engine service restarted.' });
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
app.get('/api/occupancy/summary/:date', (req, res) => {
    const { date } = req.params;
    const summaryPath = path.join('/app/www/occupancy_logs', date, 'daily_summary.json');
    
    try {
        if (fs.existsSync(summaryPath)) {
            const data = fs.readFileSync(summaryPath, 'utf8');
            res.json(JSON.parse(data));
        } else {
            res.status(404).json({ error: 'Summary not found' });
        }
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/occupancy/areas', (req, res) => {
    const logsDir = '/app/www/occupancy_logs';
    try {
        if (!fs.existsSync(logsDir)) return res.json([]);
        
        let areaSet = new Set();
        const dates = fs.readdirSync(logsDir).filter(f => fs.statSync(path.join(logsDir, f)).isDirectory());
        
        // 只扫描最近 7 天的文件夹来获取场景列表，提高速度
        dates.sort().reverse().slice(0, 7).forEach(date => {
            const dateDir = path.join(logsDir, date);
            const areas = fs.readdirSync(dateDir).filter(f => fs.statSync(path.join(dateDir, f)).isDirectory());
            areas.forEach(a => areaSet.add(a.replace(/_/g, '/'))); // 恢复斜杠显示
        });
        
        res.json(Array.from(areaSet));
    } catch (e) {
        res.status(500).json({ error: e.message });
    }
});

app.get('/api/occupancy/logs', (req, res) => {
    const logsDir = '/app/www/occupancy_logs';
    const { areaCode, days } = req.query;
    const maxDays = parseInt(days) || 4;

    try {
        if (!fs.existsSync(logsDir)) {
            return res.json([]);
        }

        let results = [];
        let dates = fs.readdirSync(logsDir).filter(f => fs.statSync(path.join(logsDir, f)).isDirectory());
        
        // 按日期降序排列并截取
        dates.sort().reverse();
        const targetDates = dates.slice(0, maxDays);
        
        targetDates.forEach(date => {
            const dateDir = path.join(logsDir, date);
            let areas = fs.readdirSync(dateDir).filter(f => fs.statSync(path.join(dateDir, f)).isDirectory());
            
            // 如果提供了 areaCode，只处理该场景。注意：前端传来的 areaCode 可能是斜杠，文件夹是下划线
            if (areaCode) {
                const safeArea = areaCode.replace(/\//g, '_').replace(/\\/g, '_');
                areas = areas.filter(a => a === safeArea);
            }

            areas.forEach(area => {
                const areaDir = path.join(dateDir, area);
                const files = fs.readdirSync(areaDir);
                
                // 只看 JSON 文件
                const jsonFiles = files.filter(f => f.endsWith('.json'));
                jsonFiles.forEach(jf => {
                    try {
                        const content = fs.readFileSync(path.join(areaDir, jf), 'utf8');
                        const data = JSON.parse(content);
                        data.date = date;
                        data.id = `${date}_${area}_${jf}`;
                        results.push(data);
                    } catch (e) {
                        console.error(`Error reading json log ${jf}:`, e);
                    }
                });
            });
        });
        
        results.sort((a, b) => new Date(b.timestamp) - new Date(a.timestamp));
        res.json(results);
    } catch (e) {
        console.error("Occupancy Logs API Error:", e);
        res.status(500).json({ error: e.message });
    }
});

// --- 6. Gemma Local Model API ---
const GEMMA_HOST = process.env.GEMMA_HOST || 'host.docker.internal'; // 优先连接宿主机直跑的 Gemma
const GEMMA_PORT = process.env.GEMMA_PORT || 8080;
const AI_ENGINE_HOST = process.env.AI_ENGINE_HOST || 'host.docker.internal'; // AI Engine 在宿主机直跑
const AI_ENGINE_PORT = process.env.AI_ENGINE_PORT || 5000;

app.post('/api/ai/test', (req, res) => {
    const { image, conf_thres } = req.body;
    
    console.log(`[AI Test] Forwarding request to ${AI_ENGINE_HOST}:${AI_ENGINE_PORT}/predict...`);
    
    const payload = JSON.stringify({ image, conf_thres });
    const options = {
        hostname: AI_ENGINE_HOST,
        port: AI_ENGINE_PORT,
        path: '/predict',
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload)
        }
    };

    const aiReq = http.request(options, (aiRes) => {
        let data = '';
        aiRes.on('data', (chunk) => { data += chunk; });
        aiRes.on('end', () => {
            try {
                res.json(JSON.parse(data));
            } catch (e) {
                res.status(500).json({ error: 'Failed to parse AI Engine response', raw: data });
            }
        });
    });

    aiReq.on('error', (err) => {
        res.status(500).json({ error: 'Failed to connect to AI Engine', details: err.message });
    });

    aiReq.write(payload);
    aiReq.end();
});

app.get('/api/gemma/status', (req, res) => {
    let statusData = { status: 'Offline', details: null };

    // Helper to make GET requests to Gemma
    const fetchGemma = (path) => {
        return new Promise((resolve) => {
            http.get(`http://${GEMMA_HOST}:${GEMMA_PORT}${path}`, (res) => {
                let data = '';
                res.on('data', chunk => data += chunk);
                res.on('end', () => {
                    try {
                        resolve({ statusCode: res.statusCode, data: JSON.parse(data) });
                    } catch (e) {
                        resolve({ statusCode: res.statusCode, data: null });
                    }
                });
            }).on('error', () => resolve({ statusCode: 500, data: null }));
        });
    };

    // First check health
    fetchGemma('/health').then(async (healthRes) => {
        if (healthRes.statusCode === 200 && healthRes.data?.status === 'ok') {
            statusData.status = 'Running';
            
            // Fetch slots for real-time processing status
            const slotsRes = await fetchGemma('/slots');
            const propsRes = await fetchGemma('/props');
            
            statusData.details = {
                health: healthRes.data,
                slots: slotsRes.data || [],
                props: propsRes.data || {}
            };
        } else if (healthRes.data?.status === 'loading model') {
            statusData.status = 'Loading';
            statusData.details = { health: healthRes.data };
        } else if (healthRes.statusCode !== 500) {
            statusData.status = 'Error';
        }
        res.json(statusData);
    });
});

const clearGemmaCache = () => {
    // 兼容新版 llama-server API: POST /slots/{id}?action=release
    const postOptions = {
        hostname: GEMMA_HOST,
        port: GEMMA_PORT,
        path: '/slots/0?action=release',
        method: 'POST'
    };
    const postReq = http.request(postOptions, (res) => {
        if (res.statusCode !== 200) {
            // 如果 POST 也失败，尝试旧版 DELETE (以防万一)
            const deleteOptions = { ...postOptions, path: '/slots/0', method: 'DELETE' };
            http.request(deleteOptions).end();
        }
    });
    postReq.on('error', () => {});
    postReq.end();
};

app.post('/api/gemma/infer', (req, res) => {
    const { image, prompt, enableThinking } = req.body; 

    // 强制 JSON 输出的 System Prompt
    const systemPrompt = (
        "You are a professional image analyzer. You MUST output a JSON object ONLY. " +
        "Structure: {\"result\": \"YES/NO/SUCCESS\", \"analysis\": \"your detailed observation or result\"}"
    );

    const payload = JSON.stringify({
        model: "buildingos_review_engine",
        messages: [
            {
                role: "system",
                content: systemPrompt
            },
            {
                role: "user",
                content: [
                    { type: "image_url", image_url: { url: image } },
                    { type: "text", text: prompt || "检测图片中是否有活人存在，仔细鉴别头肩和肢体等人体要输，如果有人回答YES，并且告知在什么位置。没有则回答NO" }
                ]
            }
        ],
        chat_template_kwargs: {
            enable_thinking: enableThinking !== undefined ? enableThinking : false 
        },
        stream: false,
        temperature: 0.0,
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

    const startTime = Date.now();

    const gemmaReq = http.request(options, (gemmaRes) => {
        let data = '';
        gemmaRes.on('data', (chunk) => { data += chunk; });
        gemmaRes.on('end', () => {
            const duration = Date.now() - startTime;
            try {
                const response = JSON.parse(data);
                const content = response.choices?.[0]?.message?.content || '';
                
                let result = 'UNKNOWN';
                let reasoning = response.choices?.[0]?.message?.reasoning_content || '';
                
                try {
                    // 清理 Markdown 代码块
                    const cleanContent = content.replace(/```json/g, "").replace(/```/g, "").trim();
                    const parsed = JSON.parse(cleanContent);
                    result = parsed.result || 'UNKNOWN';
                    if (parsed.analysis) reasoning = parsed.analysis;
                } catch (e) {
                    console.warn("Manual infer JSON parse failed, fallback to text search");
                    if (content.toUpperCase().includes("YES")) result = "YES";
                    else if (content.toUpperCase().includes("NO")) result = "NO";
                    else result = content.substring(0, 50); // Fallback for descriptions
                }

                res.json({ 
                    result: result, 
                    prompt: prompt,
                    llm_response: content,
                    reasoning: reasoning,
                    usage: response.usage,
                    durationMs: duration
                });
            } catch (e) {
                res.status(500).json({ error: 'Failed to parse Gemma response', raw: data });
            } finally {
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
