#!/usr/bin/env python3
import json
import time
import os
import tempfile
import traceback

try:
    from jtop import jtop
except ImportError:
    print("jtop module not found. Please install it using: sudo pip3 install -U jetson-stats")
    # Write a fallback JSON to avoid breaking the frontend if jtop isn't available
    fallback_data = {
        "error": "jtop not installed on host"
    }
    with open('/tmp/jtop_status.json', 'w') as f:
        json.dump(fallback_data, f)
    exit(1)

OUTPUT_FILE = '/tmp/jtop_status.json'

def main():
    print("Starting jtop monitoring daemon...")
    # Initialize jtop
    with jtop() as jetson:
        while jetson.ok():
            try:
                # 1. CPU Info (兼容不同 jtop 数据结构)
                cpu_obj = jetson.cpu if isinstance(jetson.cpu, dict) else {}
                cpu_cores = cpu_obj.get('cpu', []) if isinstance(cpu_obj.get('cpu', []), list) else []
                cpu_data = {
                    "usage": 0,
                    "cores": len(cpu_cores),
                    "details": {}
                }
                if cpu_cores:
                    total_cpu_usage = 0
                    for i, core in enumerate(cpu_cores):
                        idle = float(core.get('idle', 100) or 100)
                        usage = max(0.0, min(100.0, 100.0 - idle))
                        freq_cur = 0
                        freq_info = core.get('freq', {})
                        if isinstance(freq_info, dict):
                            freq_cur = freq_info.get('cur', 0) or 0
                        if not freq_cur:
                            info_freq = core.get('info_freq', {})
                            if isinstance(info_freq, dict):
                                freq_cur = info_freq.get('cur', 0) or 0
                        cpu_data["details"][f"cpu{i}"] = {
                            "usage": usage,
                            "freq": int(freq_cur)
                        }
                        total_cpu_usage += usage
                    cpu_data["usage"] = total_cpu_usage / len(cpu_cores)
                else:
                    # fallback: 老结构 total/val
                    total = cpu_obj.get('total', {}) if isinstance(cpu_obj, dict) else {}
                    if isinstance(total, dict):
                        idle = float(total.get('idle', 100) or 100)
                        cpu_data["usage"] = max(0.0, min(100.0, 100.0 - idle))

                # 2. Memory (RAM & Swap)
                ram = jetson.memory.get('RAM', {})
                swap = jetson.memory.get('SWAP', {})
                
                # Convert KB to Bytes for frontend consistency
                memory_data = {
                    "ram": {
                        "used": ram.get('used', 0) * 1024,
                        "total": ram.get('tot', 0) * 1024,
                        "free": ram.get('free', 0) * 1024,
                        "shared": ram.get('shared', 0) * 1024,
                        "cached": ram.get('cached', 0) * 1024,
                        "buffers": ram.get('buffers', 0) * 1024,
                        "usagePercent": (ram.get('used', 0) / max(ram.get('tot', 1), 1)) * 100
                    },
                    "swap": {
                        "used": swap.get('used', 0) * 1024,
                        "total": swap.get('tot', 0) * 1024,
                        "cached": swap.get('cached', 0) * 1024,
                        "usagePercent": (swap.get('used', 0) / max(swap.get('tot', 1), 1)) * 100
                    }
                }

                # 3. GPU Info
                gpu = jetson.gpu if isinstance(jetson.gpu, dict) else {}
                if 'gpu' in gpu and isinstance(gpu.get('gpu'), dict):
                    gpu = gpu.get('gpu')
                gpu_status = gpu.get('status', {}) if isinstance(gpu.get('status', {}), dict) else {}
                gpu_freq = gpu.get('freq', {}) if isinstance(gpu.get('freq', {}), dict) else {}
                gpu_load = gpu_status.get('load', gpu.get('val', 0))
                if isinstance(gpu_load, str):
                    try:
                        gpu_load = float(gpu_load)
                    except Exception:
                        gpu_load = 0
                gpu_data = {
                    "util": float(gpu_load or 0),
                    "freq": int(gpu_freq.get('cur', gpu.get('frq', 0)) or 0),
                    "freqMax": int(gpu_freq.get('max', gpu.get('frq_max', gpu.get('max', 0))) or 0),
                    "memUsed": memory_data["ram"]["used"] / (1024*1024), # In MB, mirroring unified RAM
                    "memTotal": memory_data["ram"]["total"] / (1024*1024)
                }

                # 4. Engines (Hardware accelerators)
                engines_data = {}
                for name, engine in jetson.engine.items():
                    if isinstance(engine, dict):
                        if 'val' in engine:
                            engines_data[name] = engine.get('val', 0)
                        else:
                            # 新结构: {"NVDEC":{"online":false,"cur":...}, ...}
                            sub_values = []
                            for sub_engine in engine.values():
                                if isinstance(sub_engine, dict):
                                    if sub_engine.get('online', False):
                                        sub_values.append(100)
                                    elif 'val' in sub_engine:
                                        sub_values.append(sub_engine.get('val', 0))
                                    else:
                                        sub_values.append(0)
                            engines_data[name] = max(sub_values) if sub_values else 0
                    elif isinstance(engine, bool):
                        engines_data[name] = 100 if engine else 0
                    else:
                        engines_data[name] = 0

                # 5. Power
                power_obj = jetson.power if isinstance(jetson.power, dict) else {}
                power_tot = power_obj.get('tot', {}) if isinstance(power_obj.get('tot', {}), dict) else {}
                power_rail = power_obj.get('rail', {}) if isinstance(power_obj.get('rail', {}), dict) else {}
                vdd_cpu_gpu_cv = power_rail.get('VDD_CPU_GPU_CV', {}) if isinstance(power_rail.get('VDD_CPU_GPU_CV', {}), dict) else {}
                vdd_soc = power_rail.get('VDD_SOC', {}) if isinstance(power_rail.get('VDD_SOC', {}), dict) else {}
                power_data = {
                    "total": power_tot.get('avg', power_tot.get('power', 0)),
                    "gpu": power_obj.get('gpu', {}).get('avg', 0) if isinstance(power_obj.get('gpu', {}), dict) else 0,
                    "cpu": power_obj.get('cpu', {}).get('avg', 0) if isinstance(power_obj.get('cpu', {}), dict) else 0,
                    "soc": power_obj.get('soc', {}).get('avg', vdd_soc.get('avg', vdd_soc.get('power', 0))) if isinstance(power_obj.get('soc', {}), dict) else vdd_soc.get('avg', vdd_soc.get('power', 0)),
                    "cv": power_obj.get('cv', {}).get('avg', 0) if isinstance(power_obj.get('cv', {}), dict) else 0,
                    "vdd_in": power_tot.get('avg', power_tot.get('power', 0)),
                    "cpu_gpu_cv": vdd_cpu_gpu_cv.get('avg', vdd_cpu_gpu_cv.get('power', 0))
                }

                # 6. Temperatures
                temp_data = {}
                for name, sensor in jetson.temperature.items():
                    temp_data[name] = sensor.get('temp', 0)

                # 7. Board Info & Status
                nvpmodel_obj = getattr(jetson, 'nvpmodel', None)
                if isinstance(nvpmodel_obj, dict):
                    nvpmodel_value = nvpmodel_obj.get('name', 'Unknown')
                else:
                    nvpmodel_value = getattr(nvpmodel_obj, 'name', None) or str(nvpmodel_obj) if nvpmodel_obj is not None else 'Unknown'

                jetson_clocks_obj = getattr(jetson, 'jetson_clocks', None)
                if isinstance(jetson_clocks_obj, (bool, int, float, str)) or jetson_clocks_obj is None:
                    jetson_clocks_value = jetson_clocks_obj
                else:
                    jetson_clocks_value = str(jetson_clocks_obj)

                uptime_obj = getattr(jetson, 'uptime', 0)
                if hasattr(uptime_obj, 'total_seconds'):
                    uptime_value = int(uptime_obj.total_seconds())
                elif isinstance(uptime_obj, (int, float)):
                    uptime_value = int(uptime_obj)
                else:
                    uptime_value = 0

                board_obj = jetson.board if isinstance(jetson.board, dict) else {}
                board_info = board_obj.get('info', {}) if isinstance(board_obj.get('info', {}), dict) else {}
                board_hardware = board_obj.get('hardware', {}) if isinstance(board_obj.get('hardware', {}), dict) else {}
                board_data = {
                    "model": board_info.get('machine', board_hardware.get('Model', board_hardware.get('Module', board_obj.get('model', 'Unknown Jetson')))),
                    "jetpack": board_info.get('jetpack', board_obj.get('jetpack', board_obj.get('L4T', 'Unknown'))),
                    "nvpmodel": nvpmodel_value,
                    "jetsonClocks": jetson_clocks_value,
                    "uptime": uptime_value
                }

                # Combine all data
                status = {
                    "cpu": cpu_data,
                    "memory": memory_data,
                    "gpu": gpu_data,
                    "engines": engines_data,
                    "power": power_data,
                    "temperature": temp_data,
                    "board": board_data,
                    "timestamp": time.time()
                }

                # Write to tmp file atomically
                fd, temp_path = tempfile.mkstemp(dir='/tmp')
                with os.fdopen(fd, 'w') as f:
                    json.dump(status, f)
                # 确保权限允许其他用户（如容器内用户）读取
                os.chmod(temp_path, 0o644)
                os.replace(temp_path, OUTPUT_FILE)

            except Exception as e:
                print(f"Error reading jtop stats: {e}")
                traceback.print_exc()

            # Update frequency (1Hz)
            time.sleep(1)

if __name__ == "__main__":
    main()
