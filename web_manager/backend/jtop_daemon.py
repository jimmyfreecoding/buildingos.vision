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
                # 1. CPU Info
                cpu_data = {
                    "usage": 0,
                    "cores": len(jetson.cpu),
                    "details": {}
                }
                total_cpu_usage = 0
                active_cores = 0
                for name, core in jetson.cpu.items():
                    if 'val' in core:
                        total_cpu_usage += core['val']
                        active_cores += 1
                        cpu_data["details"][name] = {
                            "usage": core['val'],
                            "freq": core.get('frq', 0)
                        }
                if active_cores > 0:
                    cpu_data["usage"] = total_cpu_usage / active_cores

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
                gpu = jetson.gpu
                gpu_data = {
                    "util": gpu.get('val', 0),
                    "freq": gpu.get('frq', 0),
                    "freqMax": gpu.get('frq_max', gpu.get('max', 0)),
                    "memUsed": memory_data["ram"]["used"] / (1024*1024), # In MB, mirroring unified RAM
                    "memTotal": memory_data["ram"]["total"] / (1024*1024)
                }

                # 4. Engines (Hardware accelerators)
                engines_data = {}
                for name, engine in jetson.engine.items():
                    if isinstance(engine, dict):
                        engines_data[name] = engine.get('val', 0)
                    elif isinstance(engine, bool):
                        engines_data[name] = 100 if engine else 0
                    else:
                        engines_data[name] = 0

                # 5. Power
                power_data = {
                    "total": jetson.power.get('tot', {}).get('avg', 0), # mW
                    "gpu": jetson.power.get('gpu', {}).get('avg', 0),
                    "cpu": jetson.power.get('cpu', {}).get('avg', 0),
                    "soc": jetson.power.get('soc', {}).get('avg', 0),
                    "cv": jetson.power.get('cv', {}).get('avg', 0),
                    "vdd_in": jetson.power.get('VDD_IN', {}).get('avg', jetson.power.get('vdd_in', {}).get('avg', 0))
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

                board_data = {
                    "model": jetson.board.get('info', {}).get('machine', 'Unknown Jetson'),
                    "jetpack": jetson.board.get('info', {}).get('jetpack', 'Unknown'),
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
                os.replace(temp_path, OUTPUT_FILE)

            except Exception as e:
                print(f"Error reading jtop stats: {e}")
                traceback.print_exc()

            # Update frequency (1Hz)
            time.sleep(1)

if __name__ == "__main__":
    main()
