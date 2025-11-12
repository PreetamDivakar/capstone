#!/usr/bin/env python3
import os
from flask import Flask, request, jsonify, render_template

# --- Directories ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEMPLATES_DIR = os.path.join(BASE_DIR, "frontend", "templates")
STATIC_DIR = os.path.join(BASE_DIR, "frontend", "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# --- Cost Constants ---
P_CPU = 0.0316        # $ per CPU core-hour
P_MEM = 0.0045        # $ per GB-hour
P_GB_SEC = 0.00001667 # $ per GB-second (serverless memory-time cost)
P_REQ = 0.0000002     # $ per request (serverless fixed request cost)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/benchmark')
def benchmark_page():
    return render_template('benchmark.html')


def parse_and_validate_payload(data):
    """
    Parse numeric fields and validate.
    Returns tuple (cpu, mem_mb, latency, time_s, data_size_mb) or raises ValueError.
    """
    if data is None:
        raise ValueError("Missing JSON payload.")

    cpu_raw = data.get('cpu_cores', None)
    mem_raw = data.get('memory_mb', None)
    latency_raw = data.get('latency_sensitive', None)
    time_raw = data.get('execution_time', None)
    size_raw = data.get('data_size_mb', None)

    if cpu_raw is None or mem_raw is None or latency_raw is None or time_raw is None or size_raw is None:
        raise ValueError("One or more required fields are missing. Required: cpu_cores, memory_mb, latency_sensitive, execution_time, data_size_mb")

    try:
        cpu = float(cpu_raw)
        mem = float(mem_raw)
        latency = int(latency_raw)
        time_s = float(time_raw)
        data_size_mb = float(size_raw)
    except (TypeError, ValueError):
        raise ValueError("One or more fields have invalid numeric format.")

    if any(v < 0 for v in (cpu, mem, latency, time_s, data_size_mb)):
        raise ValueError("Values must be non-negative.")

    if cpu == 0 and mem == 0 and latency == 0 and time_s == 0 and data_size_mb == 0:
        raise ValueError("All inputs are zero — please provide realistic workload parameters.")

    return cpu, mem, latency, time_s, data_size_mb


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    try:
        cpu, mem_mb, latency, time_s, data_size_mb = parse_and_validate_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    mem_gb = mem_mb / 1024.0
    runtime_hr = time_s / 3600.0

    cost_traditional = ((cpu * P_CPU) + (mem_gb * P_MEM)) * runtime_hr
    cost_serverless = P_REQ + (P_GB_SEC * mem_gb * time_s)

    if cost_traditional == 0:
        cost_ratio = None
    else:
        cost_ratio = cost_serverless / cost_traditional

    short_runtime_threshold_s = 300
    small_cpu_threshold = 2.0
    small_mem_threshold_mb = 2048
    cost_ratio_threshold = 5.0

    use_serverless = False
    if (time_s <= short_runtime_threshold_s and cpu <= small_cpu_threshold and mem_mb <= small_mem_threshold_mb and latency == 0):
        if cost_ratio is None or (cost_ratio is not None and cost_ratio < cost_ratio_threshold):
            use_serverless = True

    platform = "Serverless" if use_serverless else "Traditional"

    out = {
        "cpu_cores": cpu,
        "memory_mb": mem_mb,
        "latency_sensitive": latency,
        "execution_time": time_s,
        "data_size_mb": data_size_mb,
        "cost_traditional": round(cost_traditional, 6),
        "cost_serverless": round(cost_serverless, 6),
        "cost_ratio": (round(cost_ratio, 3) if cost_ratio is not None else None),
        "ideal_platform": platform
    }

    return jsonify(out), 200


@app.route('/benchmark-data')
def benchmark_data():
    """
    Return richer mock benchmark data:
    - platform-level metrics (cost/latency/efficiency)
    - a binary classification example (actual/predicted arrays) used to render confusion matrix & metrics
    """
    data = {
        "platforms": ["Serverless", "Traditional", "Hybrid"],
        "cost": [0.14, 0.12, 0.10],
        "latency_ms": [70, 90, 65],
        "efficiency_percent": [88, 82, 95],
        # Example binary classification arrays (same length) for confusion matrix demo
        "actual": [1,0,1,1,0,0,1,0,1,0,1,0,1,1,0,0,1,0,1,0],
        "predicted": [1,0,1,0,0,1,1,0,1,0,0,0,1,1,0,0,1,0,1,1],
        # Platform-wise sample breakdown (optional; front-end can use it if needed)
        "per_platform": {
            "Serverless": {"cost": 0.14, "latency_ms": 70, "efficiency": 88},
            "Traditional": {"cost": 0.12, "latency_ms": 90, "efficiency": 82},
            "Hybrid": {"cost": 0.10, "latency_ms": 65, "efficiency": 95}
        },
        "hybrid": {"cost": 0.10, "latency_ms": 65, "efficiency_percent": 95}
    }
    return jsonify(data)


if __name__ == '__main__':
    app.run(debug=True)
