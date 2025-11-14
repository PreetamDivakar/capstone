#!/usr/bin/env python3
import io
import os
import zipfile
from datetime import datetime, timezone
from typing import Dict, Tuple

from flask import Flask, request, jsonify, render_template, send_file

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

DEFAULT_WORKLOAD = {
    "cpu_cores": 1.0,
    "memory_mb": 1024.0,
    "latency_sensitive": 0,
    "execution_time": 60.0,
    "data_size_mb": 100.0,
}

SERVERLESS_MEMORY_MIN = 128
SERVERLESS_MEMORY_MAX = 10240
SERVERLESS_TIMEOUT_MAX = 900


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


def _compute_costs(cpu: float, mem_mb: float, time_s: float) -> Tuple[float, float, float]:
    """Return (cost_traditional, cost_serverless, cost_ratio)."""
    mem_gb = mem_mb / 1024.0
    runtime_hr = time_s / 3600.0
    cost_traditional = ((cpu * P_CPU) + (mem_gb * P_MEM)) * runtime_hr
    cost_serverless = P_REQ + (P_GB_SEC * mem_gb * time_s)
    cost_ratio = (cost_serverless / cost_traditional) if cost_traditional else None
    return cost_traditional, cost_serverless, cost_ratio


def _recommend_platform(cpu: float, mem_mb: float, latency: int, time_s: float) -> str:
    """Return 'Serverless' or 'Traditional' using the same heuristic as /predict."""
    _, _, cost_ratio = _compute_costs(cpu, mem_mb, time_s)
    short_runtime_threshold_s = 300
    small_cpu_threshold = 2.0
    small_mem_threshold_mb = 2048
    cost_ratio_threshold = 5.0

    use_serverless = False
    if (time_s <= short_runtime_threshold_s and cpu <= small_cpu_threshold and mem_mb <= small_mem_threshold_mb and latency == 0):
        if cost_ratio is None or (cost_ratio is not None and cost_ratio < cost_ratio_threshold):
            use_serverless = True
    return "Serverless" if use_serverless else "Traditional"


def _merge_with_defaults(data: Dict) -> Dict:
    merged = DEFAULT_WORKLOAD.copy()
    if data:
        for key in merged:
            if data.get(key) is not None:
                merged[key] = data[key]
    return merged


def _normalize_payload(data: Dict) -> Tuple[Dict[str, float], str]:
    """Return normalized payload dict and recommended platform."""
    merged = _merge_with_defaults(data or {})
    cpu, mem_mb, latency, time_s, data_size_mb = parse_and_validate_payload(merged)
    payload = {
        "cpu_cores": cpu,
        "memory_mb": mem_mb,
        "latency_sensitive": latency,
        "execution_time": time_s,
        "data_size_mb": data_size_mb,
    }
    platform = (data or {}).get("ideal_platform")
    if not platform:
        platform = _recommend_platform(cpu, mem_mb, latency, time_s)
    return payload, platform


def render_deployment_files(payload: Dict[str, float], recommended_platform: str) -> Dict[str, str]:
    """
    Build deployment file contents keyed by filename for the given payload/platform.
    """
    platform_key = (recommended_platform or "Traditional").strip().lower()
    platform_name = "Serverless" if platform_key == "serverless" else "Traditional"

    cpu = payload["cpu_cores"]
    mem_mb = payload["memory_mb"]
    exec_time = payload["execution_time"]

    mem_limit_compose = f"{int(max(128, round(mem_mb)))}M"
    cpu_limit_compose = f"{max(0.1, round(cpu, 2))}"

    serverless_memory = int(min(SERVERLESS_MEMORY_MAX, max(SERVERLESS_MEMORY_MIN, round(mem_mb))))
    serverless_timeout = int(min(SERVERLESS_TIMEOUT_MAX, max(1, round(exec_time))))

    readme = f"""# Deployment Bundle

This bundle targets the **{platform_name}** strategy recommended by the orchestrator.

## Contents
- `deploy_local.sh`: run a local build/test cycle.
- `Dockerfile`: minimal Python container entrypoint.
- `docker-compose.yml`: maps CPU/memory limits to your workload.
- Additional files tailored to the {platform_name} path.

Customize the files as needed before deploying to your environment.
"""

    deploy_script = """#!/bin/bash
set -euo pipefail

echo "[deploy] Building image..."
docker build -t orchestrator-app .

echo "[deploy] Running docker-compose (override docker-compose.yml to customize)..."
docker-compose up --build
"""

    dockerfile = """# Minimal Python runtime
FROM python:3.11-slim
WORKDIR /app
COPY . /app

# Install runtime deps here, e.g.: RUN pip install -r requirements.txt

CMD ["python", "app.py"]
"""

    docker_compose = f"""version: "3.9"
services:
  orchestrator-job:
    build: .
    environment:
      CPU_CORES: "{cpu}"
      MEMORY_MB: "{mem_mb}"
      EXECUTION_TIME: "{exec_time}"
    deploy:
      resources:
        limits:
          cpus: "{cpu_limit_compose}"
          memory: "{mem_limit_compose}"
"""

    files = {
        "README.md": readme,
        "deploy_local.sh": deploy_script,
        "Dockerfile": dockerfile,
        "docker-compose.yml": docker_compose,
    }

    if platform_name == "Serverless":
        lambda_handler = """import json

P_CPU = 0.0316
P_MEM = 0.0045
P_GB_SEC = 0.00001667
P_REQ = 0.0000002


def _compute_costs(payload):
    mem_gb = payload["memory_mb"] / 1024.0
    runtime_hr = payload["execution_time"] / 3600.0
    cost_traditional = ((payload["cpu_cores"] * P_CPU) + (mem_gb * P_MEM)) * runtime_hr
    cost_serverless = P_REQ + (P_GB_SEC * mem_gb * payload["execution_time"])
    cost_ratio = cost_serverless / cost_traditional if cost_traditional else None
    return cost_traditional, cost_serverless, cost_ratio


def lambda_handler(event, _context):
    cpu = float(event.get("cpu_cores", 1.0))
    memory_mb = float(event.get("memory_mb", 1024))
    exec_time = float(event.get("execution_time", 60))
    latency = int(event.get("latency_sensitive", 0))

    payload = {
        "cpu_cores": cpu,
        "memory_mb": memory_mb,
        "execution_time": exec_time,
        "latency_sensitive": latency,
    }
    cost_traditional, cost_serverless, cost_ratio = _compute_costs(payload)
    return {
        "statusCode": 200,
        "body": json.dumps({
            "payload": payload,
            "cost_traditional": cost_traditional,
            "cost_serverless": cost_serverless,
            "cost_ratio": cost_ratio,
        }),
    }
"""

        serverless_yml = f"""service: orchestrator-serverless
provider:
  name: aws
  runtime: python3.11
  region: us-east-1
  timeout: {serverless_timeout}
  memorySize: {serverless_memory}

functions:
  predict:
    handler: lambda_handler.lambda_handler
"""

        files["lambda_handler.py"] = lambda_handler
        files["serverless.yml"] = serverless_yml
    else:
        run_job = f"""import os
import subprocess

CPU = os.getenv("CPU_CORES", "{cpu}")
MEM_MB = os.getenv("MEMORY_MB", "{mem_mb}")
EXEC_TIME = os.getenv("EXECUTION_TIME", "{exec_time}")

print("Running traditional workload with:")
print(f"  CPU cores: {{CPU}}")
print(f"  Memory MB: {{MEM_MB}}")
print(f"  Execution time (s): {{EXEC_TIME}}")

subprocess.run(["python", "app.py"], check=False)
"""
        files["run_job.py"] = run_job

    return files


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)
    try:
        cpu, mem_mb, latency, time_s, data_size_mb = parse_and_validate_payload(data)
    except ValueError as e:
        return jsonify({'error': str(e)}), 400

    cost_traditional, cost_serverless, cost_ratio = _compute_costs(cpu, mem_mb, time_s)
    platform = _recommend_platform(cpu, mem_mb, latency, time_s)

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


@app.route('/deploy', methods=['POST'])
def deploy_bundle():
    data = request.get_json(silent=True) or {}
    try:
        payload, platform = _normalize_payload(data)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    files = render_deployment_files(payload, platform)
    mem_zip = io.BytesIO()
    with zipfile.ZipFile(mem_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
        for name, content in files.items():
            zf.writestr(name, content)
    mem_zip.seek(0)

    platform_slug = (platform or "traditional").lower().replace(" ", "_")
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    filename = f"deploy_bundle_{platform_slug}_{timestamp}.zip"

    return send_file(
        mem_zip,
        mimetype='application/zip',
        as_attachment=True,
        download_name=filename
    )


if __name__ == '__main__':
    app.run(debug=True)
