import io
import os
import sys
import zipfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.app import app, render_deployment_files


def _sample_payload(**overrides):
    payload = {
        "cpu_cores": 1.5,
        "memory_mb": 1024,
        "latency_sensitive": 0,
        "execution_time": 120,
        "data_size_mb": 300,
    }
    payload.update(overrides)
    return payload


def test_render_deployment_files_serverless():
    payload = _sample_payload(memory_mb=20480, execution_time=2000)
    files = render_deployment_files(payload, "Serverless")
    assert "lambda_handler.py" in files
    assert "serverless.yml" in files
    assert "run_job.py" not in files
    assert "memorySize: 10240" in files["serverless.yml"]
    assert "timeout: 900" in files["serverless.yml"]
    assert 'cpus: "1.5"' in files["docker-compose.yml"]


def test_render_deployment_files_traditional():
    payload = _sample_payload(cpu_cores=8, memory_mb=16384, execution_time=1800, latency_sensitive=1)
    files = render_deployment_files(payload, "Traditional")
    assert "run_job.py" in files
    assert "lambda_handler.py" not in files
    assert "serverless.yml" not in files
    compose = files["docker-compose.yml"]
    assert 'cpus: "8"' in compose
    assert 'memory: "16384M"' in compose


def test_deploy_endpoint_creates_zip():
    client = app.test_client()
    resp = client.post("/deploy", json=_sample_payload())
    assert resp.status_code == 200
    assert resp.headers["Content-Type"] == "application/zip"
    disposition = resp.headers.get("Content-Disposition", "")
    assert ".zip" in disposition

    archive = io.BytesIO(resp.data)
    with zipfile.ZipFile(archive) as zf:
        names = zf.namelist()
        assert "README.md" in names
        assert "docker-compose.yml" in names
        assert any(name.startswith("lambda_handler") for name in names)

