# MLOps Kubernetes Stack

A compact cloud-native reference stack for serving a machine-learning model behind FastAPI, packaging it with Docker, deploying it to Kubernetes, exposing health/metrics endpoints and validating the whole path in CI.

## Stack

- FastAPI model service
- Docker
- Kubernetes Deployment + Service + HPA
- Prometheus-compatible metrics endpoint
- readiness/liveness probes
- GitHub Actions CI
- reproducible local inference baseline

## Architecture

```text
client
  |
  v
Kubernetes Service
  |
  v
FastAPI model pods  <---- HorizontalPodAutoscaler
  |
  +---- /health
  +---- /metrics
  +---- /predict
```

The public baseline uses a tiny deterministic scikit-learn model and synthetic data so the repository is fully runnable without proprietary artifacts.

## Local run

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```

Then:

```bash
curl http://localhost:8000/health
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{"features":[0.2,0.4,0.8,0.1]}'
```

## Kubernetes

```bash
kubectl apply -f k8s/
```

The manifests demonstrate deployment, service exposure, resource requests/limits, health probes and horizontal autoscaling.
