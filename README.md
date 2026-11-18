# MLOps Kubernetes Stack

A runnable reference project for the **model build → container → Kubernetes rollout → autoscaling → observability → rollback** lifecycle.

The model is intentionally tiny and synthetic. The point of the repository is the operational path around it.

## Architecture

```mermaid
flowchart LR
    D[Synthetic training data] --> T[train_model.py]
    T --> A[Model artifact]
    A --> I[Non-root Docker image]
    I --> S[Kubernetes Service]
    S --> P1[FastAPI pod]
    S --> P2[FastAPI pod]
    HPA[HPA CPU policy] --> P1
    HPA --> P2
    P1 --> M[/Prometheus metrics/]
    P2 --> M
    CI[GitHub Actions] --> T
    CI --> I
    CI --> V[Manifest validation]
    R[Release helper] --> S
    R -->|failed rollout| RB[Rollback]
```

## What is implemented

- deterministic scikit-learn training artifact;
- FastAPI prediction service;
- separate liveness, readiness and startup probes;
- Prometheus counters, histogram latency and model-readiness gauge;
- non-root Docker image;
- Kubernetes Deployment + ClusterIP Service + HPA;
- zero-unavailable rolling-update strategy;
- CPU requests/limits and scale-up/scale-down stabilization policies;
- PodDisruptionBudget;
- ingress/egress NetworkPolicy;
- dropped Linux capabilities, read-only root filesystem and seccomp runtime-default;
- application-aware pre-stop drain and termination grace budget;
- rollout helper with automatic rollback when deployment status fails;
- CI that runs tests, Ruff, manifest checks, Docker build and container health/metrics smoke tests.

## Local run

```bash
pip install -r requirements.txt
python train_model.py
uvicorn app:app --reload
```

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
curl http://localhost:8000/metrics
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features":[0.2,0.4,0.8,0.1]}'
```

## Build the container

```bash
docker build -t ml-model-service:local .
docker run --rm -p 8000:8000 ml-model-service:local
```

The image trains the public demo artifact at build time and then runs as UID `10001`.

## Kubernetes

```bash
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/resilience.yaml
```

The service starts with two replicas. HPA can scale to eight replicas based on CPU utilization. Scale-down is intentionally slower than scale-up to reduce oscillation.

### Release and rollback

Use immutable tags or image digests in a real deployment:

```bash
python scripts/release.py deploy \
  --image ghcr.io/cagataykavas/mlops-kubernetes-stack@sha256:<digest> \
  --namespace ml
```

If `kubectl rollout status` fails, the helper requests `kubectl rollout undo` and waits for the previous revision to become healthy.

Manual rollback is also available:

```bash
python scripts/release.py rollback --namespace ml
```

## Health and drain semantics

`/health/live` only answers whether the process is alive. `/health/ready` additionally requires the model artifact to be loaded. Kubernetes therefore stops routing traffic to an unready model without necessarily restarting a healthy process.

Before terminating a pod, the pre-stop hook sends a loopback-only request to
`POST /health/drain`. The service atomically closes prediction admission, readiness begins
returning `503`, and the hook waits up to 20 seconds for already admitted predictions. The
30-second pod grace period leaves a separate budget for Uvicorn shutdown after the drain call.
Prometheus exposes both `ml_accepting_predictions` and `ml_inflight_predictions` so a rollout can
distinguish an intentional drain from an unloaded model.

The drain endpoint trusts the pod loopback boundary and this manifest runs one Uvicorn process per
pod. A multi-worker server would need a shared admission state or one drain signal per worker. The
bounded wait also cannot prove that a timed-out downstream side effect was cancelled; mutating
inference dependencies still require idempotency and reconciliation.

## Observability

`/metrics` is Prometheus text format and exposes:

- `ml_predictions_total{predicted_class,model_version}`;
- `ml_prediction_latency_seconds` histogram;
- `ml_model_ready` gauge.
- `ml_accepting_predictions` drain-state gauge;
- `ml_inflight_predictions` active-work gauge.

This gives the repository enough structure to discuss request rate, latency distributions, rollout comparison by model version and readiness alarms.

## Reliability and security choices

- `maxUnavailable: 0` keeps capacity during rolling updates.
- PDB protects voluntary disruptions from taking every replica down.
- startup/readiness/liveness probes have different responsibilities.
- pre-stop closes application admission and waits within a smaller budget than pod termination.
- NetworkPolicy narrows traffic instead of leaving every pod implicitly reachable.
- the container is non-root, cannot escalate privileges, drops capabilities and uses a read-only root filesystem.

## Interview topics

This project is designed to support concrete discussion of:

**Docker layers · Kubernetes Deployment/Service · probes · HPA · requests vs limits · rolling updates · rollback · PDB · NetworkPolicy · Prometheus · immutable image tags · CI/CD · graceful termination · readiness vs liveness.**
