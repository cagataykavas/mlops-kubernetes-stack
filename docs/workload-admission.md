# Kubernetes workload admission policy

`scripts/validate_manifests.py` is a deterministic, fail-closed pre-deployment
check for the repository's Kubernetes workload contract. It validates each
resource and also checks relationships that schema validation alone does not
cover.

```bash
python scripts/validate_manifests.py --json
```

Exit codes are stable for CI use:

- `0`: the workload is admitted;
- `2`: valid YAML violates policy;
- `3`: the evidence is malformed or cannot be read.

## Enforced contract

For each Deployment, the gate requires at least two replicas, a zero-unavailable
rolling update, explicit non-`latest` images, HTTP startup/readiness/liveness
probes, CPU and memory requests/limits, a termination grace period, non-root
execution, RuntimeDefault seccomp, a read-only root filesystem, disabled
privilege escalation and all Linux capabilities dropped.

The gate then proves that Service and Deployment selectors agree, that
ServiceMonitor selectors and named ports resolve to real Services, and that
every Deployment is covered by an HPA, PodDisruptionBudget and ingress/egress
NetworkPolicy. Duplicate resource identities and orphaned policies fail closed.
Output violations contain stable reason codes, resource identities and field
paths so CI and deployment tooling can consume the evidence without parsing
human prose.

## Trust boundary and limitations

This is a repository admission control, not a Kubernetes API-server admission
webhook. It does not apply organization policy, expand Helm/Kustomize templates,
resolve image tags, inspect image signatures/SBOMs, calculate effective RBAC or
prove that a NetworkPolicy provider enforces the declared rules. An explicit
non-`latest` tag passes this local template check; the release helper should
replace it with a verified digest before production rollout.

The next production step is to validate the fully rendered server-side manifest,
verify the image signature and digest provenance, and enforce the same contract
with Kyverno, Gatekeeper or a validating admission policy.
