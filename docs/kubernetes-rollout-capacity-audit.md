# Kubernetes rollout capacity-envelope audit

A Deployment can be valid, healthy at its current replica count, and still be unable to roll out at the HPA ceiling. `maxSurge` temporarily creates extra pods, while CPU, memory, or pod-count ResourceQuota may have no room for them. Percentage rounding and a conflicting PodDisruptionBudget can make that failure easy to miss in review.

`capacity_audit.py` evaluates these controls as one release boundary. It resolves Kubernetes `IntOrString` values at the HPA maximum, using round-up for `maxSurge`, round-down for Deployment `maxUnavailable`, and round-up for percentage PodDisruptionBudget values. It then checks:

- the current desired replica count is inside HPA bounds;
- the rollout strategy can make progress;
- the rollout's minimum availability is not below the declared disruption objective;
- namespace CPU-request, memory-request, and pod-count quota can hold the HPA maximum plus surge after other workloads' reservations.

The quota `reserved_*` fields intentionally exclude the audited workload. This lets the audit calculate the workload's entire peak requirement instead of double-counting its current pods.

## Evidence contract

Schema `mlops-kubernetes-rollout-capacity/v1` binds the calculation to a namespace, workload, pod-template SHA-256, generation time, Deployment strategy, HPA bounds, PodDisruptionBudget and ResourceQuota snapshot. Evidence must be timezone-aware, fresh, bounded and structurally exact. Duplicate JSON fields, non-finite values, invalid percentages, impossible quota usage and inconsistent replica bounds fail closed.

Reports hash namespace and workload names and do not copy raw identifiers. They expose resource calculations, headroom and stable reason codes so CI or a deployment controller can distinguish policy rejection from malformed evidence.

```bash
python capacity_audit.py rollout-capacity.json --output capacity-report.json
```

Exit codes are `0` for acceptance, `2` for a valid capacity-policy rejection and `3` for malformed evidence. Output replacement is atomic.

## Interpretation and limits

This is a deterministic admission calculation, not a cluster scheduler simulation. ResourceQuota headroom does not prove that any node has a compatible CPU/memory shape, topology, taint/toleration set, affinity, volume attachment, GPU, IP address or image-pull capacity. Deployment-driven pod replacement is not governed by PodDisruptionBudget; the comparison detects a configuration that permits a rollout availability floor below the team's declared voluntary-disruption objective.

The evidence producer remains a trust boundary. Production integration should collect a server-side-rendered Deployment, HPA, PDB and live ResourceQuota usage under one authenticated snapshot, bind the collector identity, and retain both input and report as deployment evidence.

## Next integration step

Add a least-privilege Kubernetes collector for rendered resources and quota usage, then augment the envelope with scheduler preflight evidence for node allocatable resources and topology constraints.
