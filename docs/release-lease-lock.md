# Fenced Kubernetes releases with Lease

The rollout helper now serializes deploy and rollback operations with a
namespaced `coordination.k8s.io/v1` Lease. Without this boundary, two CI jobs can
set different images concurrently and a failed older job can roll back a newer,
healthy revision.

## Protocol

The helper performs an atomic `create` when no Lease exists. For an existing
Lease it reads `holderIdentity`, `renewTime`, `leaseDurationSeconds`,
`leaseTransitions`, and `metadata.resourceVersion`:

- an unexpired foreign holder causes an immediate fail-closed exit;
- the same holder may renew its Lease without incrementing transitions;
- an expired or explicitly released Lease may be taken over;
- takeover and release use `kubectl replace`, so Kubernetes rejects stale
  `resourceVersion` writes;
- bounded retries handle create/replace races without an unbounded wait;
- release re-reads the Lease and clears it only if ownership still matches.

The default duration is 600 seconds, bounded to 30–3,600 seconds. Timestamps
must be UTC and a future-skew budget prevents a bad clock from pinning the lock.
Lease JSON and holder/name fields have explicit size and syntax constraints.

## Usage and RBAC

Supply a unique pipeline identity; GitHub Actions can use its run attempt:

```bash
python scripts/release.py deploy \
  --image ghcr.io/cagataykavas/mlops-kubernetes-stack@sha256:<digest> \
  --namespace ml \
  --lock-holder "github-${GITHUB_RUN_ID}-${GITHUB_RUN_ATTEMPT}" \
  --lease-seconds 600
```

The deployment identity needs `get`, `create`, and `update` on the single Lease
name in addition to its existing Deployment permissions. A production Role can
scope these verbs with `resourceNames: [ml-model-service-release]` for `get` and
`update`; Kubernetes RBAC cannot restrict `create` by resource name, so creation
permission should remain namespace-scoped and the namespace should be dedicated
or admission-controlled.

Lock failures exit with code `4` and disclose only the exception class, not the
foreign holder identity or raw Kubernetes response.

## Failure behavior

- A crashed job stops renewing and its Lease becomes eligible after the TTL.
- A stale job cannot release a Lease that another holder has acquired.
- A `resourceVersion` conflict is retried during acquisition and surfaced during
  release rather than silently ignored.
- Rollback runs under the same lock as deployment, preventing independent undo
  jobs from racing a live rollout.

## Limitations and next step

The current release operation is expected to finish within the configured TTL;
there is no heartbeat goroutine. Set the TTL above the rollout plus rollback
budget. A production controller should periodically renew, bind the Lease to an
immutable deployment intent digest, record audit events, and use a dedicated
service account with least-privilege RBAC.

Kubernetes Lease is a coordination primitive, not a distributed transaction.
It does not make external image publication, registry mutation, or post-deploy
verification atomic with the rollout.
