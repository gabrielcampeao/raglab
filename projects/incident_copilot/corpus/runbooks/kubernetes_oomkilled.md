# Kubernetes pod eviction and exit code 137

## Symptoms and paging alerts

Pods in namespace `payments_prod` restart repeatedly with termination reason `OOMKilled` and container exit code `137`. Pod readiness drops below the 80 percent disruption budget during peak batch reconciliation.

## Diagnosis

Exit code `137` means the Linux kernel cgroup memory killer sent `SIGKILL` (`128 + 9`) because container RSS plus page cache crossed `resources.limits.memory`. Compare `container_memory_working_set_bytes` against runtime heap metrics to distinguish an unbounded Python object retention leak from native C allocator fragmentation.

## Immediate mitigation

Raise container memory limits by 35 percent as a temporary stabilization step while pinning `MALLOC_ARENA_MAX=2` on native worker pools to curb arena bloat. Before rolling all replicas, capture an allocation snapshot from one degraded canary pod using `py_spy` or `tracemalloc` upload to the incident bucket.
