# Redis hot key expiration and cache stampede

## Symptoms and paging alerts

Read replicas on the merchant profile database spike from 15 percent to 95 percent CPU within seconds while Redis hit ratio drops below 70 percent. Latency on `GET /v2/merchants` jumps above 900 milliseconds.

## Diagnosis

Run `redis_cli --hotkeys` against the replica telemetry port or inspect proxy metrics for a single key prefix receiving more than 25000 reads per second right as its TTL reaches zero. Simultaneous cache misses cause every application worker to query Postgres for the identical merchant configuration row.

## Immediate mitigation

Enable request coalescing (`singleflight`) in the merchant service runtime flag `cache_singleflight_enabled = true` so only one worker recomputes a missing key per pod while concurrent callers wait on the shared promise. Add random TTL jitter of 10 to 25 percent around the 300 second base expiration and turn on probabilistic early refresh for keys accessed more than 100 times per minute.
