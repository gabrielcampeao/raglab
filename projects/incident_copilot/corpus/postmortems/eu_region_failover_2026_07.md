# Postmortem: EU primary database failover and stale DNS traffic

## Incident summary

On July 9, 2026, an unplanned storage controller reset in Frankfurt triggered automated Postgres failover to replica `eu_central_1b`. Write availability recovered in 38 seconds, but 11 percent of API pods continued routing writes to the demoted node for 6 minutes.

## Root cause

Legacy worker pools cached resolved database host IPs at startup instead of respecting the 5 second internal DNS TTL. In addition, asynchronous replication lag stood at 42 seconds on the third reporting replica, which briefly served stale merchant balances until read routing health checks fenced replicas with lag above 5 seconds.

## Corrective actions

All database clients now connect exclusively through local PgBouncer sidecars that re resolve upstream endpoints on every server connection reset, and Patroni fencing hooks drop client traffic on demoted nodes via iptables before promoting the standby.
