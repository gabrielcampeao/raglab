# Postgres connection pool saturation

## Symptoms and paging alerts

Checkout and ledger workers emit `SQLSTATE 53300` (`too many connections for role atlas_app`) and p99 write latency climbs past 1200 milliseconds. The alert `PgBouncerWaitingClientsHigh` fires whenever waiting client sessions stay above 80 for two consecutive minutes.

## Diagnosis

Inspect active versus idle transactional sessions on the primary instance before restarting any application pods:

```sql
SELECT state, wait_event_type, count(*)
FROM pg_stat_activity
WHERE usename = 'atlas_app'
GROUP BY state, wait_event_type;
```

If `idle in transaction` exceeds 30 sessions, a downstream HTTP call is likely executing inside an open database transaction block.

## Immediate mitigation

Switch PgBouncer `pool_mode` to `transaction` on the affected pool and cap `default_pool_size` at 60 per primary core group so background workers cannot starve checkout writes. Terminate sessions stuck in `idle in transaction` for more than 30 seconds by setting `idle_in_transaction_session_timeout = '30s'` via `ALTER ROLE atlas_app`.

## Escalation criteria

Page the database reliability lead if primary CPU exceeds 85 percent after pool capping or if `pg_stat_ssl` connection churn exceeds 500 handshakes per second.
