# Postmortem: checkout duplicate charges during mobile retry storm

## Incident summary

On April 14, 2026, 412 mobile checkout requests produced duplicate card captures during a 19 minute regional packet loss event. All duplicate captures were automatically refunded within 4 hours.

## Root cause

Mobile clients retried timed out `POST /v2/charges` calls with the same `Idempotency_Key`, as designed. However, the Redis distributed lock protecting in flight idempotency keys had a fixed 3000 millisecond lease while downstream card network authorization took 4400 milliseconds under packet loss. Lock expiration allowed the second retry to enter authorization concurrently before the first response wrote its ledger record.

## Corrective actions

We moved idempotency state coordination from volatile Redis locks into Postgres unique key constraints backed by monotonic fencing tokens on `idempotency_records`. Any worker whose fencing token is superseded aborts before submitting capture to the card network.
