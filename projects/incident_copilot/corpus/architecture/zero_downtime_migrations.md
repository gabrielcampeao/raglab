# Architecture record: zero downtime Postgres schema migrations

## Expand and contract policy

Production tables serving checkout or ledger traffic never run locking DDL in a single deployment. Schema changes execute in three separate releases: expand (add nullable column or dual write trigger), backfill in throttled batches of 2000 rows with 50 millisecond sleep, and contract (drop legacy column only after zero reads remain).

## Safe indexing and constraints

Every new index must use `CREATE INDEX CONCURRENTLY` outside an explicit transaction block, with `lock_timeout = '2s'` and `statement_timeout = '0'` configured on the migration session. Foreign keys and check constraints are added with `NOT VALID` first so Postgres skips the full table scan under `ACCESS EXCLUSIVE` lock, followed by `VALIDATE CONSTRAINT` in a separate step.
