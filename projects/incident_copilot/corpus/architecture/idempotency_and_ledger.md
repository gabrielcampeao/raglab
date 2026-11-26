# Architecture record: double entry ledger and idempotency invariants

## Context and invariants

Every monetary movement in Atlas Pay writes balanced debit and credit rows inside `ledger_entries` within a single Postgres transaction under `REPEATABLE READ` or `SERIALIZABLE` isolation. Account balances derive from immutable journal entries; updating or deleting a posted ledger row is forbidden by database trigger.

## Idempotency contract

Mutating endpoints require a client supplied `Idempotency_Key` header up to 128 ASCII characters. Keys are scoped per merchant account and retained for 24 hours. Replaying a key with an identical request payload hash returns the stored HTTP status and response body verbatim; replaying the same key with a modified payload hash returns `HTTP 422` (`idempotency_payload_mismatch`).
