# Outbound payment webhook delivery backlog

## Symptoms and paging alerts

Merchant webhook delivery p95 delay exceeds 120 seconds and outbound queue `webhook_dispatch_pending` grows past 250000 jobs.

## Diagnosis

Group pending deliveries by destination host. A single degraded merchant endpoint timing out at 15 seconds can monopolize shared worker threads when per tenant concurrency caps are disabled.

## Immediate mitigation

Activate per merchant concurrency isolation (`max_inflight_per_endpoint = 8`) and trip the endpoint circuit breaker after 20 consecutive 5xx or timeout responses. Retries follow exponential backoff starting at 15 seconds up to 6 hours while preserving the original `X_Atlas_Signature` timestamp header and replay window rules.
