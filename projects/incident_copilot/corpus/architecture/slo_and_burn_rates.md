# Architecture record: availability SLOs and multi window burn rates

## Service level objectives

Checkout authorization maintains a 30 day rolling availability SLO of `99.95%` (21.6 minutes of monthly error budget) and a p99 latency objective of `450ms` measured at the edge ingress gateway.

## Multi window error budget alerts

Paging on raw 1 minute error spikes causes severe alert fatigue. Atlas Pay pages on call engineers only when both a short window and a long window exceed burn rate thresholds simultaneously:

* Page immediately (Sev1): `14.4x` error budget burn over both `5 minutes` and `1 hour` (consumes 2 percent of monthly budget in one hour).
* Page urgently (Sev2): `6.0x` error budget burn over both `30 minutes` and `6 hours` (consumes 5 percent of monthly budget in six hours).
* Ticket only (Sev3): `1.0x` sustained burn over `24 hours`.
