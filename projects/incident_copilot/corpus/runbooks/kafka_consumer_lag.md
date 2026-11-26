# Kafka consumer lag and rebalance storms

## Symptoms and paging alerts

The alert `KafkaSettlementLagCritical` pages when consumer group `settlement_projector` falls more than 150000 messages behind the log head or partition lag age exceeds 180 seconds.

## Diagnosis

Check whether lag is concentrated on a single hot merchant partition or spread evenly across all 24 partitions. Frequent `Revoking previously assigned partitions` log entries indicate a consumer group rebalance storm caused by batch processing exceeding `max.poll.interval.ms`.

```sql
SELECT partition_id, lag_messages, consumer_owner
FROM ops_kafka_lag_snapshot
WHERE consumer_group = 'settlement_projector'
ORDER BY lag_messages DESC;
```

## Immediate mitigation

Lower `max.poll.records` from 500 to 100 and raise `max.poll.interval.ms` to 600000 so heavy settlement batches finish before the broker evicts the worker from the consumer group. When a poison payload blocks deserialization on one partition, route that offset to `settlement_projector_dlq` and advance the committed offset by one.

## Recovery verification

Confirm rebalance rate drops to zero over 10 minutes and consumer throughput exceeds producer ingress by at least 40 percent until lag clears.
