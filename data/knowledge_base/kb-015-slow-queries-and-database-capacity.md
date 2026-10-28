---
id: KB-015
title: Slow queries, deadlocks, replication lag and database storage
category: Database
tags: slow query, deadlock, replication lag, backup failed, disk space, index, reporting database, transaction
---
## Summary
Queries and reports are slow or time out, transactions fail with deadlocks, replicas fall behind, or backups fail because storage is full.

## Symptoms
A report that took seconds now takes minutes. Transactions roll back with deadlock errors. A read replica is minutes behind the primary. A nightly backup fails with an insufficient space message. The data file is nearly full.

## Likely causes
- Missing or fragmented indexes, or a changed execution plan.
- Long-running transactions holding locks.
- Heavy write load or a network issue delaying replication.
- Data growth or old backups consuming storage.

## Resolution steps
1. Identify the slowest query from the monitoring or query store and capture its execution plan.
2. Check for long-running or blocking sessions and note which statements are involved.
3. For deadlocks, review the deadlock graph and make sure the application accesses tables in a consistent order.
4. Check the replication status and the network link between primary and replica to measure the lag.
5. Check free space on data, log and backup volumes, and remove expired backups according to the retention policy.
6. Schedule index maintenance outside business hours instead of during peak load.

## Escalate when
The data volume is above ninety percent and growing, backups have failed more than once, or production reports are unavailable.
