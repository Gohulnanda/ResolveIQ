---
id: KB-014
title: Database connection timeouts, refused connections and login failures
category: Database
tags: database, sql server, connection timeout, connection refused, login failed, connection pool, port 1433, too many connections
---
## Summary
An application cannot connect to its database server, connections time out, the pool is exhausted, or the database login is rejected.

## Symptoms
Logs show connection timeout, connection refused, pool exhausted or too many connections. A service account gets login failed errors. The database host cannot be reached on its port from the application server.

## Likely causes
- The database service is stopped or the host is unreachable.
- A firewall or security rule blocking the database port.
- Connections not being released, exhausting the pool or the server limit.
- An expired or changed service account password or a missing login permission.

## Resolution steps
1. Check whether the database service is running and whether the host answers from the application server.
2. Test the database port from the application server to separate a network block from a database fault.
3. Review the error text: a refused connection points to the service or firewall, a login error points to credentials or permissions.
4. Verify that the service account password has not expired and that the login still has access to the database.
5. Compare the number of active connections with the configured maximum and look for sessions that never close.
6. Restart the affected application pool, not the database, once the cause is known and record the time.

## Escalate when
A production database is unreachable, the connection count stays at its limit, or the issue follows a network or firewall change.
