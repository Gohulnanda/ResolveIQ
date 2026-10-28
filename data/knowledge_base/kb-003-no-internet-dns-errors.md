---
id: KB-003
title: No internet access or DNS errors
category: Network
tags: dns, no internet, website not found, ethernet, ip address, internal sites
---
## Summary
The device shows connected but websites or internal portals do not load, often with DNS errors.

## Symptoms
Browsers report that the server cannot be found, or DNS_PROBE_FINISHED_NO_INTERNET. Public sites work but internal sites time out, or the reverse. A wall socket shows no link light.

## Likely causes
- A stale DNS cache or incorrect DNS servers on the adapter.
- No valid IP address because DHCP failed.
- A damaged patch cable or an inactive wall port.
- An outage of the internal DNS service.

## Resolution steps
1. Try both a public website and an internal portal to see whether only one is failing.
2. Flush the DNS cache from an administrator command prompt and retry.
3. Release and renew the IP address, or disable and re-enable the network adapter.
4. Swap the ethernet cable and try a different wall port, noting the port label.
5. Check that the adapter uses automatic DNS settings unless IT assigned fixed ones.
6. Test from another device on the same port or network to isolate the laptop.

## Escalate when
The wall port stays dead with a known-good cable, several users cannot resolve internal names, or addresses start with 169.254.
