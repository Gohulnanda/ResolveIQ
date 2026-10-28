---
id: KB-001
title: VPN connection fails or keeps disconnecting
category: Network
tags: vpn, remote access, tunnel, timeout, disconnect, work from home
---
## Summary
Remote users cannot establish a VPN tunnel, or the tunnel drops repeatedly after connecting.

## Symptoms
The VPN client spins and then reports a timeout after credentials are entered. The tunnel connects but disconnects every few minutes. Internal sites are unreachable even though the client says it is connected.

## Likely causes
- An unstable home internet link or a captive portal on public Wi-Fi.
- An outdated VPN client or an expired client certificate.
- Local firewall or router settings blocking the VPN ports.
- The wrong VPN gateway selected in the client profile.

## Resolution steps
1. Confirm the home internet works by opening a public website before connecting to the VPN.
2. Disconnect fully, close the VPN client, and reconnect using the primary gateway profile.
3. Switch from Wi-Fi to a wired connection, or move closer to the router, and test again.
4. Update the VPN client to the version published in the software center, then restart the laptop.
5. Temporarily disable third-party security software or a home-router VPN passthrough restriction and retry.
6. If a certificate error appears, note the exact message and request certificate renewal.

## Escalate when
Several users cannot connect at the same time, the gateway is unreachable from every network, or certificate errors persist after the client update.
