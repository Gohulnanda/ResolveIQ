---
id: KB-010
title: Multi-factor authentication codes rejected or authenticator lost
category: Account and Access
tags: mfa, authenticator, verification code, new phone, single sign on, token, two factor
---
## Summary
Verification codes are rejected, the authenticator app was lost with a phone, or sign-in loops back to the login page.

## Symptoms
Six-digit codes are rejected. The user changed phones and the app no longer has the account. Single sign-on repeatedly returns to the login page.

## Likely causes
- The phone clock is out of sync, which invalidates time-based codes.
- The authenticator was not transferred to the new phone.
- Stale browser cookies or a cached session breaking single sign-on.
- A disabled or expired MFA registration.

## Resolution steps
1. Set the phone date and time to automatic, then generate a fresh code and try again.
2. Try an alternative verification method such as a text message or backup code if one is registered.
3. For a new phone, ask the service desk to reset the registered method and enroll the new device.
4. Clear browser cookies for the sign-in site, or use a private window, to break a sign-in loop.
5. Confirm you are using the entry for the correct account in the authenticator app.
6. Be ready to verify your identity with your manager or employee details during the reset.

## Escalate when
A user is locked out of every method, MFA prompts arrive that the user did not request, or enrollment fails for a group of users.
