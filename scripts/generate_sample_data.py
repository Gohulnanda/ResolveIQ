"""Generate data/sample_tickets.csv: SYNTHETIC, template-based training tickets.

These are not real tickets. Each ticket is a title plus two description sentences drawn
at random (seeded) from per-category phrase pools. About 12% of tickets also receive a
sentence from a different category, and about 20% a neutral sentence, so classes overlap
somewhat. Because the text comes from a small phrase pool, hold-out scores on this data
are optimistic; see data/challenge_tickets.csv for hand-written tickets kept out of training.

Usage:  python scripts/generate_sample_data.py
"""

from __future__ import annotations

import random
from pathlib import Path

import pandas as pd

SEED = 42
PER_CATEGORY = 55
OUT_PATH = Path(__file__).resolve().parents[1] / "data" / "sample_tickets.csv"

POOLS: dict[str, tuple[list[str], list[str]]] = {
    "Network": (
        ["Cannot connect to VPN", "Wi-Fi keeps dropping", "No internet access on my laptop",
         "Very slow network speed", "Cannot reach internal websites", "VPN disconnects every few minutes",
         "DNS error when opening websites", "Network drive unreachable from office",
         "Ethernet port not working at desk", "Intranet site times out"],
        ["The VPN client shows a timeout error after I enter my credentials.",
         "Wi-Fi disconnects every ten minutes and I have to reconnect manually.",
         "The browser says DNS_PROBE_FINISHED_NO_INTERNET on every site.",
         "Other people on my floor also report the network is very slow.",
         "Ping to the gateway shows packet loss and high latency.",
         "The ethernet cable is plugged in but the link light is off.",
         "I can open public websites but internal sites time out.",
         "The connection works on the guest network but not the corporate SSID.",
         "The access point in the meeting room seems to have no signal.",
         "Mapped network drives show a red X after I connect remotely."],
    ),
    "Hardware": (
        ["Laptop will not turn on", "Blue screen error on startup", "Printer is offline",
         "Monitor shows no signal", "Keyboard keys not working", "Laptop battery drains quickly",
         "Mouse not responding", "Laptop fan very loud and hot", "Docking station not detected",
         "Printer paper jam error"],
        ["The power light does not come on even when the charger is connected.",
         "The screen flickers and then goes black randomly.",
         "The printer shows offline although it is powered on and has paper.",
         "The device makes a clicking noise from the hard drive.",
         "USB ports do not recognise any device I plug in.",
         "The battery drops from 100 to 20 percent in under an hour.",
         "The laptop gets very hot and shuts down by itself.",
         "The external monitor is blank even though the cable is secure.",
         "A few keys on the keyboard do not register when pressed.",
         "The webcam is not detected by any application."],
    ),
    "Software": (
        ["Excel crashes when opening large file", "Application freezes on launch",
         "Cannot install software update", "License activation failed", "PDF editor not working",
         "Application error after update", "Browser extension breaks the web app",
         "Software not installing error 1603", "Laptop very slow after update",
         "Design tool closes unexpectedly"],
        ["It shows a not responding message and then closes without saving.",
         "The installer fails with error code 1603 near the end.",
         "After the latest update the program crashes at startup.",
         "The license key is rejected saying it was already activated.",
         "Reinstalling did not fix the crash and the same error appears.",
         "The application uses all memory and the computer becomes very slow.",
         "A dialog says a required DLL file is missing.",
         "Task manager shows the process using one hundred percent CPU.",
         "The add-in fails to load every time I open the application.",
         "I need a newer version installed but the software center shows an error."],
    ),
    "Account and Access": (
        ["Forgot my password", "Account locked after failed attempts", "Need access to shared folder",
         "MFA code not working", "Cannot log in to HR portal", "Access request for finance system",
         "Permission denied on SharePoint site", "New joiner needs account setup",
         "Password expired and cannot reset", "Authenticator app lost with new phone"],
        ["It says my account is locked and to contact the administrator.",
         "I changed phones and the authenticator codes are no longer accepted.",
         "The self service reset page sends no email with the reset link.",
         "I get an access denied message when opening the project folder.",
         "My manager approved access but the permissions have not been applied.",
         "Single sign on loops back to the login page.",
         "The new employee starts Monday and needs a user account and groups.",
         "The password was rejected even though I typed it correctly.",
         "I am no longer able to sign in since my role changed.",
         "I need to be added to the security group for the reporting tool."],
    ),
    "Database": (
        ["Database connection timeout", "Slow SQL queries on reporting database",
         "Cannot connect to SQL Server", "Deadlock errors in application",
         "Database disk space almost full", "Replication lag on read replica",
         "Login failed for database user", "Stored procedure runs very slowly",
         "Backup job failed overnight", "Too many connections error"],
        ["The application logs show connection pool exhausted errors.",
         "A query that used to take seconds now takes several minutes.",
         "The nightly backup job failed with an error about insufficient space.",
         "Transactions are rolled back because of deadlock victims.",
         "The read replica is several minutes behind the primary instance.",
         "Login failed for the service account on the SQL instance.",
         "The tablespace is ninety five percent full and growing.",
         "Reports time out when they query the orders table.",
         "Connections to port 1433 are refused from the app server.",
         "The index rebuild job has been running for hours without finishing."],
    ),
    "Email and Collaboration": (
        ["Outlook not receiving emails", "Cannot send email attachments",
         "Teams meeting audio not working", "Shared mailbox missing from Outlook",
         "Calendar invites not syncing", "Mailbox is full", "Zoom camera not working in meetings",
         "Emails going to spam", "Outlook keeps asking for password",
         "Teams chat messages not delivering"],
        ["Outlook shows disconnected in the status bar.",
         "Messages stay in the outbox and are not sent.",
         "Colleagues cannot hear me during Teams calls.",
         "The shared mailbox I had last week no longer appears.",
         "Meeting invites I accept do not show on my phone calendar.",
         "I received a warning that my mailbox quota is exceeded.",
         "Large attachments bounce back with a size limit error.",
         "Teams shows a red banner saying messages may be delayed.",
         "Outlook keeps prompting for credentials every few minutes.",
         "Screen sharing in the video meeting shows a black screen."],
    ),
    "Other": (
        ["Request to move desk and phone", "Question about IT policy", "Schedule training session",
         "Need conference room equipment setup", "Equipment return for leaving employee",
         "Request quote for new monitors", "Where to find IT guidelines",
         "Onboarding checklist question", "Request for visitor wifi code",
         "General question about helpdesk hours"],
        ["I would like to know who to contact about this.",
         "Please advise on the procedure and expected timeline.",
         "This is not urgent and can be handled when convenient.",
         "A colleague is relocating to the second floor next month.",
         "We are planning an event and need support from the team.",
         "I could not find any documentation on the intranet about this.",
         "Can someone confirm the process for returning company equipment?",
         "Please let me know the working hours of the service desk.",
         "We need an estimate for equipment for the new team.",
         "Just checking whether a training slot is available next week."],
    ),
}

NEUTRAL = [
    "This started this morning.",
    "I have already restarted my computer.",
    "Thanks in advance for the help.",
    "It was working fine last week.",
    "I am working from the main office today.",
]


def generate() -> pd.DataFrame:
    rng = random.Random(SEED)
    rows: list[dict[str, str]] = []
    categories = list(POOLS)
    for category, (titles, sentences) in POOLS.items():
        seen: set[str] = set()
        while len(seen) < PER_CATEGORY:
            title = rng.choice(titles)
            parts = rng.sample(sentences, 2)
            if rng.random() < 0.12:  # cross-category noise
                other = rng.choice([c for c in categories if c != category])
                parts.insert(rng.randrange(len(parts) + 1), rng.choice(POOLS[other][1]))
            if rng.random() < 0.20:
                parts.append(rng.choice(NEUTRAL))
            description = " ".join(parts)
            key = f"{title}|{description}"
            if key in seen:
                continue
            seen.add(key)
            rows.append({"title": title, "description": description, "category": category})
    rng.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"SYN-{i:04d}" for i in range(1, len(df) + 1)])
    df["source"] = "synthetic-template"
    return df


if __name__ == "__main__":
    frame = generate()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(frame)} rows to {OUT_PATH}")
    print(frame["category"].value_counts().to_string())
