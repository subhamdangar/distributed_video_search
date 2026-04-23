# Distributed Video Search System — Demo Troubleshooting Guide

## 🎯 Setup Overview

Each group:

- 1 Scheduler (Ubuntu machine)
- 2 Workers (Ubuntu machines)

Total:

- 4 groups → 12 machines

---

## ✅ Step 0: Pre-check (Before Demo Starts)

### On ALL machines:

```bash
python --version        # Should be 3.10.x
pip list                # Ensure same packages installed
```

### Check project runs locally:

```bash
python main.py
```

If this fails → fix locally first.

---

## 🌐 Step 1: Network Verification (MOST COMMON ISSUE)

### 🔍 Check and set IP addresses

#### On Scheduler:

```bash
ifconfig | grep inet
```

Look for:

```text
inet 10.x.x.x OR 192.168.x.x
```

#### On Workers:

```bash
ifconfig | grep inet
```

---

### ✅ Test connectivity

From Worker:

```bash
ping <scheduler_ip>
```

### ❌ Problem:

```text
Destination Host Unreachable
```

### 🧠 Cause:

- Different WiFi networks
- Wrong IP used

### ✅ Fix:

- Connect all machines to SAME WiFi
- Use correct IP (not 127.0.0.1)

---

## 🔥 Step 2: Start Scheduler

### ⚠️ Important: Start scheduler with PYTHONPATH inline (prevents import issues)

#### 🔧 Step 2.1 — Stop any running scheduler

```bash
# If running, stop with Ctrl + C
```

#### 🔧 Step 2.2 — Start scheduler with PYTHONPATH inline

```bash
PYTHONPATH=/path/to/your/project dask scheduler
```

Example:

```bash
PYTHONPATH=/home/user/distributed-video-search dask scheduler
```

#### ⚠️ Why this matters

- ❌ This may NOT propagate correctly:

```bash
export PYTHONPATH=/path/to/project
dask scheduler
```

- ✔ This guarantees the scheduler process sees it:

```bash
PYTHONPATH=/path/to/project dask scheduler
```

#### 🔧 Step 2.3 — Verify scheduler started

Expected:

```text
Scheduler at: tcp://<IP>:8786
Dashboard at: http://<IP>:8787
```

---

## 🔧 Step 3: Connect Workers

On each Worker:

```bash
dask worker tcp://<scheduler_ip>:8786
```

---

### ✅ Verify connection

On Scheduler terminal:

```text
Workers: 2
```

OR open browser:

```text
http://<scheduler_ip>:8787
```

---

## 🚫 Problem 1: Worker not connecting

### Symptom:

```text
Waiting to connect...
```

### Causes:

- Wrong IP
- Firewall blocking
- Scheduler not running

### Fix:

```bash
ping <scheduler_ip>   # must work
```

Disable firewall (temporary):

```bash
sudo ufw disable
```

---

## 🚫 Problem 2: ModuleNotFoundError (agents)

### Symptom:

```text
No module named 'agents'
```

### Fix (quick):

```bash
cd /path/to/project
export PYTHONPATH=$(pwd)
dask worker tcp://<scheduler_ip>:8786
```

---

## 🚫 Problem 3: Python version mismatch

### Symptom:

```text
VersionMismatchWarning
```

### Fix:

```text
IGNORE (safe)
```

---

## 🔍 Step 4: Verify System is Working

### Check 1: Scheduler dashboard

```text
http://<scheduler_ip>:8787
```

### Check 2: Worker logs

```text
Stage 1 START
Stage 1 DONE
```

### Check 3: Final output

```text
RESULT
```

---

## 🎯 Final Demo Line

> “We use a scheduler–worker architecture. Tasks are distributed across workers, executed concurrently, and results are aggregated efficiently.”

---

## ✅ You are safe if:

✔ Ping works  
✔ Workers connected  
✔ PYTHONPATH set  
✔ Scheduler running

---

## 🔧 Cross-Platform Notes (Windows + macOS)

### Windows (PowerShell)

```powershell
$env:PYTHONPATH="C:\path\to\project"
dask scheduler
```

### macOS / Linux

```bash
export PYTHONPATH=$(pwd)
python -m distributed.cli.dask_worker tcp://<scheduler_ip>:8786
```

### Important

- Folder names can differ across machines
- Structure MUST be identical:

```text
agents/
config/
utils/
main.py
```

- Always run from project root

---

**End of Guide**