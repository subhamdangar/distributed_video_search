# Distributed Video Search System
## Demo Setup & Troubleshooting Guide

> **Platform:** Ubuntu Linux &nbsp;|&nbsp; **All commands run from project root**

---

### Setup Overview

| Role | Count per Group | Total (4 groups) |
|------|----------------|------------------|
| Scheduler | 1 Ubuntu machine | 4 machines |
| Workers | 2 Ubuntu machines | 8 machines |
| **Total** | **3 machines** | **12 machines** |

---

## Step 0 — Folder Structure Check

Verify the project folder has the **exact same structure** on every machine.

**Run on ALL machines:**
```bash
ls
```

**Expected output (must contain):**
```
agents/
config/
utils/
main.py
requirements.txt
check_requirements.py
```

> ⚠️ **Important:** Folder names must be **identical** across all machines.
> If any folder is missing, copy the full project before proceeding.

---

## Step 1 — Check Python Version

Run on **all machines**. Python version must match.

```bash
python --version
```

**Expected:**
```
Python 3.10.x
```

> ❌ **Problem:** Version mismatch between machines
>
> ✅ **Fix:** Ensure all machines use the same Python version.
> Use `which python` to confirm the active interpreter.

---

## Step 2 — Scheduler Setup: Verify `main.py` & Create Database

> 🔒 **Scheduler machine only** — Workers do not need this step.

**Verify `main.py` exists:**
```bash
ls main.py
```
**Expected:**
```
main.py
```

**Create the `db/` folder:**
```bash
mkdir -p db
```

**Create the cache database file:**
```bash
touch db/cache.db
```

**Verify `db/` contents:**
```bash
ls db/
```
**Expected:**
```
cache.db
```

> ⚠️ **Important:** This step is **only** for the Scheduler machine.

---

## Step 3 — Create & Activate Virtual Environment

Do this on **every machine** inside the project root.

**Create venv:**
```bash
python -m venv venv
```

**Activate venv:**
```bash
source venv/bin/activate
```

**Install dependencies:**
```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** Always activate venv before running **any** command.
> Your prompt should show `(venv)` at the start of the line.

---

## Step 4 — Get OpenRouter API Key (Free)

OpenRouter provides free LLM API access for query routing.

| # | Action | Detail |
|---|--------|--------|
| 1 | Go to | https://openrouter.ai |
| 2 | Click | Sign Up and create a free account |
| 3 | Open | Dashboard → API Keys |
| 4 | Click | Create Key |
| 5 | Copy | your generated API key |
| 6 | Create | a `.env` file in the project root |
| 7 | Paste | the API key inside the `.env` file |

---

### Example `.env`

```env
OPENROUTER_API_KEY=sk-or-xxxxxxxxxxxxxxxxxxxx
```

---

### Example Project Structure

```text
.
├── .env
├── main.py
├── agents/
├── config/
├── utils/
└── requirements.txt
```

---

## Step 5 — Run Requirement Check

Run on **all machines** to verify all packages are correctly installed.

```bash
python check_requirements.py
```

**Expected output (all passing):**
```
[OK] langchain
[OK] dask
[OK] distributed
[OK] sentence_transformers
[OK] All requirements satisfied.
```

> ❌ **Problem:** `ModuleNotFoundError` or `[FAIL]` for any package
>
> ✅ **Fix:**
> ```bash
> pip install -r requirements.txt   # inside activated venv
> python check_requirements.py      # re-run to verify
> ```

---

## Step 6 — Check Working Directory

> ⚠️ **All commands must be run from the project root.** Verify before each command.

```bash
pwd
```

**Expected (example):**
```
/home/user/distributed-video-search
```

> ⚠️ **Important:** Never run `dask scheduler` / `dask worker` from a sub-directory.
> Always `cd /path/to/project` first.

---

## Step 7 — Network Verification

Run on **Scheduler** and **each Worker** to find the correct LAN IP.

```bash
ifconfig | grep inet
```

**Look for (LAN IP):**
```
inet 10.x.x.x     OR     inet 192.168.x.x
```

> ⚠️ **Important:**
> - Do **NOT** use `127.0.0.1` — this is loopback (local only).
> - Write down the Scheduler's LAN IP — you will need it in Steps 9 and 10.

---

## Step 8 — Test Connectivity

From **each Worker machine**, ping the Scheduler to confirm network reach.

```bash
ping <scheduler_ip>
```

**Expected:**
```
64 bytes from <scheduler_ip>: icmp_seq=1 ttl=64 time=0.4 ms
```

> ❌ **Problem:** `Destination Host Unreachable`
>
> ✅ **Fix:**
> - Connect **all machines** to the **same WiFi / LAN** network.
> - Double-check the IP — use `ifconfig` output, not `127.0.0.1`.

---

## Step 9 — Start Scheduler

> 🔒 **Scheduler machine only.**

Always use `PYTHONPATH` **inline** — not via `export`.

```bash
PYTHONPATH=$(pwd) dask scheduler
```

| ❌ May NOT work | ✅ Guaranteed to work |
|----------------|----------------------|
| `export PYTHONPATH=$(pwd)` then `dask scheduler` | `PYTHONPATH=$(pwd) dask scheduler` |

**Expected output:**
```
Scheduler at:  tcp://<IP>:8786
Dashboard at:  http://<IP>:8787
```

---

## Step 10 — Set Scheduler IP in `config.py`

Update **`config.py` on ALL machines** (Scheduler + Workers) with the Scheduler's IP from Step 7.

```python
SCHEDULER_IP = "10.x.x.x"    # Replace with actual Scheduler IP
```

> ⚠️ **Important:**
> - This must be the **same IP** on every machine.
> - Use the IP from `ifconfig` output on the Scheduler (Step 7).

---

## Step 11 — Connect Workers

Run on **each Worker machine** after Scheduler is confirmed running.

```bash
PYTHONPATH=$(pwd) dask worker tcp://<scheduler_ip>:8786
```

**Verify — Scheduler terminal should show:**
```
Workers: 2
```

Or open the Dask dashboard in a browser on the Scheduler:
```
http://<scheduler_ip>:8787
```

> ❌ **Problem:** `Waiting to connect...` (worker hangs)
>
> ✅ **Fix:**
> - `ping <scheduler_ip>` must succeed from Worker.
> - Confirm Scheduler is running (Step 9).
> - Temporarily disable firewall: `sudo ufw disable`

> ❌ **Problem:** `No module named 'agents'`
>
> ✅ **Fix:**
> - Use `PYTHONPATH=$(pwd)` inline — not via `export`.
> - Confirm `pwd` is the project root before running.

---

## Step 12 — Run `main.py` (Scheduler Only)

Once all workers are connected, run the system from the **Scheduler machine**.

```bash
python main.py
```

**Expected log output (Worker terminals):**
```
Stage 1 START
Stage 1 DONE
...
RESULT
```

> ⚠️ **Important:**
> - Run **only** on Scheduler — never on Worker machines.
> - All Workers must be connected before this step.

---

## ✅ You Are Safe If:

- [ ] Folder structure is identical on all machines
- [ ] `main.py` present on Scheduler; `db/cache.db` created
- [ ] `venv` activated and requirements installed on all machines
- [ ] OpenRouter API key set in `config.py`
- [ ] `check_requirements.py` passes on all machines
- [ ] `pwd` is project root on all machines
- [ ] Ping from Worker to Scheduler works
- [ ] `PYTHONPATH=$(pwd)` used inline — not via `export`
- [ ] Scheduler running — `tcp://<IP>:8786` confirmed
- [ ] Scheduler IP set in `config.py` on ALL machines
- [ ] Workers connected — `Workers: 2` visible on Scheduler
- [ ] `main.py` running on Scheduler — `RESULT` seen in logs

---

## 🎯 Final Demo Line

> *"We use a scheduler–worker architecture. Queries are distributed across workers, executed concurrently via Dask, and results are semantically reranked and aggregated on the scheduler."*

---

<div align="center">

**Distributed Video Search System &nbsp;•&nbsp; Demo Guide &nbsp;•&nbsp; Ubuntu Linux**

</div>
