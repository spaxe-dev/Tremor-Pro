## TremorSense AI – How to Run Everything

This guide shows the **exact steps** to run the Tremor dashboard and backend on your local machine.
No external AI API key is needed — the system uses **rule-based biomarker interpretation** to generate clinical reports directly from your sensor data.

---

## 1. What you have

- **Frontend** (Vite + Tailwind + GSAP)
  - Entry: `index.html`, `dashboard.js`
  - Profiles screen: `profiles.html`, `profiles.js`
- **Backend** (FastAPI + Uvicorn)
  - Entry: `backend/main.py`
  - Local SQL database: `profiles.db` (SQLite, no server needed)
- **Device**
  - ESP32 tremor monitor streaming SSE to the dashboard.

All data stays **local**, including the SQLite database.

---

## 2. Prerequisites

- **Node.js + npm** installed
- **Python 3.13** installed (you already have this)
- PowerShell as your terminal (what you’re using now)

Recommended directory:

```text
D:\hackathon\Tremor-Pro\ai_dashboard\
```

---

## 3. Backend setup (FastAPI + SQLite)

From PowerShell:

```powershell
cd D:\hackathon\Tremor-Pro\ai_dashboard\backend

# (optional but recommended) create and activate venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install backend dependencies
pip install -r requirements.txt
```

Then **start the backend**:

```powershell
cd D:\hackathon\Tremor-Pro\ai_dashboard\backend
python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

- It will print something like:
  - `Uvicorn running on http://0.0.0.0:8000`
- On your machine, open: `http://localhost:8000`

Quick checks:

- Health: `http://localhost:8000/health`
- Empty sessions list (at first): `http://localhost:8000/profile/sessions`

The backend will automatically create a **local SQLite file**:

- `D:\hackathon\Tremor-Pro\ai_dashboard\backend\profiles.db`

This stores your **previously analyzed sessions** and profile info.

---

## 4. Frontend setup (Vite dashboard)

In a **second PowerShell window**:

```powershell
cd D:\hackathon\Tremor-Pro\ai_dashboard

# install front-end deps once
npm install

# run the dashboard dev server
npm run dev
```

Vite will print a URL, typically:

- `http://localhost:5173/`

Open that in your browser:

- **Main clinical dashboard**: `http://localhost:5173/`
- **Profile / trend screen**: `http://localhost:5173/profiles.html`

The frontend is configured to talk to the backend at:

- `http://localhost:8000`

If you want to override this, create a `.env` file in `ai_dashboard`:

```env
VITE_BACKEND_URL=http://localhost:8000
```

---

## 5. Running a tremor session (end‑to‑end)

1. **Start backend** (section 3) – keep it running.
2. **Start frontend** with `npm run dev` (section 4).
3. In the browser (`http://localhost:5173/`):
   - Enter your **ESP32 IP** in the left sidebar (`ESP32 IP` field).
   - Click **Connect**.
   - When connected:
     - `Status` becomes **Connected**.
     - **Start Recording** is enabled.
4. Click **Start Recording**:
   - The dashboard accumulates windows from the SSE `bands` stream.
5. Click **Stop Session**:
   - This finalizes the in-session statistics.
6. Click **Generate Report**:
   - The frontend builds a `SessionSummary` from your biomarkers.
   - Sends it to `POST /profile/session/direct` on the backend.
   - The backend generates a **rule-based clinical report** from the raw biomarkers (no MedGemma / external AI).
   - Stores the session + report into **`profiles.db`**.
   - Shows the clinical report on the right.

Every time you generate a report, that session is saved and appears in the profile view.

---

## 6. Using the profile / trend screen

Once you’ve run at least one analyzed session:

1. With `npm run dev` still running, open:

   ```text
   http://localhost:5173/profiles.html
   ```

2. The backend endpoint `GET /profile/sessions` is called automatically.
3. You’ll see:
   - **Sessions** count
   - **Mean score (all)** across sessions
   - **Trend** label (Improving / Stable / Worsening) based on weekly slope
   - **Severity trend chart**: mean tremor score per session over time
   - **Dominant band over time**: 4–6 Hz / 6–8 Hz / 8–12 Hz
   - **Previous sessions list**:
     - Date / time
     - `session_id`
     - Mean score
     - Dominant band
     - Confidence (High / Moderate / Low)

All values are read from the **local SQLite DB** (`profiles.db`) via:

- `GET /profile/sessions`
- (Optionally) `GET /profile/sessions/{session_id}` for detail views.

---

## 7. Stopping and restarting

- To stop the backend: press **Ctrl + C** in the backend terminal.
- To stop Vite: press **Ctrl + C** in the frontend terminal.
- To restart, just re-run:
  - Backend: `python -m uvicorn main:app --host 0.0.0.0 --port 8000 --reload`
  - Frontend: `npm run dev`

The **profile history is preserved** because it lives in `profiles.db`, not in memory.

---

## 8. Summary of endpoints and ports

- **Backend (FastAPI)**
  - Base: `http://localhost:8000`
  - `GET /health` – health check
  - `POST /analyze` – MedGemma tremor analysis (legacy, still available)
  - `POST /profile/session/direct` – **analyse + store** session (rule-based, no AI key needed)
  - `POST /tests/analyze-phase/direct` – rule-based analysis for a standardized test phase
  - `POST /profile/session` – store one pre-analysed session (legacy)
  - `GET /profile/sessions` – list all stored sessions
  - `GET /profile/sessions/{session_id}` – full details for one session

- **Frontend (Vite)**
  - Base: `http://localhost:5173`
  - `/` – live dashboard
  - `/profiles.html` – profile + trends page

If these steps are followed in order (backend first, then frontend, then device connect), the whole system will work end‑to‑end on your machine. 

