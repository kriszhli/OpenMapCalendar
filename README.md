# Open Map Calendar

Plan schedules and movement together: calendar timeline + live map + local AI planner.

![Demo GIF](/docs/screenshots/DEMO.gif)

## Why This Project

Most calendar apps ignore location context. This project combines:
- Time planning (day/row/grid calendar)
- Spatial planning (origin/destination + route visualization)
- AI scheduling (natural language -> structured events with safety checks)

## Quick Tutorial

### 1. Run locally

```bash
npm install
npm run dev
```

- App: `http://localhost:5173`
- API: `http://localhost:3000`

### 1b. Host on LAN (`npm run host`)

Use this when you want phones/laptops on the same Wi-Fi to open the same calendar server.

```bash
npm install
npm run host
```

Then:
- Find your machine LAN IP (example: `192.168.1.23`).
- Open `http://<LAN_IP>:3000` on other devices.
- Keep the terminal running while others use it.

### 2. Create or open a calendar

- On first launch, create a calendar in **Calendar Manager**.
- Use top controls to set **Start Date**, **Days**, and visible hour range.

### 3. Create events manually

- Drag in a day column to create a block.
- Edit title/notes.
- Add `origin` and `destination` to place pins on the map.

### 4. Choose route mode

- `simple`: curved line
- `precise`: OSRM geometry + cached route
- `hidden`: pins only

### 5. Use AI Planner

- Click the floating AI chat button.
- Ask naturally, e.g. “On the second day I want to visit the Statue of Liberty.”
- AI proposes/creates events and can ask clarifying questions.
- Use **Rollback** in chat to undo the last AI-generated batch.

## AI Engineering Highlights

### LLM backend design

The AI layer is engineered as a constrained planning system, not just free-form chat:

- Calendar-aware prompt context:
  - window start/end date
  - visible day count
  - working-hour bounds
  - timezone
  - existing events snapshot
  - relative-day mapping (e.g. `second day -> 2027-01-02`)
- Strict output contract: JSON schema with `status`, `assistantMessage`, and normalized `events[]`.
- Deterministic safety rules in prompt:
  - exact `YYYY-MM-DD` dates
  - 24h `HH:MM` times
  - no overlap with existing events unless explicitly requested

### Robustness features

- Input sanitization for all context fields.
- Output normalization + validation before applying events.
- Automatic retry pass when model asks for exact dates that are already resolved by relative-day mappings.
- Graceful fallback to clarification mode when model output is malformed.

### Location disambiguation

- Geocoding uses multiple candidates.
- Candidate selection is biased to the trip’s existing location centroid to reduce wrong-country results.

### User safety / control

- AI state is isolated per calendar.
- Overlap checks run before insertion.
- One-click rollback for last AI-generated batch.

## Tech Stack

- Frontend: React + TypeScript + Vite
- Backend: Express
- Map: Leaflet
- Routing: OSRM
- Geocoding: Nominatim
- Local LLM: Ollama (`gemma3:1b` by default)

## Optional AI Setup

```bash
ollama serve
ollama pull gemma3:1b
```

Optional env vars:
- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default `gemma3:1b`)
- `OLLAMA_TIMEOUT_MS` (default `45000`)

## Scripts

- `npm run dev` - API + Vite
- `npm run build` - type-check + production build
- `npm run start` - run server
- `npm run host` - build + host
- `npm run lint` - lint checks
