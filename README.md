# Open Map Calendar

Plan time blocks and real-world movement in one place.

Open Map Calendar is a lightweight calendar UI paired with a live map. Add an origin/destination to events, see the route, and keep a shared calendar state synced via a simple server-backed store.

## Features

- Timeline-first scheduling: create events by dragging on the day grid, then resize start/end by dragging handles.
- Map-aware events: attach an origin and/or destination to any event and see pins on the map.
- Calendar-map linking: hover an event to highlight its pins and route on the map.
- Route modes per event: `simple` (curved), `precise` (OSRM geometry + cached), `hidden` (no route lines).
- Route chaining: if an event only has one endpoint, routes can connect it to the previous event in the day.
- Multiple planning views: `Day`, `Row`, `Grid`.
- Multiple calendars: create, rename, switch, and delete calendars.
- Lightweight collaboration: clients poll and save to a shared calendar store (good for small teams on a LAN).
- AI Planner (optional): chat in plain language and generate events (date/time + locations) via a local Ollama model.
- Bright/Dark theme toggle.

## Great For

- Travel days: meetings with real drive time between stops.
- Errands: chain a route through multiple places, then adjust times by drag/resize.
- Small teams: share one live plan on a LAN without setting up a database.

## UI Tour

### 1. First launch: Calendar Manager (create a calendar)

On a fresh install (no calendars yet), the app opens to **Calendar Manager** first. Create a calendar, then select it to start planning.

![Calendar manager](docs/screenshots/calendar-manager.png)

### 2. Planner workspace (after selecting a calendar)

After you select/create a calendar, you can schedule events and (optionally) add locations to see pins and routes on the map.

![Planner workspace](docs/screenshots/dashboard.png)

### 3. Pick your planning window (top controls)

- Use **Start Date**, **Days**, and **From/To** to define what you want to plan.
- Switch views with **Day / Row / Grid**.

![Theme toggle + top controls](docs/screenshots/theme-toggle.png)

### 4. Day view (focus on one day)

- Switch to **Day** when you want a single-day schedule plus the day list sidebar.
- Use the sidebar to jump between days.

![Day view](docs/screenshots/dayview.png)

![Day sidebar](docs/screenshots/day-sidebar.png)

### 5. Row view (week-style, multi-day timeline)

- Switch to **Row** to see multiple days as columns with a continuous time ruler.

![Row view](docs/screenshots/rowview.png)

### 6. Grid view (overview planning)

- Switch to **Grid** when you want a compact overview across many days.
- Useful for roughing in a trip before you fine-tune times.

![Grid view](docs/screenshots/gridview.png)

### 7. Create an event (drag to create, drag handles to resize)

1. Drag in a day column to create a block.
2. Drag the top/bottom handles to adjust start/end times.

![Event editing](docs/screenshots/event-edit.png)

### 8. Add locations (origin + destination)

- Add an origin and/or destination to an event to drop pins on the map.
- If you add both, you get a route for that event.

![Simple route](docs/screenshots/simple-route.png)

### 9. Choose route mode (simple / precise / hidden)

- `simple`: curved line (no routing API call).
- `precise`: OSRM driving geometry (cached onto the event).
- `hidden`: keep pins, hide lines.

![Route mode toggle](docs/screenshots/route-mode-toggle.png)

![Precise route](docs/screenshots/precise-route.png)

### 10. Re-open Calendar Manager (create/switch/rename/delete)

Use the calendar button in the top bar to open Calendar Manager again.

![Calendar manager](docs/screenshots/calendar-manager.png)

### 11. Delete a calendar

- Use the **Delete Calendar** button in the top bar (only shows when a calendar is selected).

![Delete calendar](docs/screenshots/delete-calendar.png)

### 12. AI Planner (optional)

- Open **AI Planner** and describe your day in plain language.
- If the model can infer exact dates/times/locations, it proposes events and adds them.
- Use **Rollback** to undo the last AI-generated batch.

![AI Planner](docs/screenshots/ai-planner.png)

## How It Works (Under The Hood)

- Events live inside a calendar "window": start date, number of visible days, and working-hour bounds.
- Geocoding uses the public Nominatim endpoint from the browser.
- The map uses Leaflet with CARTO basemaps.
- `precise` routing uses the public OSRM demo server (`router.project-osrm.org`) and caches geometry on the event.
- `simple` routing draws a curved line without calling a routing service.

## Quick Start (Local)

### Requirements

- Node.js 18+ (Node 20+ recommended)
- npm

### Run in dev mode (API + Vite)

```bash
npm install
npm run dev
```

- App: `http://localhost:5173`
- API: `http://localhost:3000` (Vite proxies `/api` to the API)

## Using AI Planner (Optional)

The AI Planner endpoint is implemented for local Ollama.

### Start Ollama

```bash
ollama serve
ollama pull gemma3:1b
```

### Configure (Optional)

- `OLLAMA_URL` (default `http://127.0.0.1:11434`)
- `OLLAMA_MODEL` (default `gemma3:1b`)
- `OLLAMA_TIMEOUT_MS` (default `45000`)

If Ollama is not running, the app still works; only the planner chat fails.

## Data, Sync, and Caveats

- Persistence: calendars are stored as JSON files in `calendars/`.
- Legacy migration: if `calendar-data.json` exists and `calendars/` is empty, the server migrates it into a new calendar file on boot.
- Collaboration model: this is a simple shared-state server with no authentication. Do not expose it to the public internet as-is.
- External services: public Nominatim + OSRM endpoints are great for personal/light use; for heavier or production usage, you should run your own geocoding/routing services or proxy them responsibly.

## Project Layout

- `src/`: React app (calendar UI + map)
- `server.js`: Express API + optional static hosting for `dist/`
- `calendars/`: persisted calendars (JSON)
- `scripts/dev.js`: runs `server.js` (API) + Vite (client)

## Scripts

- `npm run dev`: start API + Vite dev server
- `npm run dev:client`: start Vite only
- `npm run build`: type-check + production build (outputs `dist/`)
- `npm run start`: start the API server (serves `dist/` if built)
- `npm run host`: build + start production server
- `npm run lint`: run ESLint

## Deployment (Production)

This is a single Node process that can serve both the API and the built frontend.

### 1. Build and run

```bash
npm install
npm run host
```

- Server listens on `0.0.0.0:3000` (port is currently fixed in `server.js`).
- The app is served from the same origin as the API when `SERVE_STATIC` is not `false`.

### 2. Persist calendar data

Make sure the `calendars/` directory is writable and persisted (a volume if you deploy via containers).

### 3. Environment variables

- `SERVE_STATIC=false` to disable serving `dist/` (API-only mode)
- `OLLAMA_URL`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT_MS` for AI Planner
