# Open Map Calendar

![Routes overview](/screenshot_of_routes.png)

Plan your schedule and your movement in one view.

Open Map Calendar is built for people who care about *where* things happen, not just *when*. Instead of juggling a calendar app and a map app, you can plan both together in a single workflow.

## Why People Like It

- You can see your timeline and map side by side.
- Every event can include a start point and destination.
- Routes are visual, fast to switch, and easy to hide when you want a cleaner map.
- Labels stay visible as `time | Title`, so the map is readable at a glance.
- Shared calendars keep everyone on the same page during trips, events, or team schedules.

## What It Improves In Real Life

- Less context switching between apps.
- Better day planning when movement time matters.
- Faster "what's next" decisions because map + schedule are already connected.
- Easier collaboration for families, travel groups, and teams.

## Quick Product Tour

### 1. Create or open a calendar
Use **Calendars** in the top controls to start a new plan or continue an existing one.

### 2. Add events by dragging time
Drag inside a day column to create a block, then add title, notes, and locations.

### 3. Add locations
- `📍` Start location
- `🏁` Destination

If an event has one location only, it can still connect to your previous event endpoint on the same day.

### 4. Control route style with one button
The same button cycles through:
- `simple` -> `precise` -> `hidden` -> `simple`

- `simple`: lightweight curved route
- `precise`: provider route with cached geometry
- `hidden`: keeps pins, hides route line

### 5. Focus mode for a single day
Switch to **Day view** for tighter planning and fine zoom interaction on the map.

### 6. AI planner (new)
Use the floating chat button in the bottom-right corner to describe your plan in plain language:
- where you are going
- when you are going
- where you are going from / to

The assistant asks follow-up clarification questions when dates or places are ambiguous.  
Once details are clear, it creates events directly in your calendar and geocodes locations for map pins/routes.

## Screenshots

![Calendar and map layout](/screenshot_calendar_and_map.png)

![Day view planning](/screenshot_day_view.png)

![Route modes button](/screenshot_route_toggle.png)

## Collaboration and Data

- Calendar changes are persisted on the server.
- Multiple clients connected to the same hosted app stay synced.
- Precise routes are cached in event data, so unchanged routes are reused.
- Cached route geometry is removed only when it is no longer valid or needed.

## Getting Started

### Use it on your network (recommended)
```bash
npm install
npm run host
```

Open the printed URL in your browser (and on other devices in the same network if needed).

### Local AI setup (Ollama)
Run Ollama locally and pull the model you want (default in app is `gemma3:1b`):

```bash
ollama serve
ollama pull gemma3:1b
```

Optional environment variables before starting the app server:
- `OLLAMA_MODEL` (default: `gemma3:1b`)
- `OLLAMA_URL` (default: `http://127.0.0.1:11434`)
- `OLLAMA_TIMEOUT_MS` (default: `45000`)

### Develop locally
```bash
npm install
npm run dev
```

- App: `http://localhost:5173`
- API: `http://localhost:3000`

## Engineering Notes

### Stack
- React + TypeScript
- Vite
- Express
- Leaflet
- Framer Motion

### Main folders/files
- `src/` - frontend app
- `server.js` - API + hosting server
- `calendars/` - persisted calendar data
- `scripts/dev.js` - runs API + Vite in development

### Scripts
- `npm run dev` - API + Vite dev server
- `npm run dev:client` - Vite only
- `npm run start` - server only
- `npm run build` - production build
- `npm run host` - build + host
- `npm run lint` - lint checks

### Data model (high-level)
Stored calendar state includes:
- date range and view settings
- events by day
- event metadata (title, notes, color, locations)
- route mode and cached precise route geometry

## GitHub

Repository:
- [OpenMapCalender](https://github.com/kriszhli/OpenMapCalender)
