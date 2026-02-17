# Open Map Calendar

Interactive calendar + map planner for multi-day scheduling, with shared file-backed persistence for LAN collaboration.

## What It Does

- Create, edit, and delete time blocks across days.
- Switch between `row`, `grid`, and `day` views.
- Add event locations/destinations and visualize routes on the map.
- Persist all calendar state to a shared JSON file (`calendar-data.json`):
  - start date
  - number of days
  - start/end hours
  - view mode
  - all events and route/location data
- Live sync between connected clients (same hosted instance).

## Tech Stack

- React + TypeScript
- Vite
- Express
- Leaflet
- Framer Motion

## Project Structure

- `/Users/kris/AI/Interactive_Calender/src` - React frontend
- `/Users/kris/AI/Interactive_Calender/server.js` - Express API + static hosting
- `/Users/kris/AI/Interactive_Calender/calendar-data.json` - persisted shared calendar state
- `/Users/kris/AI/Interactive_Calender/scripts/dev.js` - runs API server + Vite together in dev

## Local Development

1. Install dependencies:
```bash
npm install
```

2. Run frontend + API together:
```bash
npm run dev
```

3. Open:
```text
http://localhost:5173
```

Notes:
- Vite proxies `/api/*` to `http://localhost:3000`.
- `npm run dev` starts both processes so proxy errors are avoided.

## Production/LAN Hosting

Build and serve the app from Express:

```bash
npm run host
```

Server listens on `0.0.0.0:3000` and prints LAN URLs, for example:

```text
http://192.168.1.15:3000
```

Anyone opening that URL on your network edits the same shared calendar file.

## Scripts

- `npm run dev` - start API server + Vite dev server
- `npm run dev:client` - start Vite only
- `npm run start` - run Express server
- `npm run build` - TypeScript build + Vite build
- `npm run host` - build then run Express server
- `npm run lint` - run ESLint

## Data Model

Persisted state includes:

- `numDays`
- `startDate` (ISO string)
- `startHour`
- `endHour`
- `viewMode`
- `events` keyed by day index, including block metadata and optional route points (`location`, `destination`)

## GitHub

Remote:

- `origin`: [https://github.com/kriszhli/OpenMapCalender.git](https://github.com/kriszhli/OpenMapCalender.git)
