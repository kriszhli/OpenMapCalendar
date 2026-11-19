# Open Map Calendar

A local-first trip calendar with an editable timeline and a live map. You can drag an event onto the timeline yourself or ask an external agent to schedule it. Both write to the same calendar.

![Calendar and map demo](docs/screenshots/DEMO.gif)

## Run locally

Use Node.js 22.12 or newer and npm.

```sh
npm ci
npm run dev
```

Open the app at [localhost:5173](http://localhost:5173). Vite proxies calendar requests to the Express API on port 3000. For a single server, run `npm run host` and open [localhost:3000](http://localhost:3000). Shell agents only need the API, started with `npm start`. By default the app supports access from your LAN. Set `HOST=127.0.0.1` to bind the app and API to loopback.

In Calendar Manager, create or select a calendar. Choose its start date, hours and Day, Row or Grid view. Drag to create a block, resize it to adjust the time, and add notes or map locations. Route lines can use a simple curve or a driving route, or you can hide them. The server stores each calendar in `calendars/` and an open calendar polls for changes once per second.

## Use HTTP directly

List calendars with `GET /api/calendars`. On a fresh install, create one with a name and an optional starting date and day count.

```sh
api=http://127.0.0.1:3000
curl -fsS "$api/api/calendars"
curl -fsS "$api/api/calendars" -H 'Content-Type: application/json' \
  -d '{"name":"November trip","startDate":"2026-11-19","numDays":5}'
```

Use the ID returned above in place of `CALENDAR_ID`. Generate a fresh UUID for each intended addition and reuse it for retries.

```sh
event_id=$(node -p "crypto.randomUUID()")
curl -fsS "$api/api/calendars/CALENDAR_ID/events" -H 'Content-Type: application/json' \
  -d "{\"eventId\":\"$event_id\",\"title\":\"Train to Albany\",\"date\":\"2026-11-20\",\"startTime\":\"09:00\",\"endTime\":\"11:00\"}"
curl -fsS "$api/api/calendars/CALENDAR_ID/events"
```

- Dates use `YYYY-MM-DD` and times use local 24-hour `HH:MM`. Each event fits within one day. Midnight at the end of the day can be written as `24:00`.
- Optional fields include `description`, a six-digit hex `color`, `location`, `destination`, and `routeMode` (`simple`, `precise`, or `hidden`). Both map positions take an object such as `{"name":"Penn Station","lat":40.75,"lng":-73.99}`.
- The timeline expands its hours and can extend forward to 31 days from its start. An empty calendar can start over on a more distant or earlier event date. A populated calendar returns HTTP 422 for a date outside that span. Existing events keep their dates. Overlapping events are allowed.
- A new event returns HTTP 201 with its calendar ID, revision and saved event. Repeating the same payload and UUID returns HTTP 200 and the same event. Reusing an existing UUID for different details returns HTTP 409. Invalid fields return HTTP 400 and an unknown calendar returns HTTP 404.

## Development

`npm test` runs event validation and retry tests. `npm run lint` checks the source. `npm run build` type-checks the frontend and creates the production bundle.
