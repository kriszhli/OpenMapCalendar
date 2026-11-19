# Open Map Calendar

A local-first trip calendar with an editable timeline and a live map.

![Calendar and map demo](docs/screenshots/DEMO.gif)

## Run locally

Use Node.js 22.12 or newer and npm.

```sh
npm ci
npm run dev
```

Open the app at [localhost:5173](http://localhost:5173). Vite proxies calendar requests to the Express API on port 3000. For a single server, run `npm run host` and open [localhost:3000](http://localhost:3000).

In Calendar Manager, create or select a calendar. Choose its start date, hours and Day, Row or Grid view. Drag to create a block, resize it to adjust the time, and add notes or map locations. Route lines can use a simple curve or a driving route, or you can hide them. The server stores each calendar in `calendars/` and an open calendar polls for changes once per second.
