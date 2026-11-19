# Working with the calendar

Use the running Open Map Calendar API for scheduling requests in this checkout. It owns the live state shared by the browser and other agents. Direct edits to files in `calendars/` bypass the running server.

1. Use the Open Map Calendar MCP tools if connected. Shell agents can use the equivalent HTTP calls below. The default API is `http://127.0.0.1:3000`.
2. List calendars and select the one the user named. If multiple calendars could be the intended target, ask which one. On a fresh checkout with no calendars, create a calendar for the request and set its start date to the event date.
3. Resolve relative dates in the user's local time. Check existing events when the request depends on availability. Submit a title, calendar date, start time and end time. Generate an eventId with `node -p "crypto.randomUUID()"` and reuse it if the request needs a retry.
4. Report the saved calendar, date and time from the successful response. Include a location if supplied. The returned event ID identifies the saved item.

| Operation | MCP tool | HTTP |
| --- | --- | --- |
| Find the calendar | `list_calendars` | `GET /api/calendars` |
| Create a calendar | `create_calendar` | `POST /api/calendars` with `name` and optional `startDate`, `numDays` |
| Read scheduled events | `list_events` | `GET /api/calendars/{id}/events` |
| Add an event | `add_event` | `POST /api/calendars/{id}/events` |

If the API is offline, run `npm ci` once in this checkout, then `npm start` (or `npm run dev` for frontend development). Use an existing running instance when available. See [README.md](README.md) for payload examples, MCP setup and calendar window behavior.
