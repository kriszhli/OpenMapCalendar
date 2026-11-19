import express from 'express';
import { openJournal } from './lib/journal-db.js';
import { journalRoutes } from './lib/journal-api.js';
import path from 'path';
import { fileURLToPath } from 'url';
import os from 'os';
import crypto from 'crypto';
import { addEvent, calendarInput, calendarWindow, describeEvent, eventInput, localDate } from './lib/events.js';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = Number(process.env.PORT ?? 3000);
const SHOULD_SERVE_STATIC = process.env.SERVE_STATIC !== 'false';
const CALENDARS_DIR = process.env.CALENDARS_DIR || path.join(__dirname, 'calendars');
const LEGACY_DATA_FILE = path.join(__dirname, 'calendar-data.json');
const DEFAULT_STATE = {
    numDays: 5,
    startDate: localDate(),
    startHour: 7,
    endHour: 22,
    viewMode: 'row',
    events: {},
};

// Browser writes and private media use the same origin as the app. CLI and MCP
// clients send no Origin. Vite forwards the original Host through its API proxy.
app.use('/api', (req, res, next) => {
    const origin = req.get('Origin');
    if (origin) {
        let host;
        try { host = new URL(origin).host; } catch { /* opaque origin */ }
        if (host !== req.get('Host')) return res.status(403).json({ error: 'Unrecognized browser origin' });
    }
    next();
});
app.use(express.json({ limit: '8mb' }));

if (SHOULD_SERVE_STATIC) {
    // Serve static files from the build directory
    app.use(express.static(path.join(__dirname, 'dist')));
}

const deepEqual = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const normalizeState = (raw) => {
    const settings = raw?.settings ?? {};
    const numDays = Number(raw?.numDays ?? settings.numDays);
    const startHour = Number(raw?.startHour ?? settings.startHour);
    const endHour = Number(raw?.endHour ?? settings.endHour);
    return {
        numDays: Number.isFinite(numDays) && numDays > 0 ? numDays : DEFAULT_STATE.numDays,
        startDate: typeof raw?.startDate === 'string' ? localDate(raw.startDate) : DEFAULT_STATE.startDate,
        startHour: Number.isFinite(startHour) ? startHour : DEFAULT_STATE.startHour,
        endHour: Number.isFinite(endHour) ? endHour : DEFAULT_STATE.endHour,
        viewMode: raw?.viewMode ?? settings.viewMode ?? DEFAULT_STATE.viewMode,
        events: typeof raw?.events === 'object' && raw.events ? raw.events : {},
    };
};

const flattenEvents = (eventsByDay) => {
    const result = new Map();
    for (const dayEvents of Object.values(eventsByDay || {})) {
        for (const ev of dayEvents || []) {
            if (!ev?.id) continue;
            result.set(ev.id, ev);
        }
    }
    return result;
};

const groupByDay = (eventsMap) => {
    const grouped = {};
    for (const event of eventsMap.values()) {
        const dayKey = String(event.dayIndex);
        if (!grouped[dayKey]) grouped[dayKey] = [];
        grouped[dayKey].push(event);
    }
    for (const dayEvents of Object.values(grouped)) {
        dayEvents.sort((a, b) => a.startMinutes - b.startMinutes);
    }
    return grouped;
};

const mergeState = (current, base, incoming) => {
    const merged = { ...current };

    for (const key of ['numDays', 'startDate', 'startHour', 'endHour', 'viewMode']) {
        if (!deepEqual(incoming[key], base[key])) {
            merged[key] = incoming[key];
        }
    }

    const baseMap = flattenEvents(base.events);
    const incomingMap = flattenEvents(incoming.events);
    const currentMap = flattenEvents(current.events);

    for (const [id, incomingEvent] of incomingMap.entries()) {
        const baseEvent = baseMap.get(id);
        if (deepEqual(incomingEvent, baseEvent)) continue;
        const currentEvent = currentMap.get(id);
        if (!baseEvent || !currentEvent) {
            currentMap.set(id, incomingEvent);
            continue;
        }
        // A timeline edit to the title must preserve a travel mode edited in the atlas.
        const next = { ...currentEvent };
        for (const key of new Set([...Object.keys(baseEvent), ...Object.keys(incomingEvent)])) {
            if (deepEqual(baseEvent[key], incomingEvent[key])) continue;
            if (Object.hasOwn(incomingEvent, key)) next[key] = incomingEvent[key];
            else delete next[key];
        }
        currentMap.set(id, next);
    }

    for (const id of baseMap.keys()) {
        if (!incomingMap.has(id)) {
            currentMap.delete(id);
        }
    }

    merged.events = groupByDay(currentMap);
    return merged;
};

const calendars = openJournal(CALENDARS_DIR, normalizeState, process.env.CALENDARS_DIR ? null : LEGACY_DATA_FILE);
journalRoutes(app, calendars);
const saveState = (cal, state) => {
    const next = { ...cal, state, revision: cal.revision + 1, updatedAt: new Date().toISOString() };
    calendars.save(next);
    return next;
};

// ─── API: List all calendars ───
app.get('/api/calendars', (req, res) => {
    const list = calendars.list().map(cal => ({
        id: cal.id, name: cal.name, updatedAt: cal.updatedAt, ...calendarWindow(cal.state),
    }));
    res.json(list);
});

// ─── API: Create a new calendar ───
app.post('/api/calendars', (req, res) => {
    try {
        const id = crypto.randomUUID();
        const input = calendarInput.safeParse({ name: 'Untitled', ...req.body });
        if (!input.success) return res.status(400).json({ error: input.error.message });
        const { name, startDate = localDate(), numDays = DEFAULT_STATE.numDays } = input.data;
        const calData = {
            id,
            name,
            state: { ...DEFAULT_STATE, startDate, numDays },
            revision: 0,
            updatedAt: new Date().toISOString(),
        };
        calendars.save(calData);
        res.json({ id, name: calData.name });
    } catch (err) {
        res.status(500).json({ error: 'Failed to create calendar' });
    }
});

// ─── API: Rename a calendar ───
app.put('/api/calendars/:id/rename', (req, res) => {
    const { id } = req.params;
    const cal = calendars.get(id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });

    const name = req.body?.name;
    if (!name || typeof name !== 'string') return res.status(400).json({ error: 'Name is required' });

    cal.name = name;
    cal.updatedAt = new Date().toISOString();
    calendars.save(cal);
    res.json({ success: true, id, name: cal.name });
});

// ─── API: Delete a calendar ───
app.delete('/api/calendars/:id', (req, res) => {
    const { id } = req.params;
    if (!calendars.get(id)) return res.status(404).json({ error: 'Calendar not found' });

    calendars.delete(id);
    res.json({ success: true });
});

// ─── API: Get a specific calendar ───
app.get('/api/calendars/:id', (req, res) => {
    const { id } = req.params;
    const cal = calendars.get(id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });

    res.json({
        state: cal.state,
        revision: cal.revision,
        name: cal.name,
        updatedAt: cal.updatedAt,
    });
});

// ─── API: Save a specific calendar ───
app.post('/api/calendars/:id', (req, res) => {
    const { id } = req.params;
    const cal = calendars.get(id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });

    try {
        const incoming = normalizeState(req.body?.state ?? req.body);
        const base = req.body?.baseState ? normalizeState(req.body.baseState) : null;
        const baseRevision = Number.isInteger(req.body?.baseRevision) ? req.body.baseRevision : null;

        const nextState =
            baseRevision !== null && baseRevision !== cal.revision && base
                ? mergeState(cal.state, base, incoming)
                : incoming;

        if (deepEqual(nextState, cal.state)) {
            return res.json({ success: true, state: cal.state, revision: cal.revision });
        }

        const saved = saveState(cal, nextState);
        res.json({ success: true, state: saved.state, revision: saved.revision });
    } catch (err) {
        res.status(500).json({ error: 'Failed to save data' });
    }
});

app.patch('/api/calendars/:id/events/:eventId/transport', (req, res) => {
    const cal = calendars.get(req.params.id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });
    const transport = req.body?.transport;
    if (transport !== null && !(typeof transport === 'string' && eventInput.shape.transport.safeParse(transport).success)) {
        return res.status(400).json({ error: 'Choose walking, transit, driving, bicycling or null' });
    }
    const block = Object.values(cal.state.events).flat().find(event => event.id === req.params.eventId);
    if (!block) return res.status(404).json({ error: 'Event not found' });
    if (transport === null) delete block.transport;
    else block.transport = transport;
    const saved = saveState(cal, cal.state);
    res.json({ event: describeEvent(saved.state, block), revision: saved.revision });
});

app.get('/api/calendars/:id/events', (req, res) => {
    const cal = calendars.get(req.params.id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });
    const events = Object.values(cal.state.events).flat()
        .sort((a, b) => a.dayIndex - b.dayIndex || a.startMinutes - b.startMinutes)
        .map(event => describeEvent(cal.state, event));
    res.json({ calendarId: cal.id, ...calendarWindow(cal.state), revision: cal.revision, events });
});

app.post('/api/calendars/:id/events', (req, res) => {
    const cal = calendars.get(req.params.id);
    if (!cal) return res.status(404).json({ error: 'Calendar not found' });
    const input = eventInput.safeParse(req.body);
    if (!input.success) return res.status(400).json({ error: input.error.message });
    const result = addEvent(cal.state, input.data);
    if (result.error) return res.status(result.status).json({ error: result.error });
    const saved = result.state ? saveState(cal, result.state) : cal;
    res.status(result.status).json({ calendarId: cal.id, revision: saved.revision, event: result.event });
});

// ─── Legacy endpoints (backwards compat) ───
app.get('/api/calendar', (req, res) => {
    // Return the first (most recent) calendar or default
    const list = calendars.list();
    if (list.length > 0) {
        const cal = calendars.get(list[0].id);
        return res.json({ state: cal.state, revision: cal.revision });
    }
    res.json({ state: DEFAULT_STATE, revision: 0 });
});

app.post('/api/calendar', (req, res) => {
    // Save to the first (most recent) calendar
    const list = calendars.list();
    if (list.length === 0) {
        return res.status(404).json({ error: 'No calendars exist' });
    }
    const cal = calendars.get(list[0].id);
    try {
        const incoming = normalizeState(req.body?.state ?? req.body);
        const saved = saveState(cal, incoming);
        res.json({ success: true, state: saved.state, revision: saved.revision });
    } catch (err) {
        res.status(500).json({ error: 'Failed to save data' });
    }
});

app.use('/api', (_req, res) => res.status(404).json({ error: 'Unknown API route' }));
app.use('/api', (err, _req, res, _next) => {
    res.status(err.status || 500).json({ error: err.status === 400 ? 'Invalid JSON request' : 'Calendar request failed' });
});

if (SHOULD_SERVE_STATIC) {
    // Fallback for SPA routing (must correspond to dist/index.html)
    app.use((req, res) => {
        res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
}

const server = app.listen(PORT, process.env.HOST || '127.0.0.1', (error) => {
    if (error) {
        console.error('Calendar API could not start:', error.message);
        process.exitCode = 1;
        return;
    }
    const { address, port } = server.address();
    console.log(`Server running at http://${address}:${port}`);

    if (address !== '0.0.0.0') return;
    // Print LAN IP addresses
    const nets = os.networkInterfaces();
    const results = {};

    for (const name of Object.keys(nets)) {
        for (const net of nets[name]) {
            // Skip over non-IPv4 and internal (i.e. 127.0.0.1) addresses
            if (net.family === 'IPv4' && !net.internal) {
                if (!results[name]) {
                    results[name] = [];
                }
                results[name].push(net.address);
            }
        }
    }

    console.log('Available on your LAN:');
    Object.keys(results).forEach((name) => {
        results[name].forEach((ip) => {
            console.log(`  http://${ip}:${port}`);
        });
    });
});
