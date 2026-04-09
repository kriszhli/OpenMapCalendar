import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import os from 'os';
import crypto from 'crypto';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
const SHOULD_SERVE_STATIC = process.env.SERVE_STATIC !== 'false';
const CALENDARS_DIR = path.join(__dirname, 'calendars');
const LEGACY_DATA_FILE = path.join(__dirname, 'calendar-data.json');
const PLANNER_SERVICE_URL = process.env.PLANNER_SERVICE_URL || 'http://127.0.0.1:8001';
const DEFAULT_STATE = {
    numDays: 5,
    startDate: new Date().toISOString(),
    startHour: 7,
    endHour: 22,
    viewMode: 'row',
    events: {},
};

// In-memory cache: Map<id, { name, state, revision, updatedAt }>
const calendars = new Map();

app.use(cors());
app.use(express.json());

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
        startDate: typeof raw?.startDate === 'string' ? raw.startDate : DEFAULT_STATE.startDate,
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
        if (!deepEqual(incomingEvent, baseEvent)) {
            currentMap.set(id, incomingEvent);
        }
    }

    for (const id of baseMap.keys()) {
        if (!incomingMap.has(id)) {
            currentMap.delete(id);
        }
    }

    merged.events = groupByDay(currentMap);
    return merged;
};

const generateId = () => crypto.randomUUID();

const ensureCalendarsDir = () => {
    if (!fs.existsSync(CALENDARS_DIR)) {
        fs.mkdirSync(CALENDARS_DIR, { recursive: true });
    }
};

const calendarFilePath = (id) => path.join(CALENDARS_DIR, `${id}.json`);

const readCalendarFromDisk = (id) => {
    const filePath = calendarFilePath(id);
    if (!fs.existsSync(filePath)) return null;
    try {
        const raw = JSON.parse(fs.readFileSync(filePath, 'utf8'));
        return {
            id,
            name: raw.name || 'Untitled',
            state: normalizeState(raw.state || raw),
            updatedAt: raw.updatedAt || new Date().toISOString(),
        };
    } catch {
        return null;
    }
};

const writeCalendarToDisk = (id, data) => {
    ensureCalendarsDir();
    fs.writeFileSync(calendarFilePath(id), JSON.stringify({
        name: data.name,
        state: data.state,
        updatedAt: data.updatedAt,
    }, null, 2));
};

const deleteCalendarFromDisk = (id) => {
    const filePath = calendarFilePath(id);
    if (fs.existsSync(filePath)) fs.unlinkSync(filePath);
};

// ─── Initialize: load calendars from disk, migrate legacy data ───

const loadAllCalendars = () => {
    ensureCalendarsDir();

    // Migrate legacy calendar-data.json if it exists and calendars dir is empty
    const existingFiles = fs.readdirSync(CALENDARS_DIR).filter(f => f.endsWith('.json'));

    if (existingFiles.length === 0 && fs.existsSync(LEGACY_DATA_FILE)) {
        try {
            const legacyRaw = JSON.parse(fs.readFileSync(LEGACY_DATA_FILE, 'utf8'));
            const id = generateId();
            const calData = {
                name: 'My Calendar',
                state: normalizeState(legacyRaw),
                updatedAt: new Date().toISOString(),
            };
            writeCalendarToDisk(id, calData);
            calendars.set(id, { ...calData, id, revision: 0 });
            console.log(`Migrated legacy calendar-data.json → calendars/${id}.json`);
        } catch (err) {
            console.error('Failed to migrate legacy data:', err);
        }
    }

    // Load all calendar files
    const files = fs.readdirSync(CALENDARS_DIR).filter(f => f.endsWith('.json'));
    for (const file of files) {
        const id = file.replace('.json', '');
        if (!calendars.has(id)) {
            const cal = readCalendarFromDisk(id);
            if (cal) {
                calendars.set(id, { ...cal, revision: 0 });
            }
        }
    }
};

loadAllCalendars();

// ─── API: List all calendars ───
app.get('/api/calendars', (req, res) => {
    const list = [];
    for (const [id, cal] of calendars.entries()) {
        list.push({ id, name: cal.name, updatedAt: cal.updatedAt });
    }
    list.sort((a, b) => new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime());
    res.json(list);
});

// ─── API: Create a new calendar ───
app.post('/api/calendars', (req, res) => {
    try {
        const id = generateId();
        const name = req.body?.name || 'Untitled';
        const calData = {
            id,
            name,
            state: { ...DEFAULT_STATE, startDate: new Date().toISOString() },
            revision: 0,
            updatedAt: new Date().toISOString(),
        };
        writeCalendarToDisk(id, calData);
        calendars.set(id, calData);
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
    writeCalendarToDisk(id, cal);
    res.json({ success: true, id, name: cal.name });
});

// ─── API: Delete a calendar ───
app.delete('/api/calendars/:id', (req, res) => {
    const { id } = req.params;
    if (!calendars.has(id)) return res.status(404).json({ error: 'Calendar not found' });

    calendars.delete(id);
    deleteCalendarFromDisk(id);
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

        cal.state = nextState;
        cal.revision += 1;
        cal.updatedAt = new Date().toISOString();
        writeCalendarToDisk(id, cal);
        res.json({ success: true, state: cal.state, revision: cal.revision });
    } catch (err) {
        res.status(500).json({ error: 'Failed to save data' });
    }
});

app.post('/api/ai/plan-events', async (req, res) => {
    try {
        const plannerRes = await fetch(`${PLANNER_SERVICE_URL}/api/ai/plan-events`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req.body ?? {}),
        });

        const text = await plannerRes.text();
        res.status(plannerRes.status);
        res.setHeader('Content-Type', plannerRes.headers.get('content-type') || 'application/json');
        res.send(text);
    } catch (err) {
        res.status(503).json({
            error: `Unable to reach planner service at ${PLANNER_SERVICE_URL}.`,
            detail: err?.message || 'Planner proxy request failed',
        });
    }
});

app.post('/api/ai/confirm-plan', async (req, res) => {
    try {
        const plannerRes = await fetch(`${PLANNER_SERVICE_URL}/api/ai/confirm-plan`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(req.body ?? {}),
        });

        const text = await plannerRes.text();
        res.status(plannerRes.status);
        res.setHeader('Content-Type', plannerRes.headers.get('content-type') || 'application/json');
        res.send(text);
    } catch (err) {
        res.status(503).json({
            error: `Unable to reach planner service at ${PLANNER_SERVICE_URL}.`,
            detail: err?.message || 'Planner confirmation proxy request failed',
        });
    }
});

// ─── Legacy endpoints (backwards compat) ───
app.get('/api/calendar', (req, res) => {
    // Return the first (most recent) calendar or default
    const list = [...calendars.values()].sort((a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    if (list.length > 0) {
        const cal = list[0];
        return res.json({ state: cal.state, revision: cal.revision });
    }
    res.json({ state: DEFAULT_STATE, revision: 0 });
});

app.post('/api/calendar', (req, res) => {
    // Save to the first (most recent) calendar
    const list = [...calendars.values()].sort((a, b) =>
        new Date(b.updatedAt).getTime() - new Date(a.updatedAt).getTime()
    );
    if (list.length === 0) {
        return res.status(404).json({ error: 'No calendars exist' });
    }
    const cal = list[0];
    try {
        const incoming = normalizeState(req.body?.state ?? req.body);
        cal.state = incoming;
        cal.revision += 1;
        cal.updatedAt = new Date().toISOString();
        writeCalendarToDisk(cal.id, cal);
        res.json({ success: true, state: cal.state, revision: cal.revision });
    } catch (err) {
        res.status(500).json({ error: 'Failed to save data' });
    }
});

if (SHOULD_SERVE_STATIC) {
    // Fallback for SPA routing (must correspond to dist/index.html)
    app.use((req, res) => {
        res.sendFile(path.join(__dirname, 'dist', 'index.html'));
    });
}

app.listen(PORT, '0.0.0.0', () => {
    console.log(`Server running at http://0.0.0.0:${PORT}`);

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
            console.log(`  http://${ip}:${PORT}`);
        });
    });
});
