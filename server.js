import express from 'express';
import cors from 'cors';
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import os from 'os';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = 3000;
const DATA_FILE = path.join(__dirname, 'calendar-data.json');
const DEFAULT_STATE = {
    numDays: 5,
    startDate: new Date().toISOString(),
    startHour: 7,
    endHour: 22,
    viewMode: 'row',
    events: {},
};

let currentState = { ...DEFAULT_STATE };
let revision = 0;
let updatedAt = new Date().toISOString();

app.use(cors());
app.use(express.json());

// Serve static files from the build directory
app.use(express.static(path.join(__dirname, 'dist')));

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

const readDiskState = () => {
    if (!fs.existsSync(DATA_FILE)) {
        fs.writeFileSync(DATA_FILE, JSON.stringify(DEFAULT_STATE, null, 2));
    }

    const raw = fs.readFileSync(DATA_FILE, 'utf8');
    currentState = normalizeState(JSON.parse(raw));
};

const writeDiskState = () => {
    fs.writeFileSync(DATA_FILE, JSON.stringify(currentState, null, 2));
};

const responsePayload = () => ({
    state: currentState,
    revision,
    updatedAt,
});

// API Endpoints
app.get('/api/calendar', (req, res) => {
    try {
        res.json(responsePayload());
    } catch (err) {
        res.status(500).json({ error: 'Failed to read data' });
    }
});

app.post('/api/calendar', (req, res) => {
    try {
        const incoming = normalizeState(req.body?.state ?? req.body);
        const base = req.body?.baseState ? normalizeState(req.body.baseState) : null;
        const baseRevision = Number.isInteger(req.body?.baseRevision) ? req.body.baseRevision : null;

        const nextState =
            baseRevision !== null && baseRevision !== revision && base
                ? mergeState(currentState, base, incoming)
                : incoming;

        if (deepEqual(nextState, currentState)) {
            return res.json({ success: true, ...responsePayload() });
        }

        currentState = nextState;
        revision += 1;
        updatedAt = new Date().toISOString();
        writeDiskState();
        res.json({ success: true, ...responsePayload() });
    } catch (err) {
        res.status(500).json({ error: 'Failed to save data' });
    }
});

// Fallback for SPA routing (must correspond to dist/index.html)
app.use((req, res) => {
    res.sendFile(path.join(__dirname, 'dist', 'index.html'));
});

readDiskState();

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
