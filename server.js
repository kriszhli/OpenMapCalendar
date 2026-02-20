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
const OLLAMA_URL = process.env.OLLAMA_URL || 'http://127.0.0.1:11434';
const OLLAMA_MODEL = process.env.OLLAMA_MODEL || 'gemma3:1b';
const OLLAMA_TIMEOUT_MS = Number(process.env.OLLAMA_TIMEOUT_MS ?? 45000);
const AI_MESSAGE_LIMIT = 14;
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

const cleanString = (value, maxLen = 500) =>
    typeof value === 'string' ? value.trim().slice(0, maxLen) : '';

const isIsoDate = (value) => /^\d{4}-\d{2}-\d{2}$/.test(value);
const is24hTime = (value) => /^([01]\d|2[0-3]):[0-5]\d$/.test(value);

const parseModelJson = (content) => {
    const text = cleanString(content, 10000);
    if (!text) return null;

    const strippedFence = text
        .replace(/^```json\s*/i, '')
        .replace(/^```\s*/i, '')
        .replace(/\s*```$/, '')
        .trim();

    for (const candidate of [text, strippedFence]) {
        try {
            return JSON.parse(candidate);
        } catch {
            // Try next parse strategy.
        }
    }

    const objectMatch = strippedFence.match(/\{[\s\S]*\}/);
    if (!objectMatch) return null;
    try {
        return JSON.parse(objectMatch[0]);
    } catch {
        return null;
    }
};

const sanitizeAiEvent = (raw) => {
    if (!raw || typeof raw !== 'object') return null;

    const title = cleanString(raw.title ?? raw.name, 120);
    const description = cleanString(raw.description ?? raw.notes, 800);
    const date = cleanString(raw.date, 10);
    const startTime = cleanString(raw.startTime ?? raw.start, 5);
    const endTime = cleanString(raw.endTime ?? raw.end, 5);
    const origin = cleanString(raw.origin ?? raw.location ?? raw.from, 160);
    const destination = cleanString(raw.destination ?? raw.to, 160);
    const color = cleanString(raw.color, 20);

    if (!title || !isIsoDate(date) || !is24hTime(startTime) || !is24hTime(endTime)) {
        return null;
    }

    return {
        title,
        description,
        date,
        startTime,
        endTime,
        origin,
        destination,
        color,
    };
};

const sanitizeContextEvent = (raw) => {
    if (!raw || typeof raw !== 'object') return null;

    const date = cleanString(raw.date, 10);
    const startTime = cleanString(raw.startTime, 5);
    const endTime = cleanString(raw.endTime, 5);
    const title = cleanString(raw.title, 120);
    const origin = cleanString(raw.origin, 140);
    const destination = cleanString(raw.destination, 140);
    const dayIndexRaw = Number(raw.dayIndex);

    if (!isIsoDate(date) || !is24hTime(startTime) || !is24hTime(endTime)) {
        return null;
    }

    return {
        dayIndex: Number.isFinite(dayIndexRaw) ? dayIndexRaw : 0,
        date,
        startTime,
        endTime,
        title: title || 'Untitled',
        origin,
        destination,
    };
};

const formatContextEventsForPrompt = (events) => {
    if (!events.length) return '[]';
    const lines = events.map((event) => {
        const routeBits = [event.origin, event.destination].filter(Boolean).join(' -> ');
        const route = routeBits ? ` | ${routeBits}` : '';
        return `${event.date} [day ${event.dayIndex}] ${event.startTime}-${event.endTime} ${event.title}${route}`;
    });
    return lines.join('\n');
};

const normalizeAiPlan = (raw) => {
    const source = raw && typeof raw === 'object' ? raw : {};
    const rawEvents = Array.isArray(source.events) ? source.events : [];
    const events = rawEvents.map(sanitizeAiEvent).filter(Boolean);

    let status =
        source.status === 'ready' || source.status === 'needs_clarification'
            ? source.status
            : events.length > 0
                ? 'ready'
                : 'needs_clarification';

    if (status === 'ready' && events.length === 0) {
        status = 'needs_clarification';
    }

    const fallbackMessage =
        status === 'ready'
            ? `I found ${events.length} event${events.length === 1 ? '' : 's'} to add.`
            : 'I need a bit more detail to schedule this accurately. Please clarify exact date/time or locations.';

    const assistantMessage = cleanString(
        source.assistantMessage ?? source.message ?? source.reply,
        1200
    ) || fallbackMessage;

    return { status, assistantMessage, events };
};

const buildAiSystemPrompt = (context) => `You are an assistant that converts travel/scheduling chat into calendar events.
Respond with STRICT JSON only, no markdown.

Current date: ${context.today}
User timezone: ${context.timezone}
Calendar visible window starts on: ${context.calendarStartDate}
Calendar visible window ends on: ${context.calendarEndDate}
Visible days in UI: ${context.visibleDays}
Typical hourly bounds: ${context.dayStartHour}:00-${context.dayEndHour}:00

Existing events (do not overlap with these unless user explicitly asks to replace):
${context.existingEventsText}

Output JSON schema:
{
  "status": "ready" | "needs_clarification",
  "assistantMessage": "short message for user",
  "events": [
    {
      "title": "string",
      "description": "string",
      "date": "YYYY-MM-DD",
      "startTime": "HH:MM",
      "endTime": "HH:MM",
      "origin": "free-text place",
      "destination": "free-text place",
      "color": "#RRGGBB optional"
    }
  ]
}

Rules:
1. If date/time or locations are ambiguous, set status to "needs_clarification", ask specific follow-up questions, and return an empty events array.
2. If enough detail exists, set status to "ready" and return all events.
3. Always use 24-hour HH:MM format.
4. Use exact dates in YYYY-MM-DD format; resolve relative dates from current date and timezone.
5. Do not overlap new events with existing events. If overlap is unavoidable, ask a clarification question.
6. Do not include extra keys.`;

// ─── Disk I/O Helpers ───

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

// ─── AI Planner endpoint (Ollama) ───
app.post('/api/ai/plan-events', async (req, res) => {
    const rawMessages = Array.isArray(req.body?.messages) ? req.body.messages : [];
    const contextRaw = req.body?.context && typeof req.body.context === 'object'
        ? req.body.context
        : {};

    const messages = rawMessages
        .slice(-AI_MESSAGE_LIMIT)
        .map((item) => {
            const role = item?.role === 'assistant' ? 'assistant' : 'user';
            const content = cleanString(item?.text, 2000);
            if (!content) return null;
            return { role, content };
        })
        .filter(Boolean);

    if (messages.length === 0) {
        return res.status(400).json({ error: 'At least one message is required.' });
    }

    const today = new Date().toISOString().slice(0, 10);
    const context = {
        today,
        timezone: cleanString(contextRaw.timezone, 80) || 'UTC',
        calendarStartDate: isIsoDate(cleanString(contextRaw.calendarStartDate, 10))
            ? cleanString(contextRaw.calendarStartDate, 10)
            : today,
        calendarEndDate: isIsoDate(cleanString(contextRaw.calendarEndDate, 10))
            ? cleanString(contextRaw.calendarEndDate, 10)
            : today,
        visibleDays: Number.isFinite(Number(contextRaw.visibleDays))
            ? Math.max(1, Math.min(90, Number(contextRaw.visibleDays)))
            : 5,
        dayStartHour: Number.isFinite(Number(contextRaw.dayStartHour))
            ? Math.max(0, Math.min(23, Number(contextRaw.dayStartHour)))
            : 7,
        dayEndHour: Number.isFinite(Number(contextRaw.dayEndHour))
            ? Math.max(1, Math.min(24, Number(contextRaw.dayEndHour)))
            : 22,
        existingEvents: Array.isArray(contextRaw.existingEvents)
            ? contextRaw.existingEvents.map(sanitizeContextEvent).filter(Boolean).slice(0, 200)
            : [],
    };
    context.existingEventsText = formatContextEventsForPrompt(context.existingEvents);

    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), OLLAMA_TIMEOUT_MS);

    try {
        const ollamaRes = await fetch(`${OLLAMA_URL}/api/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            signal: controller.signal,
            body: JSON.stringify({
                model: OLLAMA_MODEL,
                stream: false,
                format: 'json',
                options: { temperature: 0.15 },
                messages: [
                    { role: 'system', content: buildAiSystemPrompt(context) },
                    ...messages,
                ],
            }),
        });

        if (!ollamaRes.ok) {
            const detail = cleanString(await ollamaRes.text(), 250);
            return res.status(502).json({
                error: 'Ollama returned an error.',
                detail: detail || `${ollamaRes.status} ${ollamaRes.statusText}`,
            });
        }

        const payload = await ollamaRes.json();
        const modelContent = payload?.message?.content ?? '';
        const parsed = parseModelJson(modelContent);

        if (!parsed) {
            return res.json({
                status: 'needs_clarification',
                assistantMessage: 'I could not parse that reliably. Please restate with exact date, time, and locations.',
                events: [],
            });
        }

        res.json(normalizeAiPlan(parsed));
    } catch (err) {
        const isTimeout = err?.name === 'AbortError';
        res.status(503).json({
            error: isTimeout
                ? `Ollama timed out after ${OLLAMA_TIMEOUT_MS}ms.`
                : `Unable to reach Ollama at ${OLLAMA_URL}.`,
            detail: cleanString(err?.message, 250),
        });
    } finally {
        clearTimeout(timeoutId);
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
