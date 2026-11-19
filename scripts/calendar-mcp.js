import { McpServer } from '@modelcontextprotocol/sdk/server/mcp.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import { z } from 'zod';
import { calendarInput, eventInput } from '../lib/events.js';

const base = process.env.OPENMAPCALENDAR_URL || 'http://127.0.0.1:3000';
const server = new McpServer({ name: 'open-map-calendar', version: '1.0.0' });
const calendarId = z.string().min(1).describe('Calendar ID returned by list_calendars or create_calendar');
const eventPath = id => `/api/calendars/${encodeURIComponent(id)}/events`;

async function request(path, body) {
    const response = await fetch(new URL(path, base), body === undefined ? {} : {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    }).catch(() => { throw new Error(`Cannot reach Open Map Calendar at ${base}. Start the API with npm start in its checkout.`); });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Calendar API returned HTTP ${response.status}`);
    return { content: [{ type: 'text', text: JSON.stringify(result) }] };
}

server.registerTool('list_calendars', {
    description: 'List local calendars with IDs, names and planning windows.',
    inputSchema: {}, annotations: { readOnlyHint: true },
}, () => request('/api/calendars'));

server.registerTool('create_calendar', {
    description: 'Create a named planning calendar. Supply startDate for a trip or future schedule.',
    inputSchema: calendarInput.shape,
}, args => request('/api/calendars', args));

server.registerTool('list_events', {
    description: 'Read scheduled events with calendar dates, local times and map locations.',
    inputSchema: { calendarId }, annotations: { readOnlyHint: true },
}, ({ calendarId }) => request(eventPath(calendarId)));

server.registerTool('add_event', {
    description: 'Add a timed event to the selected calendar. Supply a stable eventId for safe retries. Times use the local calendar day.',
    inputSchema: { calendarId, ...eventInput.shape },
}, ({ calendarId, ...event }) => request(eventPath(calendarId), event));

await server.connect(new StdioServerTransport());
