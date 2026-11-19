import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { mkdtemp, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { createInterface } from 'node:readline';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { addEvent, calendarInput, eventInput } from '../lib/events.js';

const event = { title: 'Train to Albany', date: '2026-11-02', startTime: '05:45', endTime: '07:30',
    location: { name: 'Penn Station', lat: 40.75, lng: -73.99 } };
const state = { startDate: '2026-10-31', numDays: 2, startHour: 7, endHour: 22, viewMode: 'row', events: {} };

test('calendar dates, extended hours and retry identity', () => {
    const input = eventInput.parse({ ...event, transport: 'transit', eventId: randomUUID() });
    const added = addEvent(state, input);
    assert.equal(added.state.numDays, 3);
    assert.equal(added.state.startHour, 5);
    assert.equal(added.state.events[2][0].startMinutes, 345);
    assert.deepEqual(added.event.location, event.location);
    assert.equal(added.event.transport, 'transit');
    assert.equal(addEvent(added.state, input).status, 200);
    assert.equal(addEvent(added.state, { ...input, title: 'Another train' }).status, 409);
    assert.equal(addEvent(added.state, { ...event, date: '2027-01-01' }).status, 422);
    const midnight = addEvent(state, { ...event, date: '2027-01-01', endTime: '24:00' });
    assert.equal(midnight.state.startDate, '2027-01-01');
    assert.equal(midnight.state.endHour, 24);
});

test('invalid dates, times, colors and coordinates are rejected', () => {
    for (const fields of [{ title: ' ' }, { date: '2026-02-30' }, { startTime: '24:00' },
        { endTime: '10:60' }, { color: 'invalid' }, { transport: 'plane' }, { location: { ...event.location, lng: 190 } }]) {
        assert.equal(eventInput.safeParse({ ...event, ...fields }).success, false);
    }
    assert.equal(addEvent(state, { ...event, endTime: event.startTime }).status, 400);
    assert.equal(calendarInput.safeParse({ name: 'Trip', numDays: 32 }).success, false);
});

async function startApi(dir) {
    const child = spawn(process.execPath, ['server.js'], { env: { ...process.env, TZ: 'America/New_York',
        CALENDARS_DIR: dir, HOST: '127.0.0.1', PORT: '0', SERVE_STATIC: 'false' }, stdio: ['ignore', 'pipe', 'inherit'] });
    for await (const line of createInterface({ input: child.stdout })) {
        const match = line.match(/Server running at (http:\/\/127\.0\.0\.1:\d+)/);
        if (match) return { url: match[1], close: async () => {
            const closed = once(child, 'exit');
            child.kill();
            await closed;
        } };
    }
    throw new Error('Calendar API exited before it was ready');
}

test('an external MCP client and a stale browser save share durable events', async t => {
    const dir = await mkdtemp(join(tmpdir(), 'openmapcalendar-test-'));
    const legacyId = randomUUID();
    await writeFile(join(dir, legacyId + '.json'), JSON.stringify({ name: 'Existing trip',
        state: { ...state, startDate: '2026-11-01T02:30:00.000Z' } }));
    let api = await startApi(dir);
    t.after(async () => { await api.close(); await rm(dir, { recursive: true, force: true }); });
    const client = new Client({ name: 'calendar-test', version: '1' });
    await client.connect(new StdioClientTransport({ command: process.execPath, args: ['scripts/calendar-mcp.js'],
        env: { OPENMAPCALENDAR_URL: api.url } }));
    t.after(() => client.close());
    const call = (name, args = {}) => client.callTool({ name, arguments: args });
    const value = async (name, args) => {
        const result = await call(name, args);
        assert.ok(!result.isError, result.content[0].text);
        return JSON.parse(result.content[0].text);
    };
    const names = (await client.listTools()).tools.map(tool => tool.name).sort();
    assert.deepEqual(names, ['add_event', 'create_calendar', 'list_calendars', 'list_events']);
    const [legacy] = await value('list_calendars');
    assert.equal(legacy.startDate, '2026-10-31');
    const { id: calendarId } = await value('create_calendar', { name: 'Train trip', startDate: state.startDate });
    const path = '/api/calendars/' + calendarId;
    const browser = await (await fetch(api.url + path)).json();
    const input = { calendarId, ...event, eventId: randomUUID() };
    const first = await value('add_event', input);
    assert.equal(first.event.id, input.eventId);
    assert.equal((await value('add_event', input)).revision, first.revision);
    assert.equal((await call('add_event', { ...input, title: 'Other' })).isError, true);
    assert.equal((await call('add_event', { ...input, calendarId: 'missing' })).isError, true);
    const manual = { id: 'manual', dayIndex: 0, title: 'Pack', startMinutes: 600, endMinutes: 630, color: '#123456', description: '' };
    const save = await fetch(api.url + path, { method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ baseRevision: browser.revision, baseState: browser.state,
            state: { ...browser.state, events: { 0: [manual] } } }) });
    const merged = await save.json();
    assert.equal(merged.state.events[2][0].id, first.event.id);
    assert.equal(merged.state.startHour, 5);
    const schedule = await value('list_events', { calendarId });
    assert.deepEqual(schedule.events.map(e => [e.title, e.date]), [['Pack', state.startDate], [event.title, event.date]]);
    assert.deepEqual((await value('list_events', { calendarId: legacyId })).events, []);
    assert.equal((await fetch(api.url + '/api/ai/plan-events', { method: 'POST' })).status, 404);
    await api.close();
    api = await startApi(dir);
    assert.deepEqual(await (await fetch(api.url + path + '/events')).json(), schedule);
});
