import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { DatabaseSync } from 'node:sqlite';
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

test('photos, thumbnails and routes survive restart, independent of calendar polling', async t => {
    const dir = await mkdtemp(join(tmpdir(), 'openmapcalendar-photos-'));
    const block = { id:'pier',dayIndex:0,startMinutes:540,endMinutes:600,title:'Pier',color:'#447755',
        location:event.location, destination:{name:'Harbor',lat:40.76,lng:-73.9}, transport:'walking' };
    const file = join(dir,'coast.json');
    const snapshot = { name:'Coast',state:{...state,events:{0:[block]}},revision:12 };
    await writeFile(file,JSON.stringify(snapshot));
    let api = await startApi(dir);
    t.after(async()=>{await api.close();await rm(dir,{recursive:true,force:true});});
    const calendarUrl = api.url+'/api/calendars/coast';
    const stale = await (await fetch(calendarUrl)).json();
    const changeMode = await fetch(calendarUrl+'/events/pier/transport', {
        method:'PATCH',headers:{'Content-Type':'application/json'},body:JSON.stringify({transport:'transit'}) });
    assert.equal(changeMode.status,200);
    const draft = structuredClone(stale.state);
    draft.events[0][0].title = 'Harbor shuttle';
    const browserEdit = await fetch(calendarUrl,{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({baseState:stale.state,baseRevision:stale.revision,state:draft})});
    assert.equal(browserEdit.status,200);
    const photoPath = '/api/calendars/coast/events/pier/destination/photos';
    const { default:sharp } = await import('sharp');
    const image = await sharp({create:{width:800,height:600,channels:3,background:'#367d68'}}).jpeg().toBuffer();
    const send = data=>fetch(api.url+photoPath,{method:'POST',headers:{'Content-Type':'image/jpeg'},body:data});
    assert.equal((await send(Buffer.from('invalid'))).status,422);
    const upload = await send(image);
    assert.equal(upload.status,201);
    const {id} = await upload.json();
    const atlas = async()=> (await fetch(api.url+'/api/atlas')).json();
    const [trip] = await atlas();
    assert.equal(trip.events[0].transport,'transit');
    assert.equal(trip.events[0].title,'Harbor shuttle');
    assert.deepEqual(trip.events[0].photos.location,[]);
    const [photo] = trip.events[0].photos.destination;
    assert.equal(photo.id,id);
    const thumbnail = await fetch(api.url+photo.thumb);
    assert.equal(thumbnail.headers.get('content-type'),'image/webp');
    assert.equal((await sharp(Buffer.from(await thumbnail.arrayBuffer())).metadata()).width,360);
    assert.deepEqual(Buffer.from(await (await fetch(api.url+photo.url)).arrayBuffer()),image);
    await api.close();
    await writeFile(file,JSON.stringify({...snapshot,name:'Stale legacy file'}));
    api = await startApi(dir);
    assert.deepEqual(await atlas(),[trip]);
    const backupFile = join(dir,'export.sqlite');
    const exportResponse = await fetch(api.url+'/api/journal/backup');
    assert.equal(exportResponse.status,200);
    await writeFile(backupFile,Buffer.from(await exportResponse.arrayBuffer()));
    const exported = new DatabaseSync(backupFile,{ readOnly:true });
    assert.equal(exported.prepare('SELECT name FROM trips').get().name,'Coast');
    assert.deepEqual(Buffer.from(exported.prepare('SELECT original FROM photos').get().original),image);
    exported.close();
    assert.equal((await fetch(api.url+'/api/atlas',{headers:{Origin:'https://other.example'}})).status,403);
    const stateResult = await (await fetch(api.url+'/api/calendars/coast')).json();
    assert.equal(stateResult.revision,14);
    assert.equal(stateResult.state.events[0][0].photos,undefined);
    assert.equal((await fetch(api.url+'/api/calendars/coast',{method:'DELETE'})).status,200);
    assert.equal((await fetch(api.url+photo.thumb)).status,404);
    await api.close();
    api = await startApi(dir);
    assert.deepEqual(await atlas(),[]);
});
