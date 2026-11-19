import { test } from 'node:test';
import assert from 'node:assert/strict';
import { randomUUID } from 'node:crypto';
import { addEvent, calendarInput, eventInput } from '../lib/events.js';

const event = { title: 'Train to Albany', date: '2026-11-02', startTime: '05:45', endTime: '07:30',
    location: { name: 'Penn Station', lat: 40.75, lng: -73.99 } };
const state = { startDate: '2026-10-31', numDays: 2, startHour: 7, endHour: 22, viewMode: 'row', events: {} };

test('calendar dates, extended hours and retry identity', () => {
    const input = eventInput.parse({ ...event, eventId: randomUUID() });
    const added = addEvent(state, input);
    assert.equal(added.state.numDays, 3);
    assert.equal(added.state.startHour, 5);
    assert.equal(added.state.events[2][0].startMinutes, 345);
    assert.deepEqual(added.event.location, event.location);
    assert.equal(addEvent(added.state, input).status, 200);
    assert.equal(addEvent(added.state, { ...input, title: 'Another train' }).status, 409);
    assert.equal(addEvent(added.state, { ...event, date: '2027-01-01' }).status, 422);
    const midnight = addEvent(state, { ...event, date: '2027-01-01', endTime: '24:00' });
    assert.equal(midnight.state.startDate, '2027-01-01');
    assert.equal(midnight.state.endHour, 24);
});

test('invalid dates, times, colors and coordinates are rejected', () => {
    for (const fields of [{ title: ' ' }, { date: '2026-02-30' }, { startTime: '24:00' },
        { endTime: '10:60' }, { color: 'invalid' }, { location: { ...event.location, lng: 190 } }]) {
        assert.equal(eventInput.safeParse({ ...event, ...fields }).success, false);
    }
    assert.equal(addEvent(state, { ...event, endTime: event.startTime }).status, 400);
    assert.equal(calendarInput.safeParse({ name: 'Trip', numDays: 32 }).success, false);
});
