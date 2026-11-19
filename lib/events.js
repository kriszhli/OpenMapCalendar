import { randomUUID } from 'node:crypto';
import { isDeepStrictEqual } from 'node:util';
import { z } from 'zod';

const time = z.string().regex(/^([01]\d|2[0-3]):[0-5]\d$/, 'Use HH:MM in 24-hour time');
const place = z.strictObject({
    name: z.string().trim().min(1),
    lat: z.number().min(-90).max(90),
    lng: z.number().min(-180).max(180),
});

export const calendarInput = z.strictObject({
    name: z.string().trim().min(1),
    startDate: z.iso.date().optional(),
    numDays: z.number().int().min(1).max(31).optional(),
});

export const eventInput = z.strictObject({
    title: z.string().trim().min(1),
    date: z.iso.date().describe('Calendar day, YYYY-MM-DD'),
    startTime: time,
    endTime: z.union([time, z.literal('24:00')]),
    description: z.string().optional(),
    color: z.string().regex(/^#[0-9a-f]{6}$/i).optional(),
    location: place.optional(),
    destination: place.optional(),
    transport: z.enum(['walking', 'transit', 'driving', 'bicycling']).optional().describe('Travel mode for reaching this stop or its destination'),
    routeMode: z.enum(['simple', 'precise', 'hidden']).optional(),
    eventId: z.uuid().optional().describe('Reuse this UUID when retrying the same addition'),
});

export function localDate(value = new Date()) {
    if (typeof value === 'string' && z.iso.date().safeParse(value).success) return value;
    const date = new Date(value);
    if (!Number.isFinite(date.getTime())) return localDate();
    return [String(date.getFullYear()).padStart(4, '0'), String(date.getMonth() + 1).padStart(2, '0'), String(date.getDate()).padStart(2, '0')].join('-');
}

// UTC arithmetic here counts civil days across DST boundaries. Event times are wall-clock times.
const dayMs = 86_400_000;
const dateAt = (start, index) => new Date(Date.parse(start) + index * dayMs).toISOString().slice(0, 10);
const asMinutes = value => Number(value.slice(0, 2)) * 60 + Number(value.slice(3));
const asTime = minutes => [Math.floor(minutes / 60), minutes % 60].map(n => String(n).padStart(2, '0')).join(':');

export function calendarWindow(state) {
    return { startDate: state.startDate, endDate: dateAt(state.startDate, state.numDays - 1) };
}

export function describeEvent(state, block) {
    const { id, dayIndex, startMinutes, endMinutes, title, description, color, location, destination, transport, routeMode } = block;
    return {
        id,
        date: dateAt(state.startDate, dayIndex),
        startTime: asTime(startMinutes),
        endTime: asTime(endMinutes),
        title, description, color, location, destination, transport, routeMode,
    };
}

export function addEvent(state, input) {
    const { date, startTime, endTime, eventId, description = '', color = '#5B7FBF', ...fields } = input;
    const startMinutes = asMinutes(startTime);
    const endMinutes = asMinutes(endTime);
    if (endMinutes <= startMinutes) return { status: 400, error: 'endTime must be later than startTime on the same day' };

    let startDate = state.startDate;
    let dayIndex = (Date.parse(date) - Date.parse(startDate)) / dayMs;
    const blocks = Object.values(state.events).flat();
    if ((dayIndex < 0 || dayIndex >= 31) && blocks.length === 0) {
        startDate = date;
        dayIndex = 0;
    }
    const block = { id: eventId ?? randomUUID(), dayIndex, startMinutes, endMinutes, description, color, ...fields };
    const existing = eventId && blocks.find(item => item.id === eventId);
    if (existing) {
        const prior = describeEvent(state, existing);
        const requested = describeEvent({ startDate }, block);
        if (!isDeepStrictEqual(prior, requested)) {
            return { status: 409, error: 'This eventId already belongs to a different event' };
        }
        return { status: 200, event: prior };
    }
    if (dayIndex < 0 || dayIndex >= 31) {
        return { status: 422, error: `This calendar can accept dates from ${startDate} through ${dateAt(startDate, 30)}. Choose a calendar covering ${date}.` };
    }
    const numDays = Math.max(state.numDays, dayIndex + 1);
    const day = [...(state.events[dayIndex] || []), block].sort((a, b) => a.startMinutes - b.startMinutes);
    const next = {
        ...state,
        startDate,
        numDays,
        startHour: Math.min(state.startHour, Math.floor(startMinutes / 60)),
        endHour: Math.max(state.endHour, Math.ceil(endMinutes / 60)),
        events: { ...state.events, [dayIndex]: day },
    };
    return { status: 201, state: next, event: describeEvent(next, block) };
}
