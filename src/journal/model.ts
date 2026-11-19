import type { LocationData, TimeBlock } from '../types';
import { minutesToTime } from '../types';

type Endpoint = 'location' | 'destination';
export interface Photo {
  id: string;
  thumb: string;
  url: string;
}
export interface Visit extends TimeBlock {
  photoOnlyEndpoints?: Endpoint[];
  photos: Partial<Record<Endpoint, Photo[]>>;
}
export interface Trip {
  id: string;
  name: string;
  startDate: string;
  endDate: string;
  events: Visit[];
}
export interface Stop {
  key: string;
  trip: Trip;
  event: Visit;
  endpoint: Endpoint;
  place: LocationData;
  photos: Photo[];
  time: string;
}
interface Leg {
  from: Stop;
  to: Stop;
  depart: string;
  arrive: string;
  mode?: TimeBlock['transport'];
  minutes: number;
  geometry: [number, number][];
}

export function stops(trip: Trip): Stop[] {
  return trip.events.flatMap((event) =>
    (['location', 'destination'] as const).flatMap((endpoint) => {
      const place = event[endpoint];
      return place
        ? [
            {
              key: JSON.stringify([trip.id, event.id, endpoint]),
              trip,
              event,
              endpoint,
              place,
              photos: event.photos[endpoint] || [],
              time: minutesToTime(endpoint === 'location' ? event.startMinutes : event.endMinutes),
            },
          ]
        : [];
    }),
  );
}
export function connections(points: Stop[]): Leg[] {
  const result: Leg[] = [];
  for (let i = 1; i < points.length; i++) {
    const from = points[i - 1],
      to = points[i];
    const same = from.event.id === to.event.id;
    const cache = to.event.preciseRouteCache;
    const matches = (a: LocationData, b: LocationData) =>
      Math.abs(a.lat - b.lat) < 0.000001 && Math.abs(a.lng - b.lng) < 0.000001;
    const useCache =
      cache &&
      to.event.routeMode === 'precise' &&
      (!to.event.transport || to.event.transport === 'driving') &&
      matches(cache.from, from.place) &&
      matches(cache.to, to.place);
    const depart = same ? from.event.startMinutes : from.event.endMinutes;
    const arrive = to.endpoint === 'destination' ? to.event.endMinutes : to.event.startMinutes;
    result.push({
      from,
      to,
      depart: minutesToTime(depart),
      arrive: minutesToTime(arrive),
      mode: to.event.transport || (useCache ? 'driving' : undefined),
      minutes: (to.event.dayIndex - from.event.dayIndex) * 1440 + arrive - depart,
      geometry: useCache
        ? cache.coords
        : [
            [from.place.lat, from.place.lng],
            [to.place.lat, to.place.lng],
          ],
    });
  }
  return result;
}
export function dateLabel(iso: string, options: Intl.DateTimeFormatOptions) {
  return new Date(iso + 'T12:00:00').toLocaleDateString('en-US', options);
}
export function dayLabel(trip: Trip, index: number, format: Intl.DateTimeFormatOptions = {
  weekday: 'short', month: 'short', day: 'numeric',
}) {
  const date = new Date(trip.startDate + 'T12:00:00');
  date.setDate(date.getDate() + index);
  return date.toLocaleDateString('en-US', format);
}
export function directions(leg: Leg) {
  const params = new URLSearchParams({
    api: '1',
    origin: `${leg.from.place.lat},${leg.from.place.lng}`,
    destination: `${leg.to.place.lat},${leg.to.place.lng}`,
  });
  if (leg.mode) params.set('travelmode', leg.mode);
  return 'https://www.google.com/maps/dir/?' + params;
}
