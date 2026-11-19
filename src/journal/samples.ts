import type { Trip, Visit } from './model';

// Read-only examples are rendered only while the personal database is empty.
function sample(
  id: string,
  name: string,
  date: string,
  first: [string, number, number],
  last: [string, number, number],
  mode: Visit['transport'],
): Trip {
  const photo = { id, thumb: '/samples/' + id + '.jpg', url: '/samples/' + id + '.jpg' };
  const events = [first, last].map(([place, lat, lng], i): Visit => ({
    id: id + i,
    dayIndex: 0,
    title: String(place),
    description: '',
    color: '#37755d',
    startMinutes: i ? 660 : 540,
    endMinutes: i ? 720 : 600,
    location: { name: String(place), lat: Number(lat), lng: Number(lng) },
    transport: mode,
    photos: { location: i === 0 ? [photo] : [] },
  }));
  return { id, name, startDate: date, endDate: date, events };
}
export const samples = [
  sample(
    'kyoto',
    'Kyoto, Japan',
    '2025-10-14',
    ['Higashiyama', 34.998, 135.78],
    ['Yasaka Shrine', 35.003, 135.778],
    'walking',
  ),
  sample(
    'liguria',
    'Cinque Terre, Italy',
    '2025-06-08',
    ['Manarola', 44.107, 9.728],
    ['Vernazza', 44.135, 9.684],
    'transit',
  ),
  sample(
    'san-francisco',
    'San Francisco, USA',
    '2025-03-21',
    ['Golden Gate Bridge', 37.81, -122.477],
    ['Ferry Building', 37.795, -122.393],
    'bicycling',
  ),
];
