import express from 'express';
import sharp from 'sharp';
import { backup } from 'node:sqlite';
import { mkdtemp, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { randomUUID } from 'node:crypto';
import { calendarWindow } from './events.js';

const mimeTypes = { jpeg: 'image/jpeg', png: 'image/png', webp: 'image/webp' };

export function journalRoutes(app, store) {
  const { db } = store;
  app.get('/api/journal/backup', async (_req, res) => {
    const folder = await mkdtemp(join(tmpdir(), 'calendar-backup-'));
    try {
      const file = join(folder, 'journal.sqlite');
      await backup(db, file);
      res.set('Cache-Control', 'no-store');
      await new Promise((resolve, reject) =>
        res.download(file, 'journal.sqlite', (error) => (error ? reject(error) : resolve())),
      );
    } finally {
      await rm(folder, { recursive: true, force: true });
    }
  });
  const allPhotos = db.prepare('SELECT id,trip,event,endpoint,lat,lng FROM photos ORDER BY rowid');
  const allEvents = db.prepare('SELECT trip,block FROM events ORDER BY trip,day,start');
  const findPlace = db.prepare('SELECT block FROM events WHERE trip=? AND id=?');
  function place(trip, event, endpoint) {
    if (!['location', 'destination'].includes(endpoint)) return;
    const row = findPlace.get(trip, event);
    return row && JSON.parse(row.block)[endpoint];
  }
  const version = db.prepare(
    "SELECT total_changes() || '-' || data_version AS value FROM pragma_data_version",
  );
  const instance = randomUUID();
  let cachedVersion = -1;
  let payload = '';
  app.get('/api/atlas', (req, res) => {
    const { value } = version.get();
    res.set('ETag', '"' + instance + '-' + value + '"').set('Cache-Control', 'private, no-cache');
    if (req.get('If-None-Match') === res.get('ETag') || req.fresh) return res.status(304).end();
    if (cachedVersion === value) return res.type('json').send(payload);
    const trips = new Map(
      store
        .list()
        .map((cal) => [cal.id, { id: cal.id, name: cal.name, ...calendarWindow(cal.state), events: [] }]),
    );
    const photos = new Map();
    for (const p of allPhotos.all()) {
      const key = JSON.stringify([p.trip, p.event, p.endpoint]);
      if (!photos.has(key)) photos.set(key, { place: { lat: p.lat, lng: p.lng }, list: [] });
      photos
        .get(key)
        .list.push({ id: p.id, thumb: '/api/photos/' + p.id + '/thumb', url: '/api/photos/' + p.id });
    }
    for (const row of allEvents.all()) {
      const block = JSON.parse(row.block);
      // A world-map preview needs a bounded number of vertices per driving leg.
      // The timeline and the backup keep the original route resolution.
      const cache = block.preciseRouteCache;
      if (cache?.coords?.length > 256) {
        const coords = cache.coords;
        cache.coords = Array.from(
          { length: 256 },
          (_, i) => coords[Math.round((i * (coords.length - 1)) / 255)],
        );
      }
      const eventPhotos = {};
      for (const endpoint of ['location', 'destination']) {
        const saved = photos.get(JSON.stringify([row.trip, block.id, endpoint]));
        if (saved && !block[endpoint]) {
          // A photo stays reachable if its coordinates are cleared in the timeline.
          block[endpoint] = { ...saved.place, name: block.title || 'Photographed stop' };
          (block.photoOnlyEndpoints ??= []).push(endpoint);
        }
        if (block[endpoint]) eventPhotos[endpoint] = saved?.list || [];
      }
      trips.get(row.trip).events.push({ ...block, photos: eventPhotos });
    }
    payload = JSON.stringify([...trips.values()]);
    cachedVersion = value;
    res.type('json').send(payload);
  });
  app.post(
    '/api/calendars/:trip/events/:event/:endpoint/photos',
    express.raw({ type: Object.values(mimeTypes), limit: '16mb' }),
    async (req, res) => {
      const { trip, event, endpoint } = req.params;
      const stop = place(trip, event, endpoint);
      if (!stop) return res.status(404).json({ error: 'This stop has no saved location.' });
      if (!Buffer.isBuffer(req.body))
        return res.status(415).json({ error: 'Choose a JPEG, PNG or WebP photo.' });
      let thumb, mime;
      try {
        const image = sharp(req.body, { limitInputPixels: 64_000_000 });
        mime = mimeTypes[(await image.metadata()).format];
        if (!mime) throw new Error('Unsupported image');
        thumb = await image
          .rotate()
          .resize(360, 300, { fit: 'cover', withoutEnlargement: true })
          .webp({ quality: 77 })
          .toBuffer();
      } catch {
        return res
          .status(422)
          .json({ error: 'This image could not be read. Use a JPEG, PNG or WebP under 64 megapixels.' });
      }
      const current = place(trip, event, endpoint);
      if (!current || current.lat !== stop.lat || current.lng !== stop.lng)
        return res.status(409).json({ error: 'The location changed during upload.' });
      const id = randomUUID();
      db.prepare('INSERT INTO photos VALUES(?,?,?,?,?,?,?,?,?)').run(
        id,
        trip,
        event,
        endpoint,
        stop.lat,
        stop.lng,
        mime,
        req.body,
        thumb,
      );
      res.status(201).json({ id });
    },
  );
  app.delete('/api/photos/:id', (req, res) => {
    const deleted = db.prepare('DELETE FROM photos WHERE id=?').run(req.params.id);
    if (!deleted.changes) return res.status(404).json({ error: 'Photo not found.' });
    res.status(204).end();
  });
  for (const thumb of [false, true]) {
    const query = db.prepare(
      'SELECT mime,' + (thumb ? 'thumb' : 'original') + ' AS bytes FROM photos WHERE id=?',
    );
    app.get('/api/photos/:id' + (thumb ? '/thumb' : ''), (req, res) => {
      const p = query.get(req.params.id);
      if (!p) return res.status(404).end();
      res
        .set('Cache-Control', 'private, max-age=86400')
        .type(thumb ? 'image/webp' : p.mime)
        .send(Buffer.from(p.bytes));
    });
  }
}
