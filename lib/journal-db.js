import { DatabaseSync } from 'node:sqlite';
import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';

// Keep image bytes off both the calendar sync path and the atlas index.
export function openJournal(directory, normalizeState, legacyFile) {
  fs.mkdirSync(directory, { recursive: true });
  const db = new DatabaseSync(path.join(directory, 'journal.sqlite'));
  db.exec(`PRAGMA foreign_keys=ON; PRAGMA journal_mode=WAL; PRAGMA busy_timeout=5000;
        CREATE TABLE IF NOT EXISTS trips (
            id TEXT PRIMARY KEY, name TEXT NOT NULL, settings TEXT NOT NULL, revision INTEGER NOT NULL, updatedAt TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS trip_recency ON trips(updatedAt DESC);
        CREATE TABLE IF NOT EXISTS events (
            trip TEXT NOT NULL REFERENCES trips(id) ON DELETE CASCADE, id TEXT NOT NULL,
            day INTEGER NOT NULL, start INTEGER NOT NULL, block TEXT NOT NULL, PRIMARY KEY(trip,id));
        CREATE INDEX IF NOT EXISTS itinerary ON events(trip,day,start);
        CREATE TABLE IF NOT EXISTS photos (
            id TEXT PRIMARY KEY, trip TEXT NOT NULL, event TEXT NOT NULL, endpoint TEXT NOT NULL,
            lat REAL NOT NULL, lng REAL NOT NULL, mime TEXT NOT NULL, original BLOB NOT NULL, thumb BLOB NOT NULL,
            FOREIGN KEY(trip,event) REFERENCES events(trip,id) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS stop_photos ON photos(trip,event,endpoint);
    `);
  function transaction(fn) {
    db.exec('BEGIN IMMEDIATE');
    try {
      const result = fn();
      db.exec('COMMIT');
      return result;
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  }
  const upsertTrip = db.prepare(
    'INSERT INTO trips VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET name=excluded.name,settings=excluded.settings,revision=excluded.revision,updatedAt=excluded.updatedAt',
  );
  const upsertEvent = db.prepare(
    'INSERT INTO events VALUES(?,?,?,?,?) ON CONFLICT(trip,id) DO UPDATE SET day=excluded.day,start=excluded.start,block=excluded.block',
  );
  const readBlocks = db.prepare('SELECT id,block FROM events WHERE trip=? ORDER BY day,start,rowid');
  const removeEvent = db.prepare('DELETE FROM events WHERE trip=? AND id=?');
  function write(trip) {
    const { events, ...settings } = trip.state;
    upsertTrip.run(trip.id, trip.name, JSON.stringify(settings), trip.revision, trip.updatedAt);
    const old = new Map(readBlocks.all(trip.id).map((row) => [row.id, row.block]));
    for (const block of Object.values(events).flat()) {
      const json = JSON.stringify(block);
      if (json !== old.get(block.id))
        upsertEvent.run(trip.id, block.id, block.dayIndex, block.startMinutes, json);
      old.delete(block.id);
    }
    for (const id of old.keys()) removeEvent.run(trip.id, id);
  }
  // One transaction and a durable version marker prevent partial or repeated imports.
  if (db.prepare('PRAGMA user_version').get().user_version === 0)
    transaction(() => {
      const files = fs.readdirSync(directory).filter((f) => f.endsWith('.json'));
      function migrate(file, id) {
        const raw = JSON.parse(fs.readFileSync(file, 'utf8'));
        write({
          id,
          name: raw.name || 'My Calendar',
          state: normalizeState(raw.state || raw),
          revision: raw.revision ?? 0,
          updatedAt: raw.updatedAt || new Date().toISOString(),
        });
      }
      for (const file of files) migrate(path.join(directory, file), file.slice(0, -5));
      if (!files.length && legacyFile && fs.existsSync(legacyFile)) migrate(legacyFile, randomUUID());
      db.exec('PRAGMA user_version=1');
    });
  const tripRow = db.prepare('SELECT * FROM trips WHERE id=?');
  const summaries = db.prepare('SELECT * FROM trips ORDER BY updatedAt DESC');
  const shape = (row) => ({
    id: row.id,
    name: row.name,
    revision: row.revision,
    updatedAt: row.updatedAt,
    state: { ...JSON.parse(row.settings), events: {} },
  });
  function get(id) {
    const row = tripRow.get(id);
    if (!row) return;
    const cal = shape(row);
    for (const item of readBlocks.all(id)) {
      const block = JSON.parse(item.block);
      (cal.state.events[block.dayIndex] ??= []).push(block);
    }
    return cal;
  }
  return {
    db,
    get,
    list: () => summaries.all().map(shape),
    save: (cal) => transaction(() => write(cal)),
    delete: (id) => db.prepare('DELETE FROM trips WHERE id=?').run(id),
  };
}
