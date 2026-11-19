import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft,
  ArrowRight,
  ArrowUpRight,
  CalendarDays,
  Camera,
  Download,
  ChevronRight,
  Globe2,
  Images,
  MapPin,
  MapPinned,
  Plus,
  Route,
  Search,
  Upload,
  X,
} from 'lucide-react';
import { modeLabels, localDateString, minutesToTime } from '../types';
import PlaceFinder from './PlaceFinder';
import AtlasMap from './AtlasMap';
import {
  connections,
  dateLabel,
  dayLabel,
  directions,
  stops,
  type Photo,
  type Stop,
  type Trip,
} from './model';
import { samples } from './samples';
import './journal.css';

async function request(url: string, options?: RequestInit) {
  const response = await fetch(url, options);
  if (!response.ok) {
    const message = await response.json().catch(() => ({}));
    throw new Error(message.error || 'Request failed (' + response.status + ').');
  }
  return response.status === 204 ? null : response.json();
}
const post = (body: unknown) => ({
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify(body),
});
const messageOf = (e: unknown) => (e instanceof Error ? e.message : 'Please try again.');
const dateRange = (trip: Trip) =>
  dateLabel(trip.startDate, { month: 'short', day: 'numeric', year: 'numeric' });
type Coordinates = { lat: number; lng: number };
type Props = { onTimeline: (id?: string) => void; initialTrip: string | null };

function Dialog({
  title,
  close,
  children,
}: {
  title: string;
  close: () => void;
  children: React.ReactNode;
}) {
  const ref = useRef<HTMLDialogElement>(null);
  useEffect(() => {
    const el = ref.current;
    el?.showModal();
    return () => el?.close();
  }, []);
  return (
    <dialog ref={ref} className="journal-dialog" onCancel={close} aria-label={title}>
      <header>
        <h2>{title}</h2>
        <button className="icon-btn" aria-label="Close dialog" onClick={close}>
          <X />
        </button>
      </header>
      {children}
    </dialog>
  );
}

export default function Journal({ onTimeline, initialTrip }: Props) {
  const [data, setData] = useState<Trip[] | null>(null);
  const [id, setId] = useState<string | null>(initialTrip);
  const [wall, setWall] = useState(false);
  const [indexOpen, setIndexOpen] = useState(false);
  const [search, setSearch] = useState('');
  const [focus, setFocus] = useState<Stop | null>(null);
  const [selectedKey, setSelectedKey] = useState('');
  const [newTrip, setNewTrip] = useState(false);
  const [adding, setAdding] = useState(false);
  const [picked, setPicked] = useState<Coordinates | null>(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);
  const [viewer, setViewer] = useState<{ photo: Photo; stop: Stop } | null>(null);
  const version = useRef(0);
  const etag = useRef('');
  const refresh = useCallback(async () => {
    const seq = ++version.current;
    const response = await fetch('/api/atlas', {
      cache: 'no-store',
      headers: etag.current ? { 'If-None-Match': etag.current } : {},
    });
    if (response.status === 304) return;
    if (!response.ok) throw new Error('Could not load the travel atlas.');
    const result: Trip[] = await response.json();
    if (seq === version.current) {
      etag.current = response.headers.get('ETag') || '';
      setData(result);
    }
  }, []);
  useEffect(() => {
    const update = () => refresh().catch((e) => setError(messageOf(e)));
    void update();
    const timer = window.setInterval(update, 5000);
    return () => window.clearInterval(timer);
  }, [refresh]);
  const demo = data?.length === 0;
  const trips = demo ? samples : data;
  const trip = trips?.find((t) => t.id === id);
  const filtered = useMemo(
    () =>
      (trips || []).filter((t) =>
        (
          t.name +
          ' ' +
          t.events.map((e) => (e.location?.name || '') + ' ' + (e.destination?.name || '')).join(' ')
        )
          .toLowerCase()
          .includes(search.toLowerCase()),
      ),
    [trips, search],
  );
  const points = useMemo(() => (trip ? [trip] : filtered).flatMap(stops), [trip, filtered]);
  const legs = useMemo(() => connections(trip ? points : []), [points, trip]);
  const collection = useMemo(
    () => points.flatMap((stop) => stop.photos.map((photo) => ({ photo, stop }))),
    [points],
  );
  const navigate = (next: string | null) => {
    setId(next);
    setIndexOpen(false);
    setSelectedKey('');
    setFocus(null);
    setAdding(false);
    setPicked(null);
    setSearch('');
    setError('');
  };
  const select = (stop: Stop) => {
    if (trip?.id !== stop.trip.id) navigate(stop.trip.id);
    setSelectedKey(stop.key);
    setFocus({ ...stop });
    setWall(false);
  };
  async function createTrip(form: HTMLFormElement) {
    const values = new FormData(form);
    setBusy(true);
    setError('');
    try {
      const cal = await request(
        '/api/calendars',
        post({ name: values.get('name'), startDate: values.get('date') }),
      );
      await refresh();
      navigate(cal.id);
      setWall(false);
      setNewTrip(false);
      setAdding(true);
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }
  const retryId = useRef('');
  async function saveStop(form: HTMLFormElement) {
    if (!trip || demo) return;
    if (!picked) {
      setError('Choose the location on the map or enter its coordinates.');
      return;
    }
    const f = new FormData(form);
    setBusy(true);
    setError('');
    try {
      const saved = await request(
        '/api/calendars/' + trip.id + '/events',
        post({
          eventId: retryId.current || (retryId.current = crypto.randomUUID()),
          title: f.get('title'),
          date: f.get('date'),
          startTime: f.get('start'),
          endTime: f.get('end'),
          location: { name: f.get('place'), ...picked },
          transport: f.get('transport') || undefined,
        }),
      );
      await refresh();
      setAdding(false);
      setPicked(null);
      setSelectedKey(JSON.stringify([trip.id, saved.event.id, 'location']));
      retryId.current = '';
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }
  async function changeTransport(stop: Stop, transport: string) {
    setBusy(true);
    setError('');
    try {
      const url =
        '/api/calendars/' +
        encodeURIComponent(stop.trip.id) +
        '/events/' +
        encodeURIComponent(stop.event.id) +
        '/transport';
      await request(url, { ...post({ transport: transport || null }), method: 'PATCH' });
      await refresh();
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setBusy(false);
    }
  }
  async function upload(stop: Stop, files: File[]) {
    if (!files.length || demo) return;
    setBusy(true);
    setError('');
    try {
      const [tripId, eventId, endpoint] = [stop.trip.id, stop.event.id, stop.endpoint].map(
        encodeURIComponent,
      );
      for (const file of files) {
        if (file.size > 16 * 1024 * 1024) throw new Error(file.name + ' exceeds the 16 MB limit.');
        await request('/api/calendars/' + tripId + '/events/' + eventId + '/' + endpoint + '/photos', {
          method: 'POST',
          headers: { 'Content-Type': file.type },
          body: file,
        });
      }
    } catch (e) {
      setError(messageOf(e));
    } finally {
      await refresh().catch((e) => setError(messageOf(e)));
      setBusy(false);
    }
  }
  const tally = (n: number, noun: string) => n + ' ' + noun + (n === 1 ? '' : 's');
  const startAdd = () => {
    retryId.current = '';
    setAdding(true);
    setWall(false);
    setPicked(null);
    setError('');
  };
  return (
    <main className="journal">
      <header className="journal-nav">
        <button
          className="journal-brand"
          aria-label="Open Map Calendar"
          onClick={() => {
            navigate(null);
            setWall(false);
          }}
        >
          <MapPinned size={30} />
          <span>Open Map Calendar</span>
        </button>
        <nav aria-label="Workspace">
          <button
            aria-expanded={trip ? undefined : indexOpen}
            aria-controls="trip-index"
            onClick={() => {
              if (trip) navigate(null);
              else {
                setIndexOpen(!indexOpen);
                if (indexOpen) setSearch('');
              }
            }}
          >
            <Globe2 />
            {trip ? 'Atlas' : 'Trips'}
          </button>
          <button onClick={() => onTimeline(demo ? undefined : trip?.id)}>
            <CalendarDays />
            Calendar
          </button>
        </nav>
        <button
          className="primary new-trip"
          onClick={() => {
            setNewTrip(true);
            setError('');
          }}
        >
          <Plus />
          New trip
        </button>
      </header>
      <div className="journal-content">
        {error && !newTrip && (
          <div className="journal-error" role="alert">
            {error}
            <button aria-label="Dismiss error" onClick={() => setError('')}>
              <X size={16} />
            </button>
          </div>
        )}
        <div className={'atlas-workspace' + (!trip && !indexOpen ? ' map-only' : '')}>
          <aside id="trip-index" className="trip-list" hidden={!trip && !indexOpen}>
            <header className="journal-heading">
              {trip && (
                <button className="back-link" onClick={() => navigate(null)}>
                  <ArrowLeft size={14} />
                  World atlas
                </button>
              )}
              <h1>{trip ? trip.name : 'World atlas'}</h1>
              <p>
                {trip ? dateRange(trip) : tally(filtered.length, 'trip')}
                <i /> {tally(points.length, 'stop')}
                <i />
                {tally(collection.length, 'photo')}
              </p>
            </header>
            {trip && (
              <div className="list-title">
                <h2>{adding ? 'Add a stop' : 'Itinerary'}</h2>
                {demo && <span className="sample-label">Sample trip</span>}
              </div>
            )}
            {adding && trip ? (
              <form
                className="stop-form"
                onSubmit={(e) => {
                  e.preventDefault();
                  void saveStop(e.currentTarget);
                }}
              >
                <label>
                  Event title
                  <input name="title" placeholder="Morning at the museum" required autoFocus />
                </label>
                <PlaceFinder onPick={setPicked} />
                <label>
                  Date
                  <input type="date" name="date" defaultValue={trip.startDate} required />
                </label>
                <div className="form-pair">
                  <label>
                    From
                    <input name="start" type="time" defaultValue="09:00" required />
                  </label>
                  <label>
                    To
                    <input name="end" type="time" defaultValue="10:00" required />
                  </label>
                </div>
                <label>
                  Arrive by
                  <select name="transport">
                    <option value="">Unspecified</option>
                    {Object.entries(modeLabels).map(([v, t]) => (
                      <option key={v} value={v}>
                        {t}
                      </option>
                    ))}
                  </select>
                </label>
                <fieldset>
                  <legend>Map position</legend>
                  <p>Click the map or enter coordinates.</p>
                  <div className="form-pair">
                    <label>
                      Latitude
                      <input
                        aria-label="Latitude"
                        type="number"
                        step="any"
                        min="-90"
                        max="90"
                        required
                        value={picked?.lat ?? ''}
                        onChange={(e) => setPicked({ lat: Number(e.target.value), lng: picked?.lng ?? 0 })}
                      />
                    </label>
                    <label>
                      Longitude
                      <input
                        aria-label="Longitude"
                        type="number"
                        step="any"
                        min="-180"
                        max="180"
                        required
                        value={picked?.lng ?? ''}
                        onChange={(e) => setPicked({ lng: Number(e.target.value), lat: picked?.lat ?? 0 })}
                      />
                    </label>
                  </div>
                </fieldset>
                <div className="form-buttons">
                  <button
                    type="button"
                    onClick={() => {
                      setAdding(false);
                      setPicked(null);
                    }}
                  >
                    Cancel
                  </button>
                  <button className="primary" disabled={busy}>
                    Save stop
                  </button>
                </div>
              </form>
            ) : trip ? (
              <>
                {!demo && (
                  <div className="trip-actions">
                    <button onClick={startAdd}>
                      <Plus />
                      Add stop
                    </button>
                    <button onClick={() => onTimeline(trip.id)}>
                      <CalendarDays />
                      Timeline
                    </button>
                  </div>
                )}
                {!trip.events.length && (
                  <p className="journal-empty">Add a timed stop to start this itinerary.</p>
                )}
                {trip.events.map((event, index) => (
                  <div key={event.id} className="visit">
                    {(index === 0 || event.dayIndex !== trip.events[index - 1].dayIndex) && (
                      <h3>{dayLabel(trip, event.dayIndex)}</h3>
                    )}
                    <div className="visit-title">
                      <time>
                        {minutesToTime(event.startMinutes)}–{minutesToTime(event.endMinutes)}
                      </time>
                      <strong>{event.title || 'Untitled event'}</strong>
                    </div>
                    {points
                      .filter((s) => s.event.id === event.id)
                      .map((s) => {
                        const leg = legs.find((l) => l.to.key === s.key);
                        return (
                          <div className="visit-stop" key={s.key}>
                            {leg && (
                              <div className="connection">
                                <Route size={14} />
                                <div>
                                  {leg.mode ? modeLabels[leg.mode] : 'Transfer'} from {leg.from.place.name}
                                  <small>
                                    {leg.depart} → {leg.arrive} ·{' '}
                                    {leg.minutes < 0
                                      ? Math.abs(leg.minutes) + ' min overlap'
                                      : leg.minutes + ' min available'}{' '}
                                    ·{' '}
                                    <a href={directions(leg)} target="_blank" rel="noreferrer">
                                      Directions ↗
                                    </a>
                                  </small>
                                </div>
                              </div>
                            )}
                            <button
                              className={'stop-link ' + (selectedKey === s.key ? 'selected' : '')}
                              onClick={() => select(s)}
                            >
                              <MapPin />
                              {s.time} · {s.place.name}
                              <ChevronRight />
                            </button>
                            {s.key === selectedKey && (
                              <div className="stop-detail">
                                {!demo && (
                                  <label className="stop-mode">
                                    Travel mode
                                    <select
                                      disabled={busy}
                                      value={s.event.transport || ''}
                                      onChange={(e) => void changeTransport(s, e.target.value)}
                                    >
                                      <option value="">Unspecified</option>
                                      {Object.entries(modeLabels).map(([value, title]) => (
                                        <option key={value} value={value}>
                                          {title}
                                        </option>
                                      ))}
                                    </select>
                                  </label>
                                )}
                                {s.photos.map((photo) => (
                                  <button
                                    key={photo.id}
                                    aria-label={'View photo of ' + s.place.name}
                                    onClick={() => setViewer({ stop: s, photo })}
                                  >
                                    <img src={photo.thumb} alt={s.place.name} />
                                  </button>
                                ))}
                                {!demo && !s.event.photoOnlyEndpoints?.includes(s.endpoint) && (
                                  <label className="upload-btn">
                                    <Upload />
                                    {busy ? 'Uploading…' : 'Add photos'}
                                    <input
                                      type="file"
                                      accept="image/jpeg,image/png,image/webp"
                                      multiple
                                      disabled={busy}
                                      onChange={(e) => {
                                        void upload(s, Array.from(e.target.files || []));
                                        e.target.value = '';
                                      }}
                                    />
                                  </label>
                                )}
                              </div>
                            )}
                          </div>
                        );
                      })}
                    {!event.location && !event.destination && (
                      <span className="muted">No mapped location</span>
                    )}
                  </div>
                ))}
              </>
            ) : (
              <>
                <label className="trip-search">
                  <Search />
                  <input
                    aria-label="Search trips and places"
                    placeholder="Search trips or places"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                </label>
                <section className="trip-index" aria-label={demo ? 'Sample trips' : 'Trips'}>
                  <h2>{demo ? 'Sample trips' : 'Trips'}</h2>
                  {filtered.map((t) => (
                    <button className="trip-entry" key={t.id} onClick={() => navigate(t.id)}>
                      <span>
                        <strong>{t.name}</strong>
                        <small>
                          <time dateTime={t.startDate}>{dateRange(t)}</time>
                          <span aria-hidden="true"> · </span>
                          {tally(stops(t).length, 'stop')}
                        </small>
                      </span>
                      <ChevronRight className="entry-arrow" />
                    </button>
                  ))}
                </section>
                {!filtered.length && (
                  <p className="journal-empty">{data === null ? 'Loading trips…' : 'No matching trips.'}</p>
                )}
                {data && !demo && (
                  <a className="journal-backup" href="/api/journal/backup" download="journal.sqlite">
                    <Download size={16} />
                    Back up journal
                  </a>
                )}
              </>
            )}
          </aside>
          <section className="map-board" aria-label={wall ? 'Photo wall' : 'Map board'}>
            <div className="view-switch" role="group" aria-label="Atlas view">
              <button aria-pressed={!wall} onClick={() => setWall(false)}>
                <Globe2 />
                Map
              </button>
              <button
                aria-pressed={wall}
                onClick={() => {
                  setWall(true);
                  setAdding(false);
                }}
              >
                <Images />
                Photo wall
              </button>
            </div>
            {wall ? (
              <div className="photo-grid">
                {collection.map(({ photo, stop }) => (
                  <button className="wall-print" key={photo.id} onClick={() => setViewer({ photo, stop })}>
                    <img src={photo.thumb} alt={stop.place.name} loading="lazy" />
                    <strong>{stop.place.name}</strong>
                    <span>
                      {dayLabel(stop.trip, stop.event.dayIndex)} · {stop.time}
                    </span>
                  </button>
                ))}
                {!collection.length && (
                  <div className="wall-empty">
                    <Camera />
                    <h2>No photographs yet</h2>
                    <p>Select a mapped stop in its itinerary to add photos.</p>
                  </div>
                )}
              </div>
            ) : (
              <AtlasMap
                key={trip ? 'trip-' + trip.id : 'world'}
                points={points}
                tripId={trip?.id || null}
                focus={focus}
                onSelect={select}
                onWorld={() => navigate(null)}
                onPick={adding ? setPicked : undefined}
                picked={picked}
              />
            )}
          </section>
        </div>
      </div>
      {newTrip && (
        <Dialog title="New trip" close={() => setNewTrip(false)}>
          <form
            className="stop-form"
            onSubmit={(e) => {
              e.preventDefault();
              void createTrip(e.currentTarget);
            }}
          >
            <label>
              Trip name
              <input name="name" required autoFocus placeholder="A weekend in Montréal" />
            </label>
            <label>
              Start date
              <input required name="date" type="date" defaultValue={localDateString(new Date())} />
            </label>
            {error && <p role="alert">{error}</p>}
            <button className="primary" disabled={busy}>
              Create trip <ArrowRight />
            </button>
          </form>
        </Dialog>
      )}
      {viewer && (
        <Dialog title={viewer.stop.place.name} close={() => setViewer(null)}>
          <img className="photo-original" src={viewer.photo.url} alt={viewer.stop.place.name} />
          <div className="photo-caption">
            <span>
              {dayLabel(viewer.stop.trip, viewer.stop.event.dayIndex)} · {viewer.stop.time}
            </span>
            <button
              onClick={() => {
                select(viewer.stop);
                setViewer(null);
              }}
            >
              View stop <ArrowUpRight size={15} />
            </button>
          </div>
          {!demo && (
            <button
              className="remove-photo"
              disabled={busy}
              onClick={async () => {
                if (!window.confirm('Remove this photograph from the journal?')) return;
                setBusy(true);
                try {
                  await request(viewer.photo.url, { method: 'DELETE' });
                  setViewer(null);
                  await refresh();
                } catch (e) {
                  setViewer(null);
                  setError(messageOf(e));
                } finally {
                  setBusy(false);
                }
              }}
            >
              Remove photo
            </button>
          )}
        </Dialog>
      )}
    </main>
  );
}
