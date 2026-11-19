import { useState, useCallback, useEffect, useMemo, useRef, useLayoutEffect } from 'react';
import type { PreciseRouteCache, ViewMode, TimeBlock } from './types';
import { localDateString } from './types';
import AppShell from './components/AppShell';
import TopControls from './components/TopControls';
import DayGrid from './components/DayGrid';
import DaySidebar from './components/DaySidebar';
import MapView from './components/MapView';
import CalendarManager from './components/CalendarManager';
import './App.css';

interface CalendarState {
  numDays: number;
  startDate: string;
  startHour: number;
  endHour: number;
  viewMode: ViewMode;
  events: Record<number, TimeBlock[]>;
}

const DEFAULT_CALENDAR_STATE: CalendarState = {
  numDays: 5,
  startDate: localDateString(new Date()),
  startHour: 7,
  endHour: 22,
  viewMode: 'row',
  events: {},
};

interface CalendarResponse {
  state: CalendarState;
  revision: number;
}

function parseCalendarResponse(raw: unknown): CalendarResponse {
  const data = raw as Partial<CalendarState> & { state?: Partial<CalendarState>; revision?: number };
  const source = data.state ?? data;
  const date = source.startDate;
  const parsedDate = typeof date === 'string'
    ? new Date(date.length === 10 ? date + 'T00:00:00' : date)
    : null;
  const parsedNumDays = Number(source.numDays);
  const parsedStartHour = Number(source.startHour);
  const parsedEndHour = Number(source.endHour);

  const state: CalendarState = {
    numDays: Number.isFinite(parsedNumDays) && parsedNumDays > 0 ? parsedNumDays : DEFAULT_CALENDAR_STATE.numDays,
    startDate:
      parsedDate && !Number.isNaN(parsedDate.getTime())
        ? localDateString(parsedDate)
        : DEFAULT_CALENDAR_STATE.startDate,
    startHour: Number.isFinite(parsedStartHour)
      ? parsedStartHour
      : DEFAULT_CALENDAR_STATE.startHour,
    endHour: Number.isFinite(parsedEndHour) ? parsedEndHour : DEFAULT_CALENDAR_STATE.endHour,
    viewMode:
      source.viewMode === 'grid' || source.viewMode === 'day' || source.viewMode === 'row'
        ? source.viewMode
        : DEFAULT_CALENDAR_STATE.viewMode,
    events:
      typeof source.events === 'object' && source.events !== null
        ? (source.events as Record<number, TimeBlock[]>)
        : {},
  };

  return {
    state,
    revision: Number.isInteger(data.revision) ? Number(data.revision) : 0,
  };
}

interface SyncSession {
  id: string;
  revision: number;
  base: CalendarState;
  desired: CalendarState;
  busy: boolean;
}

function App() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [numDays, setNumDays] = useState(5);
  const [startDate, setStartDate] = useState(new Date());
  const [startHour, setStartHour] = useState(7);
  const [endHour, setEndHour] = useState(22);
  const [viewMode, setViewMode] = useState<ViewMode>('row');
  const [events, setEvents] = useState<Record<number, TimeBlock[]>>({});
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null);
  const [dayViewIndex, setDayViewIndex] = useState(0);
  const sessionRef = useRef<SyncSession | null>(null);
  const loadSeqRef = useRef(0);

  // Multi-calendar state
  const [currentCalendarId, setCurrentCalendarId] = useState<string | null>(null);
  const [currentCalendarName, setCurrentCalendarName] = useState('');
  const [showCalendarManager, setShowCalendarManager] = useState(false);

  const applyCalendarState = useCallback((state: CalendarState) => {
    setNumDays(state.numDays);
    setStartDate(new Date(state.startDate + 'T00:00:00'));
    setStartHour(state.startHour);
    setEndHour(state.endHour);
    setViewMode(state.viewMode);
    setEvents(state.events);
  }, []);

  // Load a specific calendar by ID
  const loadCalendar = useCallback(async (calId: string) => {
    const seq = ++loadSeqRef.current;
    try {
      const res = await fetch(`/api/calendars/${calId}`);
      if (!res.ok) throw new Error('Failed to load');
      const data = await res.json();
      const parsed = parseCalendarResponse(data);
      if (seq !== loadSeqRef.current) return;
      sessionRef.current = { id: calId, base: parsed.state, desired: parsed.state, revision: parsed.revision, busy: false };
      applyCalendarState(parsed.state);
      setCurrentCalendarId(calId);
      setCurrentCalendarName(data.name || 'Untitled');
    } catch (err) {
      console.error('Failed to load calendar:', err);
    }
  }, [applyCalendarState]);

  // On mount: check what calendars exist
  useEffect(() => {
    fetch('/api/calendars')
      .then((res) => {
        if (!res.ok) throw new Error('Failed to list');
        return res.json();
      })
      .then((list: { id: string; name: string; updatedAt: string }[]) => {
        if (list.length === 0) {
          setShowCalendarManager(true);
          setIsLoaded(true);
        } else {
          const mostRecent = list[0];
          loadCalendar(mostRecent.id).then(() => setIsLoaded(true));
        }
      })
      .catch((err) => {
        console.error('Failed to list calendars:', err);
        fetch('/api/calendar')
          .then((res) => {
            if (!res.ok) throw new Error('Failed to load');
            return res.json();
          })
          .then((data) => {
            const parsed = parseCalendarResponse(data);
            applyCalendarState(parsed.state);
          })
          .catch((err2) => {
            console.error('Failed to load calendar data:', err2);
          })
          .finally(() => setIsLoaded(true));
      });
  }, [applyCalendarState, loadCalendar]);

  // One request per calendar at a time. If the user edits during a save,
  // send their next delta against the snapshot they actually edited.
  const sync = useCallback(async (session: SyncSession) => {
    if (session.busy) return;
    session.busy = true;
    try {
      do {
        const submitted = session.desired;
        const dirty = submitted !== session.base;
        const res = await fetch(`/api/calendars/${session.id}`, dirty ? {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ state: submitted, baseState: session.base, baseRevision: session.revision }),
        } : undefined);
        if (!res.ok) throw new Error(`Calendar sync failed (HTTP ${res.status})`);
        const parsed = parseCalendarResponse(await res.json());
        if (session.desired !== submitted) {
          if (dirty) {
            session.base = submitted;
            session.revision = -1; // Force a three-way merge, including any concurrent agent addition.
          }
          continue;
        }
        if (dirty || parsed.revision !== session.revision) {
          session.revision = parsed.revision;
          session.base = session.desired = parsed.state;
          if (sessionRef.current === session) applyCalendarState(parsed.state);
        }
      } while (session.desired !== session.base);
    } catch (err) {
      console.error('Failed to sync calendar:', err);
    } finally {
      session.busy = false;
    }
  }, [applyCalendarState]);

  useLayoutEffect(() => {
    const session = sessionRef.current;
    if (!session || session.id !== currentCalendarId) return;
    const state = { numDays, startDate: localDateString(startDate), startHour, endHour, viewMode, events };
    if (JSON.stringify(state) === JSON.stringify(session.desired)) return;
    session.desired = state;
    void sync(session);
  }, [currentCalendarId, numDays, startDate, startHour, endHour, viewMode, events, sync]);

  useEffect(() => {
    if (!currentCalendarId) return;
    const interval = window.setInterval(() => {
      const session = sessionRef.current;
      if (session?.id === currentCalendarId) void sync(session);
    }, 1000);
    return () => window.clearInterval(interval);
  }, [currentCalendarId, sync]);

  useEffect(() => {
    if (dayViewIndex >= numDays) {
      setDayViewIndex(Math.max(0, numDays - 1));
    }
  }, [numDays, dayViewIndex]);

  const handleNumDaysChange = useCallback((n: number) => {
    setNumDays(n);
    if (n > 7) setViewMode('grid');
  }, []);

  const handleBlockCreated = useCallback((block: TimeBlock) => {
    setEvents((prev) => ({
      ...prev,
      [block.dayIndex]: [...(prev[block.dayIndex] || []), block],
    }));
  }, []);

  const handleBlockUpdated = useCallback((updated: TimeBlock) => {
    setEvents((prev) => ({
      ...prev,
      [updated.dayIndex]: (prev[updated.dayIndex] || []).map((block) =>
        block.id === updated.id ? updated : block
      ),
    }));
  }, []);

  const handleBlockDeleted = useCallback((block: TimeBlock) => {
    setEvents((prev) => ({
      ...prev,
      [block.dayIndex]: (prev[block.dayIndex] || []).filter((item) => item.id !== block.id),
    }));
  }, []);

  const handleHoverEvent = useCallback((id: string | null) => {
    setHoveredEventId(id);
  }, []);

  const handlePreciseRouteCacheChange = useCallback(
    (eventId: string, dayIndex: number, cache: PreciseRouteCache | null) => {
      setEvents((prev) => {
        const dayEvents = prev[dayIndex] || [];
        let changed = false;
        const nextDayEvents = dayEvents.map((event) => {
          if (event.id !== eventId) return event;
          changed = true;
          return {
            ...event,
            preciseRouteCache: cache ?? undefined,
          };
        });

        if (!changed) return prev;
        return {
          ...prev,
          [dayIndex]: nextDayEvents,
        };
      });
    },
    []
  );

  const handlePrevDay = useCallback(() => {
    setDayViewIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const handleNextDay = useCallback(() => {
    setDayViewIndex((prev) => Math.min(numDays - 1, prev + 1));
  }, [numDays]);

  const handleSelectDay = useCallback((index: number) => {
    setDayViewIndex(index);
  }, []);

  const handleSelectCalendar = useCallback(async (id: string) => {
    await loadCalendar(id);
    setShowCalendarManager(false);
  }, [loadCalendar]);

  const handleCreateCalendar = useCallback(async (name: string) => {
    try {
      const res = await fetch('/api/calendars', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (!res.ok) throw new Error('Failed to create');
      const data = await res.json();
      await loadCalendar(data.id);
      setShowCalendarManager(false);
    } catch (err) {
      console.error('Failed to create calendar:', err);
    }
  }, [loadCalendar]);

  const handleRenameCalendar = useCallback(async (id: string, name: string) => {
    try {
      await fetch(`/api/calendars/${id}/rename`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
      });
      if (id === currentCalendarId) {
        setCurrentCalendarName(name);
      }
    } catch (err) {
      console.error('Failed to rename calendar:', err);
    }
  }, [currentCalendarId]);

  const handleDeleteCalendarFromManager = useCallback(async (id: string) => {
    try {
      await fetch(`/api/calendars/${id}`, { method: 'DELETE' });
      if (id === currentCalendarId) {
        sessionRef.current = null;
        setCurrentCalendarId(null);
        setCurrentCalendarName('');
        applyCalendarState(DEFAULT_CALENDAR_STATE);
      }
    } catch (err) {
      console.error('Failed to delete calendar:', err);
    }
  }, [currentCalendarId, applyCalendarState]);

  const handleDeleteCurrentCalendar = useCallback(async () => {
    if (!currentCalendarId) return;
    const confirmed = window.confirm(`Delete "${currentCalendarName}"? This action cannot be undone.`);
    if (!confirmed) return;

    try {
      await fetch(`/api/calendars/${currentCalendarId}`, { method: 'DELETE' });
      sessionRef.current = null;
      setCurrentCalendarId(null);
      setCurrentCalendarName('');
      applyCalendarState(DEFAULT_CALENDAR_STATE);
      setShowCalendarManager(true);
    } catch (err) {
      console.error('Failed to delete calendar:', err);
    }
  }, [currentCalendarId, currentCalendarName, applyCalendarState]);

  const mapEvents = useMemo(() => {
    if (viewMode === 'day') {
      const dayEvents = events[dayViewIndex];
      return dayEvents ? { [dayViewIndex]: dayEvents } : {};
    }
    return events;
  }, [viewMode, dayViewIndex, events]);

  const isDayView = viewMode === 'day';

  if (!isLoaded) {
    return (
      <AppShell>
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            alignItems: 'center',
            height: '100vh',
            color: 'var(--text-med)',
          }}
        >
          Loading calendar...
        </div>
      </AppShell>
    );
  }

  return (
    <AppShell>
      <TopControls
        numDays={numDays}
        startDate={startDate}
        startHour={startHour}
        endHour={endHour}
        viewMode={viewMode}
        onNumDaysChange={handleNumDaysChange}
        onStartDateChange={setStartDate}
        onStartHourChange={setStartHour}
        onEndHourChange={setEndHour}
        onViewModeChange={setViewMode}
        calendarName={currentCalendarName}
        onOpenCalendarManager={() => setShowCalendarManager(true)}
        onDeleteCalendar={handleDeleteCurrentCalendar}
        hasCalendar={!!currentCalendarId}
      />
      <div className={`app-main ${isDayView ? 'app-main-day' : ''}`}>
        {isDayView && (
          <DaySidebar
            numDays={numDays}
            selectedDay={dayViewIndex}
            onSelectDay={handleSelectDay}
          />
        )}
        <div className={`app-calendar-section ${isDayView ? 'app-calendar-section-day' : ''}`}>
          <DayGrid
            numDays={numDays}
            startDate={startDate}
            startHour={startHour}
            endHour={endHour}
            viewMode={viewMode}
            events={events}
            hoveredEventId={hoveredEventId}
            onBlockCreated={handleBlockCreated}
            onBlockUpdated={handleBlockUpdated}
            onBlockDeleted={handleBlockDeleted}
            onHoverEvent={handleHoverEvent}
            dayViewIndex={dayViewIndex}
            onPrevDay={handlePrevDay}
            onNextDay={handleNextDay}
          />
        </div>
        {isDayView ? (
          <div className="app-map-section-day">
            <MapView
              events={mapEvents}
              hoveredEventId={hoveredEventId}
              onHoverEvent={handleHoverEvent}
              preciseZoomEnabled
              onPreciseRouteCacheChange={handlePreciseRouteCacheChange}
            />
          </div>
        ) : (
          <MapView
            events={mapEvents}
            hoveredEventId={hoveredEventId}
            onHoverEvent={handleHoverEvent}
            preciseZoomEnabled={false}
            onPreciseRouteCacheChange={handlePreciseRouteCacheChange}
          />
        )}
      </div>

      <CalendarManager
        visible={showCalendarManager}
        currentCalendarId={currentCalendarId}
        onSelectCalendar={handleSelectCalendar}
        onCreateCalendar={handleCreateCalendar}
        onRenameCalendar={handleRenameCalendar}
        onDeleteCalendar={handleDeleteCalendarFromManager}
        onClose={() => setShowCalendarManager(false)}
      />

    </AppShell>
  );
}

export default App;
