import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type { ViewMode, TimeBlock } from './types';
import AppShell from './components/AppShell';
import TopControls from './components/TopControls';
import DayGrid from './components/DayGrid';
import DaySidebar from './components/DaySidebar';
import MapView from './components/MapView';
import CalendarManager from './components/CalendarManager';
import './App.css';



interface CalendarState {
  numDays: number;
  startDate: string; // ISO string
  startHour: number;
  endHour: number;
  viewMode: ViewMode;
  events: Record<number, TimeBlock[]>;
}

const DEFAULT_CALENDAR_STATE: CalendarState = {
  numDays: 5,
  startDate: new Date().toISOString(),
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
  const parsedDate = typeof source.startDate === 'string' ? new Date(source.startDate) : null;

  const state: CalendarState = {
    numDays: Number.isFinite(source.numDays) && source.numDays! > 0 ? source.numDays! : DEFAULT_CALENDAR_STATE.numDays,
    startDate: parsedDate && !Number.isNaN(parsedDate.getTime()) ? parsedDate.toISOString() : DEFAULT_CALENDAR_STATE.startDate,
    startHour: Number.isFinite(source.startHour) ? source.startHour! : DEFAULT_CALENDAR_STATE.startHour,
    endHour: Number.isFinite(source.endHour) ? source.endHour! : DEFAULT_CALENDAR_STATE.endHour,
    viewMode: source.viewMode === 'grid' || source.viewMode === 'day' || source.viewMode === 'row' ? source.viewMode : DEFAULT_CALENDAR_STATE.viewMode,
    events: typeof source.events === 'object' && source.events !== null ? source.events as Record<number, TimeBlock[]> : {},
  };

  return {
    state,
    revision: Number.isInteger(data.revision) ? data.revision! : 0,
  };
}

function App() {
  const [isLoaded, setIsLoaded] = useState(false);
  const [apiOnline, setApiOnline] = useState(false);
  const [numDays, setNumDays] = useState(5);
  const [startDate, setStartDate] = useState(new Date());
  const [startHour, setStartHour] = useState(7);
  const [endHour, setEndHour] = useState(22);
  const [viewMode, setViewMode] = useState<ViewMode>('row');
  const [events, setEvents] = useState<Record<number, TimeBlock[]>>({});
  const [hoveredEventId, setHoveredEventId] = useState<string | null>(null);
  const [dayViewIndex, setDayViewIndex] = useState(0);
  const serverRevisionRef = useRef(0);
  const baseStateRef = useRef<CalendarState>(DEFAULT_CALENDAR_STATE);
  const saveSeqRef = useRef(0);

  // Multi-calendar state
  const [currentCalendarId, setCurrentCalendarId] = useState<string | null>(null);
  const [currentCalendarName, setCurrentCalendarName] = useState('');
  const [showCalendarManager, setShowCalendarManager] = useState(false);

  const applyCalendarState = useCallback((state: CalendarState) => {
    setNumDays(state.numDays);
    setStartDate(new Date(state.startDate));
    setStartHour(state.startHour);
    setEndHour(state.endHour);
    setViewMode(state.viewMode);
    setEvents(state.events);
  }, []);

  // Load a specific calendar by ID
  const loadCalendar = useCallback(async (calId: string) => {
    try {
      const res = await fetch(`/api/calendars/${calId}`);
      if (!res.ok) throw new Error('Failed to load');
      const data = await res.json();
      const parsed = parseCalendarResponse(data);
      serverRevisionRef.current = parsed.revision;
      baseStateRef.current = parsed.state;
      applyCalendarState(parsed.state);
      setCurrentCalendarId(calId);
      setCurrentCalendarName(data.name || 'Untitled');
      setApiOnline(true);
    } catch (err) {
      console.error('Failed to load calendar:', err);
      setApiOnline(false);
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
        setApiOnline(true);
        if (list.length === 0) {
          // No calendars — show manager so user can create one
          setShowCalendarManager(true);
          setIsLoaded(true);
        } else {
          // Load the most recent calendar
          const mostRecent = list[0];
          loadCalendar(mostRecent.id).then(() => setIsLoaded(true));
        }
      })
      .catch((err) => {
        console.error('Failed to list calendars:', err);
        // Fallback to legacy endpoint
        fetch('/api/calendar')
          .then((res) => {
            if (!res.ok) throw new Error('Failed to load');
            return res.json();
          })
          .then((data) => {
            const parsed = parseCalendarResponse(data);
            serverRevisionRef.current = parsed.revision;
            baseStateRef.current = parsed.state;
            applyCalendarState(parsed.state);
            setApiOnline(true);
          })
          .catch((err2) => {
            console.error('Failed to load calendar data:', err2);
            setApiOnline(false);
          })
          .finally(() => setIsLoaded(true));
      });
  }, [applyCalendarState, loadCalendar]);

  // Pull remote updates so all connected users stay in sync.
  useEffect(() => {
    if (!isLoaded || !currentCalendarId) return;

    const intervalMs = apiOnline ? 1000 : 5000;
    const intervalId = window.setInterval(() => {
      fetch(`/api/calendars/${currentCalendarId}`)
        .then((res) => {
          if (!res.ok) throw new Error('Failed to sync');
          return res.json();
        })
        .then((data) => {
          const parsed = parseCalendarResponse(data);
          setApiOnline(true);
          if (parsed.revision <= serverRevisionRef.current) return;
          serverRevisionRef.current = parsed.revision;
          baseStateRef.current = parsed.state;
          applyCalendarState(parsed.state);
        })
        .catch((err) => {
          if (apiOnline) {
            console.error('Calendar API is unreachable:', err);
          }
          setApiOnline(false);
        });
    }, intervalMs);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isLoaded, apiOnline, currentCalendarId, applyCalendarState]);

  // Persist to server on every state change
  useEffect(() => {
    if (!isLoaded || !apiOnline || !currentCalendarId) return;

    const state: CalendarState = {
      numDays,
      startDate: startDate.toISOString(),
      startHour,
      endHour,
      viewMode,
      events,
    };
    const saveSeq = saveSeqRef.current + 1;
    saveSeqRef.current = saveSeq;

    fetch(`/api/calendars/${currentCalendarId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        state,
        baseState: baseStateRef.current,
        baseRevision: serverRevisionRef.current,
      }),
    })
      .then((res) => {
        if (!res.ok) throw new Error('Failed to save');
        return res.json();
      })
      .then((data) => {
        if (saveSeq !== saveSeqRef.current) return;
        const parsed = parseCalendarResponse(data);
        setApiOnline(true);
        serverRevisionRef.current = parsed.revision;
        baseStateRef.current = parsed.state;
      })
      .catch((err) => {
        console.error('Failed to save:', err);
        setApiOnline(false);
      });
  }, [isLoaded, apiOnline, currentCalendarId, numDays, startDate, startHour, endHour, viewMode, events]);

  // Clamp dayViewIndex when numDays or viewMode changes
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
      [updated.dayIndex]: (prev[updated.dayIndex] || []).map((b) =>
        b.id === updated.id ? updated : b
      ),
    }));
  }, []);

  const handleBlockDeleted = useCallback((block: TimeBlock) => {
    setEvents((prev) => ({
      ...prev,
      [block.dayIndex]: (prev[block.dayIndex] || []).filter((b) => b.id !== block.id),
    }));
  }, []);

  const handleHoverEvent = useCallback((id: string | null) => {
    setHoveredEventId(id);
  }, []);

  const handlePrevDay = useCallback(() => {
    setDayViewIndex((prev) => Math.max(0, prev - 1));
  }, []);

  const handleNextDay = useCallback(() => {
    setDayViewIndex((prev) => Math.min(numDays - 1, prev + 1));
  }, [numDays]);

  const handleSelectDay = useCallback((index: number) => {
    setDayViewIndex(index);
  }, []);

  // ─── Calendar Manager handlers ───

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
      setCurrentCalendarId(null);
      setCurrentCalendarName('');
      applyCalendarState(DEFAULT_CALENDAR_STATE);
      setShowCalendarManager(true);
    } catch (err) {
      console.error('Failed to delete calendar:', err);
    }
  }, [currentCalendarId, currentCalendarName, applyCalendarState]);

  // In Day view, only pass events for the selected day to the map
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
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          alignItems: 'center',
          height: '100vh',
          color: 'var(--text-med)'
        }}>
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
            />
          </div>
        ) : (
          <MapView
            events={mapEvents}
            hoveredEventId={hoveredEventId}
            onHoverEvent={handleHoverEvent}
            preciseZoomEnabled={false}
          />
        )}
      </div>

      {/* Calendar Manager Modal */}
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
