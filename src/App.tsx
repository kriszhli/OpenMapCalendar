import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type { LocationData, PreciseRouteCache, ViewMode, TimeBlock } from './types';
import { generateId, minutesToTime, PALETTE_COLORS } from './types';
import AppShell from './components/AppShell';
import TopControls from './components/TopControls';
import DayGrid from './components/DayGrid';
import DaySidebar from './components/DaySidebar';
import MapView from './components/MapView';
import CalendarManager from './components/CalendarManager';
import AiPlannerChat, { type AiChatMessage } from './components/AiPlannerChat';
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

interface AiPlanEvent {
  title: string;
  description?: string;
  date: string;
  startTime: string;
  endTime: string;
  origin?: string;
  destination?: string;
  color?: string;
}

interface AiPlanResponse {
  status: 'ready' | 'needs_clarification';
  assistantMessage: string;
  events: AiPlanEvent[];
  error?: string;
  detail?: string;
}

interface ParsedPlanEvent {
  title: string;
  description: string;
  date: Date;
  dayDiff: number;
  startMinutes: number;
  endMinutes: number;
  origin?: string;
  destination?: string;
  color?: string;
}

interface CalendarAiContextEvent {
  dayIndex: number;
  date: string;
  startTime: string;
  endTime: string;
  title: string;
  origin?: string;
  destination?: string;
}

interface RollbackSnapshot {
  events: Record<number, TimeBlock[]>;
  numDays: number;
  startDateIso: string;
  dayViewIndex: number;
}

interface ApplyPlanResult {
  created: number;
  unresolvedPlaces: string[];
  skippedInvalid: number;
  skippedOverlap: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;

const DEFAULT_AI_MESSAGES: AiChatMessage[] = [
  {
    id: 'ai-welcome',
    role: 'assistant',
    text: 'Tell me where and when you want to go, and I will create calendar events for you.',
  },
];

const toLocalDay = (value: Date): Date =>
  new Date(value.getFullYear(), value.getMonth(), value.getDate());

const toLocalIsoDate = (value: Date): string => {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, '0');
  const day = String(value.getDate()).padStart(2, '0');
  return `${year}-${month}-${day}`;
};

const addDays = (value: Date, days: number): Date => {
  const next = new Date(value);
  next.setDate(next.getDate() + days);
  return next;
};

const diffInDays = (left: Date, right: Date): number =>
  Math.round((toLocalDay(left).getTime() - toLocalDay(right).getTime()) / DAY_MS);

const parseIsoDateOnly = (value: string): Date | null => {
  const match = value.trim().match(/^(\d{4})-(\d{2})-(\d{2})$/);
  if (!match) return null;
  const year = Number(match[1]);
  const month = Number(match[2]);
  const day = Number(match[3]);
  const parsed = new Date(year, month - 1, day);
  if (
    parsed.getFullYear() !== year ||
    parsed.getMonth() !== month - 1 ||
    parsed.getDate() !== day
  ) {
    return null;
  }
  return parsed;
};

const parseTimeToMinutes = (value: string): number | null => {
  const match = value.trim().match(/^([01]\d|2[0-3]):([0-5]\d)$/);
  if (!match) return null;
  return Number(match[1]) * 60 + Number(match[2]);
};

const normalizeHexColor = (value?: string): string | undefined => {
  if (!value) return undefined;
  const trimmed = value.trim();
  return /^#[0-9a-fA-F]{6}$/.test(trimmed) ? trimmed : undefined;
};

const cloneEvents = (source: Record<number, TimeBlock[]>): Record<number, TimeBlock[]> => {
  const clone: Record<number, TimeBlock[]> = {};
  for (const [dayKey, dayEvents] of Object.entries(source)) {
    clone[Number(dayKey)] = dayEvents.map((event) => ({ ...event }));
  }
  return clone;
};

const hasTimeOverlap = (aStart: number, aEnd: number, bStart: number, bEnd: number): boolean =>
  aStart < bEnd && aEnd > bStart;

const serializeCalendarEventsForAi = (
  eventsByDay: Record<number, TimeBlock[]>,
  calendarStartDate: Date
): CalendarAiContextEvent[] => {
  const rows: CalendarAiContextEvent[] = [];
  const dayIndexes = Object.keys(eventsByDay)
    .map((value) => Number(value))
    .filter((value) => Number.isFinite(value))
    .sort((a, b) => a - b);

  for (const dayIndex of dayIndexes) {
    const dayDate = addDays(calendarStartDate, dayIndex);
    const date = toLocalIsoDate(dayDate);
    const sortedEvents = [...(eventsByDay[dayIndex] || [])].sort(
      (a, b) => a.startMinutes - b.startMinutes
    );

    for (const event of sortedEvents) {
      rows.push({
        dayIndex,
        date,
        startTime: minutesToTime(event.startMinutes),
        endTime: minutesToTime(event.endMinutes),
        title: (event.title || 'Untitled').trim(),
        origin: event.location?.name,
        destination: event.destination?.name,
      });
    }
  }

  return rows;
};

function parseCalendarResponse(raw: unknown): CalendarResponse {
  const data = raw as Partial<CalendarState> & { state?: Partial<CalendarState>; revision?: number };
  const source = data.state ?? data;
  const parsedDate = typeof source.startDate === 'string' ? new Date(source.startDate) : null;
  const parsedNumDays = Number(source.numDays);
  const parsedStartHour = Number(source.startHour);
  const parsedEndHour = Number(source.endHour);

  const state: CalendarState = {
    numDays: Number.isFinite(parsedNumDays) && parsedNumDays > 0 ? parsedNumDays : DEFAULT_CALENDAR_STATE.numDays,
    startDate:
      parsedDate && !Number.isNaN(parsedDate.getTime())
        ? parsedDate.toISOString()
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

async function geocodeLocationByName(name: string): Promise<LocationData | undefined> {
  const query = name.trim();
  if (!query) return undefined;

  try {
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=1`,
      { headers: { 'Accept-Language': 'en' } }
    );
    if (!res.ok) return undefined;
    const results = (await res.json()) as { display_name: string; lat: string; lon: string }[];
    const first = results[0];
    if (!first) return undefined;

    const lat = Number(first.lat);
    const lng = Number(first.lon);
    if (!Number.isFinite(lat) || !Number.isFinite(lng)) return undefined;

    return {
      name: first.display_name.split(',').slice(0, 2).join(',').trim() || query,
      lat,
      lng,
    };
  } catch {
    return undefined;
  }
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

  // AI state (isolated per calendar)
  const [aiMessagesByCalendar, setAiMessagesByCalendar] = useState<Record<string, AiChatMessage[]>>({});
  const [aiRollbackByCalendar, setAiRollbackByCalendar] = useState<Record<string, RollbackSnapshot | null>>({});
  const [aiLoading, setAiLoading] = useState(false);
  const currentCalendarIdRef = useRef<string | null>(null);

  useEffect(() => {
    currentCalendarIdRef.current = currentCalendarId;
  }, [currentCalendarId]);

  const currentAiMessages = useMemo(() => {
    if (!currentCalendarId) return DEFAULT_AI_MESSAGES;
    return aiMessagesByCalendar[currentCalendarId] ?? DEFAULT_AI_MESSAGES;
  }, [currentCalendarId, aiMessagesByCalendar]);

  const canRollbackAiChanges = useMemo(() => {
    if (!currentCalendarId) return false;
    return !!aiRollbackByCalendar[currentCalendarId];
  }, [currentCalendarId, aiRollbackByCalendar]);

  const applyCalendarState = useCallback((state: CalendarState) => {
    setNumDays(state.numDays);
    setStartDate(new Date(state.startDate));
    setStartHour(state.startHour);
    setEndHour(state.endHour);
    setViewMode(state.viewMode);
    setEvents(state.events);
  }, []);

  const appendAiMessage = useCallback((calendarId: string, message: AiChatMessage) => {
    setAiMessagesByCalendar((prev) => {
      const current = prev[calendarId] ?? DEFAULT_AI_MESSAGES;
      return {
        ...prev,
        [calendarId]: [...current, message],
      };
    });
  }, []);

  const appendAiAssistantMessage = useCallback(
    (
      calendarId: string,
      text: string,
      variant: AiChatMessage['variant'] = 'default'
    ) => {
      const trimmed = text.trim();
      if (!trimmed) return;
      appendAiMessage(calendarId, {
        id: generateId(),
        role: 'assistant',
        text: trimmed,
        variant,
      });
    },
    [appendAiMessage]
  );

  const clearAiStateForCalendar = useCallback((calendarId: string) => {
    setAiMessagesByCalendar((prev) => {
      if (!(calendarId in prev)) return prev;
      const next = { ...prev };
      delete next[calendarId];
      return next;
    });
    setAiRollbackByCalendar((prev) => {
      if (!(calendarId in prev)) return prev;
      const next = { ...prev };
      delete next[calendarId];
      return next;
    });
  }, []);

  const applyPlannedEvents = useCallback(
    async (
      plannedEvents: AiPlanEvent[],
      snapshot: RollbackSnapshot
    ): Promise<ApplyPlanResult> => {
      const snapshotStartDate = new Date(snapshot.startDateIso);
      const currentStartDay = toLocalDay(snapshotStartDate);
      const parsed: ParsedPlanEvent[] = [];
      let skippedInvalid = 0;

      for (const item of plannedEvents) {
        const date = parseIsoDateOnly(item.date);
        const startMinutesRaw = parseTimeToMinutes(item.startTime);
        const endMinutesRaw = parseTimeToMinutes(item.endTime);
        if (!date || startMinutesRaw === null || endMinutesRaw === null) {
          skippedInvalid += 1;
          continue;
        }

        const startMinutes = Math.max(0, Math.min(23 * 60 + 30, startMinutesRaw));
        const endMinutes = Math.max(
          startMinutes + 30,
          Math.min(24 * 60, endMinutesRaw > startMinutes ? endMinutesRaw : startMinutes + 60)
        );

        parsed.push({
          title: item.title?.trim() || 'Untitled',
          description: item.description?.trim() || '',
          date,
          dayDiff: diffInDays(date, currentStartDay),
          startMinutes,
          endMinutes,
          origin: item.origin?.trim(),
          destination: item.destination?.trim(),
          color: normalizeHexColor(item.color),
        });
      }

      if (parsed.length === 0) {
        return { created: 0, unresolvedPlaces: [], skippedInvalid, skippedOverlap: 0 };
      }

      parsed.sort((a, b) => (a.dayDiff - b.dayDiff) || (a.startMinutes - b.startMinutes));

      const minDayDiff = Math.min(...parsed.map((event) => event.dayDiff));
      const offset = minDayDiff < 0 ? -minDayDiff : 0;
      const shiftedStartDate = addDays(snapshotStartDate, -offset);

      const geocodeCache = new Map<string, Promise<LocationData | undefined>>();
      const unresolvedPlaces = new Set<string>();
      const resolvePlace = async (place?: string): Promise<LocationData | undefined> => {
        const key = place?.trim();
        if (!key) return undefined;
        if (!geocodeCache.has(key)) {
          geocodeCache.set(key, geocodeLocationByName(key));
        }
        const resolved = await geocodeCache.get(key);
        if (!resolved) unresolvedPlaces.add(key);
        return resolved;
      };

      const shiftedEvents: Record<number, TimeBlock[]> = {};
      for (const [dayKey, dayEvents] of Object.entries(snapshot.events)) {
        const oldDayIndex = Number(dayKey);
        if (!Number.isFinite(oldDayIndex)) continue;
        const nextDayIndex = oldDayIndex + offset;
        shiftedEvents[nextDayIndex] = dayEvents.map((event) => ({
          ...event,
          dayIndex: nextDayIndex,
        }));
      }

      const nextEvents = cloneEvents(shiftedEvents);
      let created = 0;
      let skippedOverlap = 0;
      let colorIdx = 0;

      for (const event of parsed) {
        const dayIndex = event.dayDiff + offset;
        const dayEvents = [...(nextEvents[dayIndex] || [])];
        const overlapsExisting = dayEvents.some((existing) =>
          hasTimeOverlap(event.startMinutes, event.endMinutes, existing.startMinutes, existing.endMinutes)
        );

        if (overlapsExisting) {
          skippedOverlap += 1;
          continue;
        }

        const [location, destination] = await Promise.all([
          resolvePlace(event.origin),
          resolvePlace(event.destination),
        ]);

        const block: TimeBlock = {
          id: generateId(),
          dayIndex,
          startMinutes: event.startMinutes,
          endMinutes: event.endMinutes,
          color: event.color || PALETTE_COLORS[colorIdx % PALETTE_COLORS.length],
          title: event.title,
          description: event.description,
          location,
          destination,
          routeMode: 'simple',
        };

        dayEvents.push(block);
        dayEvents.sort((a, b) => a.startMinutes - b.startMinutes);
        nextEvents[dayIndex] = dayEvents;
        created += 1;
        colorIdx += 1;
      }

      if (created === 0) {
        return {
          created: 0,
          unresolvedPlaces: [...unresolvedPlaces],
          skippedInvalid,
          skippedOverlap,
        };
      }

      const maxIndex = Object.keys(nextEvents).reduce((max, key) => {
        const idx = Number(key);
        return Number.isFinite(idx) ? Math.max(max, idx) : max;
      }, -1);
      const nextNumDays = Math.max(snapshot.numDays + offset, maxIndex + 1);

      setStartDate(shiftedStartDate);
      setDayViewIndex(snapshot.dayViewIndex + offset);
      setNumDays(nextNumDays);
      setEvents(nextEvents);

      return {
        created,
        unresolvedPlaces: [...unresolvedPlaces],
        skippedInvalid,
        skippedOverlap,
      };
    },
    []
  );

  const handleAiMessage = useCallback(
    async (text: string) => {
      if (!currentCalendarId || aiLoading) return;

      const requestCalendarId = currentCalendarId;
      const userMessage: AiChatMessage = { id: generateId(), role: 'user', text };
      appendAiMessage(requestCalendarId, userMessage);
      setAiLoading(true);

      const requestMessages = [
        ...(aiMessagesByCalendar[requestCalendarId] ?? DEFAULT_AI_MESSAGES),
        userMessage,
      ];

      const snapshot: RollbackSnapshot = {
        events: cloneEvents(events),
        numDays,
        startDateIso: startDate.toISOString(),
        dayViewIndex,
      };

      const calendarStartLocal = toLocalIsoDate(startDate);
      const calendarEndLocal = toLocalIsoDate(addDays(startDate, numDays - 1));
      const existingEvents = serializeCalendarEventsForAi(events, startDate);

      try {
        const response = await fetch('/api/ai/plan-events', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            messages: requestMessages.map((message) => ({ role: message.role, text: message.text })),
            context: {
              calendarStartDate: calendarStartLocal,
              calendarEndDate: calendarEndLocal,
              visibleDays: numDays,
              dayStartHour: startHour,
              dayEndHour: endHour,
              timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
              existingEvents,
            },
          }),
        });

        const payload = (await response.json()) as AiPlanResponse;

        if (!response.ok) {
          const detail = [payload.error, payload.detail].filter(Boolean).join(' ');
          throw new Error(detail || 'The AI planner request failed.');
        }

        const assistantMessage =
          payload.assistantMessage?.trim() ||
          'Please provide a little more detail so I can schedule this accurately.';

        if (currentCalendarIdRef.current !== requestCalendarId) {
          appendAiAssistantMessage(
            requestCalendarId,
            `${assistantMessage}\nYou switched calendars before I could apply the result. Re-open this calendar and run it again.`,
            'error'
          );
          return;
        }

        if (payload.status === 'ready' && Array.isArray(payload.events) && payload.events.length > 0) {
          const result = await applyPlannedEvents(payload.events, snapshot);

          const lines = [assistantMessage];
          lines.push(`Added ${result.created} event${result.created === 1 ? '' : 's'} to your calendar.`);

          if (result.skippedOverlap > 0) {
            lines.push(
              `Skipped ${result.skippedOverlap} event${result.skippedOverlap === 1 ? '' : 's'} because they overlap existing events.`
            );
          }

          if (result.skippedInvalid > 0) {
            lines.push(
              `Skipped ${result.skippedInvalid} item${result.skippedInvalid === 1 ? '' : 's'} due to invalid or missing date/time.`
            );
          }

          if (result.unresolvedPlaces.length > 0) {
            lines.push(
              `I could not resolve these locations automatically: ${result.unresolvedPlaces.join(', ')}`
            );
          }

          if (result.created > 0) {
            setAiRollbackByCalendar((prev) => ({ ...prev, [requestCalendarId]: snapshot }));
            lines.push('Use Rollback to undo this AI-generated batch.');
          }

          appendAiAssistantMessage(
            requestCalendarId,
            lines.join('\n'),
            result.created > 0 ? 'success' : 'error'
          );
        } else {
          appendAiAssistantMessage(requestCalendarId, assistantMessage);
        }
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Unable to contact the AI planner. Make sure Ollama is running locally.';
        appendAiAssistantMessage(requestCalendarId, message, 'error');
      } finally {
        setAiLoading(false);
      }
    },
    [
      currentCalendarId,
      aiLoading,
      aiMessagesByCalendar,
      events,
      numDays,
      startDate,
      dayViewIndex,
      startHour,
      endHour,
      appendAiMessage,
      appendAiAssistantMessage,
      applyPlannedEvents,
    ]
  );

  const handleAiRollback = useCallback(() => {
    if (!currentCalendarId) return;

    const snapshot = aiRollbackByCalendar[currentCalendarId];
    if (!snapshot) return;

    setEvents(cloneEvents(snapshot.events));
    setNumDays(snapshot.numDays);
    setStartDate(new Date(snapshot.startDateIso));
    setDayViewIndex(snapshot.dayViewIndex);

    setAiRollbackByCalendar((prev) => ({ ...prev, [currentCalendarId]: null }));
    appendAiAssistantMessage(currentCalendarId, 'Rolled back the last AI-generated event batch.', 'success');
  }, [currentCalendarId, aiRollbackByCalendar, appendAiAssistantMessage]);

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
      clearAiStateForCalendar(id);
      if (id === currentCalendarId) {
        setCurrentCalendarId(null);
        setCurrentCalendarName('');
        applyCalendarState(DEFAULT_CALENDAR_STATE);
      }
    } catch (err) {
      console.error('Failed to delete calendar:', err);
    }
  }, [currentCalendarId, applyCalendarState, clearAiStateForCalendar]);

  const handleDeleteCurrentCalendar = useCallback(async () => {
    if (!currentCalendarId) return;
    const confirmed = window.confirm(`Delete "${currentCalendarName}"? This action cannot be undone.`);
    if (!confirmed) return;

    try {
      await fetch(`/api/calendars/${currentCalendarId}`, { method: 'DELETE' });
      clearAiStateForCalendar(currentCalendarId);
      setCurrentCalendarId(null);
      setCurrentCalendarName('');
      applyCalendarState(DEFAULT_CALENDAR_STATE);
      setShowCalendarManager(true);
    } catch (err) {
      console.error('Failed to delete calendar:', err);
    }
  }, [currentCalendarId, currentCalendarName, applyCalendarState, clearAiStateForCalendar]);

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

      <AiPlannerChat
        messages={currentAiMessages}
        loading={aiLoading}
        onSendMessage={handleAiMessage}
        disabled={!currentCalendarId}
        disabledHint="Select or create a calendar first, then ask me to schedule your travel plans."
        canRollback={canRollbackAiChanges}
        onRollback={handleAiRollback}
      />
    </AppShell>
  );
}

export default App;
