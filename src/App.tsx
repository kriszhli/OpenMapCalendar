import { useState, useCallback, useEffect, useMemo, useRef } from 'react';
import type { LocationData, PreciseRouteCache, ViewMode, TimeBlock } from './types';
import { generateId, minutesToTime, PALETTE_COLORS } from './types';
import AppShell from './components/AppShell';
import TopControls from './components/TopControls';
import DayGrid from './components/DayGrid';
import DaySidebar from './components/DaySidebar';
import MapView from './components/MapView';
import CalendarManager from './components/CalendarManager';
import PlannerApprovalPanel, {
  type AiChatMessage,
  type AiPlanDraftEvent,
  type ClarificationChoice,
  type StagedAiPlan,
} from './components/AiPlannerChat';
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

type AiPlanEvent = AiPlanDraftEvent;

interface AiPlanResponse {
  status: 'ready' | 'needs_clarification';
  assistantMessage: string;
  events: AiPlanDraftEvent[];
  schedule_draft?: {
    status?: 'ready' | 'needs_clarification';
    assistantMessage?: string;
    events?: AiPlanDraftEvent[];
    blocks?: Record<string, never>[];
    clarification_options?: ClarificationChoice[];
    debugReasoning?: string;
  };
  clarification_options?: ClarificationChoice[];
  run_id?: string;
  calendar_id?: string;
  debugReasoning?: string;
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

interface RelativeDayHint {
  phrase: string;
  dayNumber: number;
  date: string;
}

interface RollbackSnapshot {
  events: Record<number, TimeBlock[]>;
  numDays: number;
  startDateIso: string;
  dayViewIndex: number;
}

interface ProposalEventDiff {
  index: number;
  before: AiPlanEvent;
  after: AiPlanEvent;
  changedFields: string[];
}

interface ManualCorrectionDiff {
  summary: string;
  event_diffs: ProposalEventDiff[];
  preference_updates: {
    category: string;
    normalized_value: string;
    summary: string;
    confidence: number;
    forced: boolean;
    priority: number;
  }[];
}

interface StagedProposalState extends StagedAiPlan {
  runId: string;
  calendarId: string;
  snapshot: RollbackSnapshot;
  originalEvents: AiPlanEvent[];
  showReasoning: boolean;
}

interface ApplyPlanResult {
  created: number;
  unresolvedPlaces: string[];
  skippedInvalid: number;
  skippedOverlap: number;
}

const DAY_MS = 24 * 60 * 60 * 1000;
const EARTH_RADIUS_KM = 6371;

const DEFAULT_AI_MESSAGES: AiChatMessage[] = [
  {
    id: 'ai-welcome',
    role: 'assistant',
    text: 'Tell me what you want planned, and I will draft a proposal for review.',
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

const toRadians = (deg: number): number => (deg * Math.PI) / 180;

const distanceKm = (a: { lat: number; lng: number }, b: { lat: number; lng: number }): number => {
  const dLat = toRadians(b.lat - a.lat);
  const dLng = toRadians(b.lng - a.lng);
  const lat1 = toRadians(a.lat);
  const lat2 = toRadians(b.lat);
  const sinDLat = Math.sin(dLat / 2);
  const sinDLng = Math.sin(dLng / 2);
  const h = sinDLat * sinDLat + Math.cos(lat1) * Math.cos(lat2) * sinDLng * sinDLng;
  return 2 * EARTH_RADIUS_KM * Math.atan2(Math.sqrt(h), Math.sqrt(1 - h));
};

const ORDINAL_DAY_MAP: Record<string, number> = {
  first: 1,
  second: 2,
  third: 3,
  fourth: 4,
  fifth: 5,
  sixth: 6,
  seventh: 7,
  eighth: 8,
  ninth: 9,
  tenth: 10,
  eleventh: 11,
  twelfth: 12,
  thirteenth: 13,
  fourteenth: 14,
  fifteenth: 15,
  sixteenth: 16,
  seventeenth: 17,
  eighteenth: 18,
  nineteenth: 19,
  twentieth: 20,
  twentyfirst: 21,
  twentysecond: 22,
  twentythird: 23,
  twentyfourth: 24,
  twentyfifth: 25,
  twentysixth: 26,
  twentyseventh: 27,
  twentyeighth: 28,
  twentyninth: 29,
  thirtieth: 30,
  thirtyfirst: 31,
};

const normalizeOrdinalToken = (token: string): string =>
  token.toLowerCase().replace(/[\s-]/g, '');

const extractRelativeDayHints = (
  text: string,
  calendarStartDate: Date,
  numDays: number
): RelativeDayHint[] => {
  const hints: RelativeDayHint[] = [];
  const seen = new Set<number>();

  const numericRegex = /\bday\s+(\d{1,2})\b/gi;
  let numericMatch: RegExpExecArray | null;
  while ((numericMatch = numericRegex.exec(text)) !== null) {
    const dayNumber = Number(numericMatch[1]);
    if (!Number.isFinite(dayNumber) || dayNumber < 1 || dayNumber > numDays || seen.has(dayNumber)) continue;
    seen.add(dayNumber);
    hints.push({
      phrase: numericMatch[0],
      dayNumber,
      date: toLocalIsoDate(addDays(calendarStartDate, dayNumber - 1)),
    });
  }

  const ordinalRegex = /\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|twentieth|twenty[\s-]?first|twenty[\s-]?second|twenty[\s-]?third|twenty[\s-]?fourth|twenty[\s-]?fifth|twenty[\s-]?sixth|twenty[\s-]?seventh|twenty[\s-]?eighth|twenty[\s-]?ninth|thirtieth|thirty[\s-]?first)\s+day\b/gi;
  let ordinalMatch: RegExpExecArray | null;
  while ((ordinalMatch = ordinalRegex.exec(text)) !== null) {
    const key = normalizeOrdinalToken(ordinalMatch[1]);
    const dayNumber = ORDINAL_DAY_MAP[key];
    if (!dayNumber || dayNumber > numDays || seen.has(dayNumber)) continue;
    seen.add(dayNumber);
    hints.push({
      phrase: ordinalMatch[0],
      dayNumber,
      date: toLocalIsoDate(addDays(calendarStartDate, dayNumber - 1)),
    });
  }

  return hints.sort((a, b) => a.dayNumber - b.dayNumber);
};

const extractCoordinateHints = (eventsByDay: Record<number, TimeBlock[]>): { lat: number; lng: number }[] => {
  const coords: { lat: number; lng: number }[] = [];
  for (const dayEvents of Object.values(eventsByDay)) {
    for (const event of dayEvents || []) {
      if (event.location) coords.push({ lat: event.location.lat, lng: event.location.lng });
      if (event.destination) coords.push({ lat: event.destination.lat, lng: event.destination.lng });
    }
  }
  return coords;
};

const centroidOfCoords = (coords: { lat: number; lng: number }[]): { lat: number; lng: number } | undefined => {
  if (!coords.length) return undefined;
  const sum = coords.reduce(
    (acc, curr) => ({ lat: acc.lat + curr.lat, lng: acc.lng + curr.lng }),
    { lat: 0, lng: 0 }
  );
  return { lat: sum.lat / coords.length, lng: sum.lng / coords.length };
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

const cloneAiPlanEvents = (events: AiPlanEvent[]): AiPlanEvent[] => events.map((event) => ({ ...event }));

const normalizeDraftEvent = (event: Partial<AiPlanEvent>): AiPlanEvent => ({
  title: event.title?.trim() || 'Untitled',
  description: event.description?.trim() || '',
  date: event.date?.trim() || '',
  startTime: event.startTime?.trim() || '09:00',
  endTime: event.endTime?.trim() || '10:00',
  origin: event.origin?.trim() || '',
  destination: event.destination?.trim() || '',
  color: normalizeHexColor(event.color),
});

const buildPreferenceUpdatesFromDiff = (original: AiPlanEvent, edited: AiPlanEvent) => {
  const updates: ManualCorrectionDiff['preference_updates'] = [];
  const originalStart = parseTimeToMinutes(original.startTime);
  const editedStart = parseTimeToMinutes(edited.startTime);
  const originalEnd = parseTimeToMinutes(original.endTime);
  const editedEnd = parseTimeToMinutes(edited.endTime);
  const originalWindowOverlapsLunch =
    originalStart !== null && originalEnd !== null && originalStart < 13 * 60 && originalEnd > 12 * 60;
  const editedWindowOverlapsLunch =
    editedStart !== null && editedEnd !== null && editedStart < 13 * 60 && editedEnd > 12 * 60;

  if (originalWindowOverlapsLunch && !editedWindowOverlapsLunch) {
    updates.push({
      category: 'lunch_avoidance',
      normalized_value: 'avoid lunch window',
      summary: 'User moved the proposal away from lunch.',
      confidence: 1,
      forced: true,
      priority: 100,
    });
  }

  if (originalStart !== null && editedStart !== null) {
    if (originalStart >= 12 * 60 && editedStart < 12 * 60) {
      updates.push({
        category: 'prefer_mornings',
        normalized_value: 'prefer mornings',
        summary: 'User moved the plan into the morning.',
        confidence: 0.98,
        forced: true,
        priority: 100,
      });
    } else if (originalStart < 12 * 60 && editedStart >= 12 * 60) {
      updates.push({
        category: 'prefer_afternoons',
        normalized_value: 'prefer afternoons',
        summary: 'User moved the plan into the afternoon.',
        confidence: 0.98,
        forced: true,
        priority: 100,
      });
    }
  }

  if (/focus|deep work|heads-down/i.test(`${original.title} ${edited.title}`)) {
    updates.push({
      category: 'focus_block',
      normalized_value: 'protected focus block',
      summary: 'User adjusted a focus block.',
      confidence: 0.95,
      forced: true,
      priority: 95,
    });
  }

  if (/commute|travel|route/i.test(`${original.title} ${edited.title} ${original.description} ${edited.description}`)) {
    updates.push({
      category: 'commute_sensitivity',
      normalized_value: 'travel buffer',
      summary: 'User corrected a travel buffer plan.',
      confidence: 0.9,
      forced: true,
      priority: 95,
    });
  }

  return updates;
};

const buildManualCorrectionDiff = (original: AiPlanEvent[], edited: AiPlanEvent[]): ManualCorrectionDiff => {
  const event_diffs: ProposalEventDiff[] = [];
  const preference_updates: ManualCorrectionDiff['preference_updates'] = [];
  const count = Math.min(original.length, edited.length);

  for (let index = 0; index < count; index += 1) {
    const before = normalizeDraftEvent(original[index]);
    const after = normalizeDraftEvent(edited[index]);
    const changedFields = (['title', 'description', 'date', 'startTime', 'endTime', 'origin', 'destination', 'color'] as const)
      .filter((field) => before[field] !== after[field]);

    if (!changedFields.length) continue;

    event_diffs.push({
      index,
      before,
      after,
      changedFields: [...changedFields],
    });

    preference_updates.push(...buildPreferenceUpdatesFromDiff(before, after));
  }

  const summary = event_diffs.length
    ? `${event_diffs.length} edited event${event_diffs.length === 1 ? '' : 's'}`
    : 'No user edits';

  return {
    summary,
    event_diffs,
    preference_updates: preference_updates.filter((item, index, array) =>
      array.findIndex((candidate) =>
        candidate.category === item.category && candidate.normalized_value === item.normalized_value
      ) === index
    ),
  };
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

async function geocodeLocationByName(
  name: string,
  focus?: { lat: number; lng: number }
): Promise<LocationData | undefined> {
  const query = name.trim();
  if (!query) return undefined;

  try {
    const params = new URLSearchParams({
      format: 'json',
      q: query,
      limit: '6',
    });
    const res = await fetch(
      `https://nominatim.openstreetmap.org/search?${params.toString()}`,
      { headers: { 'Accept-Language': 'en' } }
    );
    if (!res.ok) return undefined;
    const results = (await res.json()) as { display_name: string; lat: string; lon: string }[];
    if (!results.length) return undefined;

    const normalized = results
      .map((item) => ({ ...item, latNum: Number(item.lat), lngNum: Number(item.lon) }))
      .filter((item) => Number.isFinite(item.latNum) && Number.isFinite(item.lngNum));
    if (!normalized.length) return undefined;

    let best = normalized[0];
    if (focus) {
      best = normalized.reduce((currentBest, candidate) => {
        const currentDist = distanceKm(
          { lat: currentBest.latNum, lng: currentBest.lngNum },
          focus
        );
        const candidateDist = distanceKm(
          { lat: candidate.latNum, lng: candidate.lngNum },
          focus
        );
        return candidateDist < currentDist ? candidate : currentBest;
      }, normalized[0]);
    }

    return {
      name: best.display_name.split(',').slice(0, 2).join(',').trim() || query,
      lat: best.latNum,
      lng: best.lngNum,
    };
  } catch {
    return undefined;
  }
}

const buildPreviewEvents = async (
  plannedEvents: AiPlanEvent[],
  calendarStartDate: Date
): Promise<Record<number, TimeBlock[]>> => {
  const nextEvents: Record<number, TimeBlock[]> = {};
  const geocodeCache = new Map<string, Promise<LocationData | undefined>>();
  const resolvePlace = async (place?: string): Promise<LocationData | undefined> => {
    const key = place?.trim();
    if (!key) return undefined;
    if (!geocodeCache.has(key)) {
      geocodeCache.set(key, geocodeLocationByName(key));
    }
    return geocodeCache.get(key) ?? undefined;
  };

  for (const [index, item] of plannedEvents.entries()) {
    const date = parseIsoDateOnly(item.date);
    const startMinutes = parseTimeToMinutes(item.startTime);
    const endMinutes = parseTimeToMinutes(item.endTime);
    if (!date || startMinutes === null || endMinutes === null) continue;

    const dayIndex = diffInDays(date, toLocalDay(calendarStartDate));
    const [location, destination] = await Promise.all([
      resolvePlace(item.origin),
      resolvePlace(item.destination),
    ]);

    const block: TimeBlock = {
      id: `preview-${index}-${generateId()}`,
      dayIndex,
      startMinutes,
      endMinutes,
      color: normalizeHexColor(item.color) || PALETTE_COLORS[index % PALETTE_COLORS.length],
      title: item.title?.trim() || 'Untitled',
      description: item.description?.trim() || '',
      location,
      destination,
      routeMode: 'simple',
    };

    nextEvents[dayIndex] = [...(nextEvents[dayIndex] || []), block].sort(
      (a, b) => a.startMinutes - b.startMinutes
    );
  }

  return nextEvents;
};

const clarificationChoiceToEvent = (choice: ClarificationChoice): AiPlanEvent => ({
  title: choice.label || 'Selected option',
  description: choice.description || '',
  date: choice.date,
  startTime: choice.start_time,
  endTime: choice.end_time,
});

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
  const [stagedPlanByCalendar, setStagedPlanByCalendar] = useState<Record<string, StagedProposalState | null>>({});
  const [previewEventsByCalendar, setPreviewEventsByCalendar] = useState<Record<string, Record<number, TimeBlock[]> | null>>({});
  const [aiLoading, setAiLoading] = useState(false);
  const [aiCommitting, setAiCommitting] = useState(false);
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
    return !!aiRollbackByCalendar[currentCalendarId] || !!stagedPlanByCalendar[currentCalendarId];
  }, [currentCalendarId, aiRollbackByCalendar, stagedPlanByCalendar]);

  const currentStagedPlan = useMemo(() => {
    if (!currentCalendarId) return null;
    return stagedPlanByCalendar[currentCalendarId] ?? null;
  }, [currentCalendarId, stagedPlanByCalendar]);

  const currentPreviewEvents = useMemo(() => {
    if (!currentCalendarId) return null;
    return previewEventsByCalendar[currentCalendarId] ?? null;
  }, [currentCalendarId, previewEventsByCalendar]);

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
    setStagedPlanByCalendar((prev) => {
      if (!(calendarId in prev)) return prev;
      const next = { ...prev };
      delete next[calendarId];
      return next;
    });
    setPreviewEventsByCalendar((prev) => {
      if (!(calendarId in prev)) return prev;
      const next = { ...prev };
      delete next[calendarId];
      return next;
    });
  }, []);

  const clearStagedProposal = useCallback((calendarId: string) => {
    setStagedPlanByCalendar((prev) => ({ ...prev, [calendarId]: null }));
    setPreviewEventsByCalendar((prev) => ({ ...prev, [calendarId]: null }));
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
      const globalFocus = centroidOfCoords(extractCoordinateHints(snapshot.events));

      const geocodeCache = new Map<string, Promise<LocationData | undefined>>();
      const unresolvedPlaces = new Set<string>();
      const resolvePlace = async (
        place?: string,
        focus?: { lat: number; lng: number }
      ): Promise<LocationData | undefined> => {
        const key = place?.trim();
        if (!key) return undefined;
        const focusKey = focus ? `|${focus.lat.toFixed(3)},${focus.lng.toFixed(3)}` : '';
        const cacheKey = `${key}${focusKey}`;
        if (!geocodeCache.has(cacheKey)) {
          geocodeCache.set(cacheKey, geocodeLocationByName(key, focus));
        }
        const resolved = await geocodeCache.get(cacheKey);
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

        const dayCoords = extractCoordinateHints({ [dayIndex]: dayEvents });
        const focus = centroidOfCoords(dayCoords) ?? globalFocus;
        const [location, destination] = await Promise.all([
          resolvePlace(event.origin, focus),
          resolvePlace(event.destination, focus),
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
      const relativeDayHints = extractRelativeDayHints(text, startDate, numDays);

      try {
        const response = await fetch('/api/ai/plan-events', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            calendarId: requestCalendarId,
            showReasoning: true,
            messages: requestMessages.map((message) => ({ role: message.role, text: message.text })),
            context: {
              calendarStartDate: calendarStartLocal,
              calendarEndDate: calendarEndLocal,
              visibleDays: numDays,
              dayStartHour: startHour,
              dayEndHour: endHour,
              timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
              existingEvents,
              relativeDayHints,
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
          'Please add a little more detail so I can build a safe proposal.';

        if (currentCalendarIdRef.current !== requestCalendarId) {
          appendAiAssistantMessage(requestCalendarId, 'You switched calendars before this request completed.', 'error');
          return;
        }

        const responseEvents = Array.isArray(payload.events) ? payload.events : [];
        const clarificationOptions =
          (Array.isArray(payload.clarification_options) && payload.clarification_options) ||
          (payload.schedule_draft && Array.isArray(payload.schedule_draft.clarification_options)
            ? payload.schedule_draft.clarification_options
            : []);
        const reasoning = payload.debugReasoning || payload.schedule_draft?.debugReasoning || '';

        if (payload.status === 'ready') {
          const staged: StagedProposalState = {
            status: 'ready',
            assistantMessage,
            events: cloneAiPlanEvents(responseEvents),
            clarificationOptions: [],
            reasoning,
            selectedClarificationOptionIndex: null,
            isEditing: false,
            runId: payload.run_id || '',
            calendarId: requestCalendarId,
            snapshot,
            originalEvents: cloneAiPlanEvents(responseEvents),
            showReasoning: false,
          };
          setStagedPlanByCalendar((prev) => ({ ...prev, [requestCalendarId]: staged }));
          appendAiAssistantMessage(requestCalendarId, 'Proposal ready for approval.', 'success');
        } else {
          const staged: StagedProposalState = {
            status: 'needs_clarification',
            assistantMessage,
            events: [],
            clarificationOptions,
            reasoning,
            selectedClarificationOptionIndex: null,
            isEditing: false,
            runId: payload.run_id || '',
            calendarId: requestCalendarId,
            snapshot,
            originalEvents: [],
            showReasoning: false,
          };
          setStagedPlanByCalendar((prev) => ({ ...prev, [requestCalendarId]: staged }));
          appendAiAssistantMessage(requestCalendarId, assistantMessage);
        }
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Unable to contact the planner service.';
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
    ]
  );

  const handleToggleProposalEdit = useCallback(() => {
    if (!currentCalendarId) return;
    setStagedPlanByCalendar((prev) => {
      const current = prev[currentCalendarId];
      if (!current) return prev;
      return {
        ...prev,
        [currentCalendarId]: {
          ...current,
          isEditing: !current.isEditing,
        },
      };
    });
  }, [currentCalendarId]);

  const handleUpdateProposalEvent = useCallback(
    (index: number, patch: Partial<AiPlanEvent>) => {
      if (!currentCalendarId) return;
      setStagedPlanByCalendar((prev) => {
        const current = prev[currentCalendarId];
        if (!current) return prev;
        const nextEvents = cloneAiPlanEvents(current.events);
        if (!nextEvents[index]) return prev;
        nextEvents[index] = normalizeDraftEvent({ ...nextEvents[index], ...patch });
        return {
          ...prev,
          [currentCalendarId]: {
            ...current,
            events: nextEvents,
            isEditing: true,
          },
        };
      });
    },
    [currentCalendarId]
  );

  const handleSelectClarificationOption = useCallback(
    (index: number) => {
      if (!currentCalendarId) return;
      setStagedPlanByCalendar((prev) => {
        const current = prev[currentCalendarId];
        if (!current || current.status !== 'needs_clarification') return prev;
        const choice = current.clarificationOptions[index];
        if (!choice) return prev;
        const selectedEvent = clarificationChoiceToEvent(choice);
        return {
          ...prev,
          [currentCalendarId]: {
            ...current,
            status: 'ready',
            events: [selectedEvent],
            originalEvents: [selectedEvent],
            selectedClarificationOptionIndex: index,
            assistantMessage: `Selected ${choice.label}. Review it and approve when ready.`,
            isEditing: false,
          },
        };
      });
    },
    [currentCalendarId]
  );

  const handleToggleReasoning = useCallback(() => {
    if (!currentCalendarId) return;
    setStagedPlanByCalendar((prev) => {
      const current = prev[currentCalendarId];
      if (!current) return prev;
      return {
        ...prev,
        [currentCalendarId]: {
          ...current,
          showReasoning: !current.showReasoning,
        },
      };
    });
  }, [currentCalendarId]);

  const handleCommitAiProposal = useCallback(async () => {
    if (!currentCalendarId || !currentStagedPlan || aiCommitting) return;

    const finalEvents = cloneAiPlanEvents(currentStagedPlan.events);
    const correctionDiff = buildManualCorrectionDiff(currentStagedPlan.originalEvents, finalEvents);
    const confirmedAt = new Date().toISOString();
    const confirmedSchedule = {
      status: currentStagedPlan.status,
      assistantMessage: currentStagedPlan.assistantMessage,
      events: finalEvents,
      run_id: currentStagedPlan.runId,
      calendar_id: currentCalendarId,
      selected_clarification_option_index: currentStagedPlan.selectedClarificationOptionIndex,
    };

    setAiCommitting(true);
    try {
      try {
        await fetch('/api/ai/confirm-plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            calendarId: currentCalendarId,
            calendar_id: currentCalendarId,
            run_id: currentStagedPlan.runId,
            user_confirmed: true,
            confirmed_at: confirmedAt,
            confirmed_schedule: confirmedSchedule,
            schedule_draft: {
              status: currentStagedPlan.status,
              assistantMessage: currentStagedPlan.assistantMessage,
              events: currentStagedPlan.events,
              clarificationOptions: currentStagedPlan.clarificationOptions,
              reasoning: currentStagedPlan.reasoning,
              selectedClarificationOptionIndex: currentStagedPlan.selectedClarificationOptionIndex,
            },
            correction_diff: correctionDiff,
            preference_updates: correctionDiff.preference_updates,
          }),
        });
      } catch (err) {
        console.error('Failed to confirm plan with planner service:', err);
      }

      const result = await applyPlannedEvents(finalEvents, currentStagedPlan.snapshot);
      if (result.created > 0) {
        setAiRollbackByCalendar((prev) => ({ ...prev, [currentCalendarId]: currentStagedPlan.snapshot }));
      }

      clearStagedProposal(currentCalendarId);
      const compactParts = [`Committed ${result.created} event${result.created === 1 ? '' : 's'}.`];
      if (correctionDiff.event_diffs.length > 0) {
        compactParts.push(`Captured ${correctionDiff.event_diffs.length} edit${correctionDiff.event_diffs.length === 1 ? '' : 's'}.`);
      }
      if (correctionDiff.preference_updates.length > 0) {
        compactParts.push('Updated local memory from the correction.');
      }
      appendAiAssistantMessage(currentCalendarId, compactParts.join(' '), 'success');
    } finally {
      setAiCommitting(false);
    }
  }, [currentCalendarId, currentStagedPlan, aiCommitting, applyPlannedEvents, appendAiAssistantMessage, clearStagedProposal]);

  const handleAiRollback = useCallback(() => {
    if (!currentCalendarId) return;

    if (stagedPlanByCalendar[currentCalendarId]) {
      clearStagedProposal(currentCalendarId);
      appendAiAssistantMessage(currentCalendarId, 'Cleared the staged proposal.', 'success');
      return;
    }

    const snapshot = aiRollbackByCalendar[currentCalendarId];
    if (!snapshot) return;

    setEvents(cloneEvents(snapshot.events));
    setNumDays(snapshot.numDays);
    setStartDate(new Date(snapshot.startDateIso));
    setDayViewIndex(snapshot.dayViewIndex);

    setAiRollbackByCalendar((prev) => ({ ...prev, [currentCalendarId]: null }));
    appendAiAssistantMessage(currentCalendarId, 'Rolled back the last proposal batch.', 'success');
  }, [currentCalendarId, aiRollbackByCalendar, stagedPlanByCalendar, clearStagedProposal, appendAiAssistantMessage]);

  useEffect(() => {
    if (!currentCalendarId) return;
    if (!currentStagedPlan) {
      setPreviewEventsByCalendar((prev) => ({ ...prev, [currentCalendarId]: null }));
      return;
    }

    let cancelled = false;
    buildPreviewEvents(currentStagedPlan.events, startDate).then((preview) => {
      if (cancelled) return;
      setPreviewEventsByCalendar((prev) => ({ ...prev, [currentCalendarId]: preview }));
    });

    return () => {
      cancelled = true;
    };
  }, [currentCalendarId, currentStagedPlan, startDate]);

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
              previewEvents={currentPreviewEvents || undefined}
              hoveredEventId={hoveredEventId}
              onHoverEvent={handleHoverEvent}
              preciseZoomEnabled
              onPreciseRouteCacheChange={handlePreciseRouteCacheChange}
            />
          </div>
        ) : (
          <MapView
            events={mapEvents}
            previewEvents={currentPreviewEvents || undefined}
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

      <PlannerApprovalPanel
        messages={currentAiMessages}
        loading={aiLoading}
        onSendMessage={handleAiMessage}
        disabled={!currentCalendarId}
        disabledHint="Select or create a calendar first, then ask me to plan your schedule."
        canRollback={canRollbackAiChanges}
        onRollback={handleAiRollback}
        stagedPlan={currentStagedPlan}
        onCommit={handleCommitAiProposal}
        onToggleProposalEdit={handleToggleProposalEdit}
        onUpdateProposalEvent={handleUpdateProposalEvent}
        onSelectClarificationOption={handleSelectClarificationOption}
        onToggleReasoning={handleToggleReasoning}
        showReasoning={!!currentStagedPlan?.showReasoning}
        commitDisabled={
          aiCommitting ||
          !currentStagedPlan ||
          (currentStagedPlan.status === 'ready' && currentStagedPlan.events.length === 0) ||
          (currentStagedPlan.status === 'needs_clarification' &&
            currentStagedPlan.selectedClarificationOptionIndex === null &&
            currentStagedPlan.events.length === 0)
        }
        commitDisabledHint="Choose a clarification option or draft a plan first."
      />
    </AppShell>
  );
}

export default App;
