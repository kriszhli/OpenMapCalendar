export type ViewMode = 'row' | 'grid' | 'day';

export interface LocationData {
    name: string;
    lat: number;
    lng: number;
}

export interface PreciseRouteCache {
    from: LocationData;
    to: LocationData;
    coords: [number, number][];
}

export const modeLabels = {
  walking: 'Walk',
  driving: 'Drive',
  transit: 'Public transit',
  bicycling: 'Cycle',
} as const;

export interface TimeBlock {
    id: string;
    dayIndex: number;
    startMinutes: number;
    endMinutes: number;
    color: string;
    title: string;
    description: string;
    location?: LocationData;      // start / origin
    destination?: LocationData;   // end / destination (route planning)
    transport?: keyof typeof modeLabels;
    routeMode?: 'simple' | 'precise' | 'hidden'; // default 'simple' (bezier curve)
    preciseRouteCache?: PreciseRouteCache;
}

export const PALETTE_COLORS = [
    '#5B7FBF', // muted blue
    '#7BA887', // sage green
    '#C4A05A', // soft amber
    '#9B8EC4', // lavender
    '#C47B6B', // coral
] as const;

export const SNAP_INTERVAL = 30; // minutes

/**
 * Vertical inset (px) applied at the top and bottom of the scheduling area.
 * Ensures the first and last time labels have room and aren't clipped.
 * All minute↔px conversions share this constant for pixel-perfect alignment.
 */
const VERTICAL_INSET = 12;

export function snapToInterval(minutes: number, interval: number = SNAP_INTERVAL): number {
    return Math.round(minutes / interval) * interval;
}

/**
 * Convert a minute value to a pixel Y position within a container.
 * Uses VERTICAL_INSET so the first label starts below 0 and the last label ends above containerHeight.
 */
export function minutesToPx(minutes: number, startHour: number, endHour: number, containerHeight: number): number {
    const totalMinutes = (endHour - startHour) * 60;
    const usable = containerHeight - 2 * VERTICAL_INSET;
    return VERTICAL_INSET + ((minutes - startHour * 60) / totalMinutes) * usable;
}

/**
 * Convert a pixel Y position to minutes, inverse of minutesToPx.
 */
export function pxToMinutes(y: number, startHour: number, endHour: number, containerHeight: number): number {
    const totalMinutes = (endHour - startHour) * 60;
    const usable = containerHeight - 2 * VERTICAL_INSET;
    return startHour * 60 + ((y - VERTICAL_INSET) / usable) * totalMinutes;
}

export function minutesToTime(minutes: number): string {
    const rounded = Math.round(minutes);
    const h = Math.floor(rounded / 60);
    const m = rounded % 60;
    return `${h.toString().padStart(2, '0')}:${m.toString().padStart(2, '0')}`;
}

export function generateId(): string {
    return `${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}

export function localDateString(date: Date): string {
    return [String(date.getFullYear()).padStart(4, '0'), String(date.getMonth() + 1).padStart(2, '0'), String(date.getDate()).padStart(2, '0')].join('-');
}
