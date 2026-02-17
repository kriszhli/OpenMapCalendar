import React, { useMemo } from 'react';
import { minutesToPx, minutesToTime } from '../types';
import type { TimeBlock } from '../types';
import './TimeRuler.css';

interface TimeRulerProps {
    startHour: number;
    endHour: number;
    height: number;
    hoveredMinute: number | null;
    events?: TimeBlock[];
    dragMinutes?: { start: number; end: number } | null;
}

const TimeRuler: React.FC<TimeRulerProps> = ({ startHour, endHour, height, hoveredMinute, events = [], dragMinutes = null }) => {
    const rulerWidth = 52;

    // All labels: hours + half-hours
    const allLabels = useMemo(() => {
        const result: { minute: number; label: string; isHalfHour: boolean }[] = [];
        const startMin = startHour * 60;
        const endMin = endHour * 60;

        for (let m = startMin; m <= endMin; m += 30) {
            const hour = Math.floor(m / 60);
            const min = m % 60;
            result.push({
                minute: m,
                label: `${hour.toString().padStart(2, '0')}:${min.toString().padStart(2, '0')}`,
                isHalfHour: min === 30,
            });
        }
        return result;
    }, [startHour, endHour]);

    // Event boundary minutes for highlighting
    const eventBoundaryMinutes = useMemo(() => {
        const set = new Set<number>();
        events.forEach((ev) => {
            set.add(ev.startMinutes);
            set.add(ev.endMinutes);
        });
        return set;
    }, [events]);

    // Non-hour, non-half-hour event boundary labels (like "13:15", "13:45")
    const boundaryLabels = useMemo(() => {
        const standardMinutes = new Set<number>();
        for (let m = startHour * 60; m <= endHour * 60; m += 30) {
            standardMinutes.add(m);
        }

        const boundaries = new Map<number, string>();
        events.forEach((ev) => {
            if (!standardMinutes.has(ev.startMinutes)) {
                boundaries.set(ev.startMinutes, minutesToTime(ev.startMinutes));
            }
            if (!standardMinutes.has(ev.endMinutes)) {
                boundaries.set(ev.endMinutes, minutesToTime(ev.endMinutes));
            }
        });

        return Array.from(boundaries.entries()).map(([minute, label]) => ({ minute, label }));
    }, [startHour, endHour, events]);

    if (height <= 0) return null;

    return (
        <div className="time-ruler">
            <svg
                viewBox={`0 0 ${rulerWidth} ${height}`}
                preserveAspectRatio="none"
                width={rulerWidth}
                height={height}
            >
                {/* Hour + Half-hour labels */}
                {allLabels.map((item) => {
                    const y = minutesToPx(item.minute, startHour, endHour, height);
                    const isHovered = hoveredMinute === item.minute;
                    const isEventBoundary = eventBoundaryMinutes.has(item.minute);
                    const isDragActive = dragMinutes !== null && (
                        item.minute === dragMinutes.start || item.minute === dragMinutes.end
                    );

                    const classes = [
                        'ruler-label',
                        item.isHalfHour ? 'half-hour' : '',
                        isHovered ? 'hovered' : '',
                        isEventBoundary ? 'event-boundary' : '',
                        isDragActive ? 'drag-active' : '',
                    ].filter(Boolean).join(' ');

                    return (
                        <text
                            key={item.minute}
                            className={classes}
                            x={48}
                            y={y}
                            textAnchor="end"
                            dominantBaseline="middle"
                        >
                            {item.label}
                        </text>
                    );
                })}

                {/* Non-standard event boundary labels (e.g., "13:15") */}
                {boundaryLabels.map((item) => {
                    const y = minutesToPx(item.minute, startHour, endHour, height);

                    return (
                        <text
                            key={`b-${item.minute}`}
                            className="ruler-label event-boundary"
                            x={36}
                            y={y}
                            textAnchor="end"
                            dominantBaseline="middle"
                        >
                            {item.label}
                        </text>
                    );
                })}
            </svg>
        </div>
    );
};

export default React.memo(TimeRuler);
