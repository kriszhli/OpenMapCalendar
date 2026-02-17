import React from 'react';
import { LayoutGroup } from 'framer-motion';
import type { ViewMode, TimeBlock } from '../types';
import DayCard from './DayCard';
import './DayGrid.css';

interface DayGridProps {
    numDays: number;
    startDate: Date;
    startHour: number;
    endHour: number;
    viewMode: ViewMode;
    events: Record<number, TimeBlock[]>;
    hoveredEventId: string | null;
    onBlockCreated: (block: TimeBlock) => void;
    onBlockUpdated: (block: TimeBlock) => void;
    onBlockDeleted: (block: TimeBlock) => void;
    onHoverEvent: (id: string | null) => void;
    // Day view props
    dayViewIndex?: number;
    onPrevDay?: () => void;
    onNextDay?: () => void;
}

const DayGrid: React.FC<DayGridProps> = ({
    numDays,
    startDate,
    startHour,
    endHour,
    viewMode,
    events,
    hoveredEventId,
    onBlockCreated,
    onBlockUpdated,
    onBlockDeleted,
    onHoverEvent,
    dayViewIndex = 0,
    onPrevDay,
    onNextDay,
}) => {
    // In day view, render only the selected day
    if (viewMode === 'day') {
        return (
            <LayoutGroup>
                <div className="day-grid day-view">
                    <DayCard
                        key={dayViewIndex}
                        dayIndex={dayViewIndex}
                        startDate={startDate}
                        startHour={startHour}
                        endHour={endHour}
                        events={events[dayViewIndex] || []}
                        hoveredEventId={hoveredEventId}
                        onBlockCreated={onBlockCreated}
                        onBlockUpdated={onBlockUpdated}
                        onBlockDeleted={onBlockDeleted}
                        onHoverEvent={onHoverEvent}
                        showNavArrows
                        onPrevDay={onPrevDay}
                        onNextDay={onNextDay}
                    />
                </div>
            </LayoutGroup>
        );
    }

    return (
        <LayoutGroup>
            <div
                className={`day-grid ${viewMode === 'grid' ? 'grid-view' : 'row-view'}`}
                style={{ '--num-cols': Math.min(numDays, 7) } as React.CSSProperties}
            >
                {Array.from({ length: numDays }, (_, i) => (
                    <DayCard
                        key={i}
                        dayIndex={i}
                        startDate={startDate}
                        startHour={startHour}
                        endHour={endHour}
                        events={events[i] || []}
                        hoveredEventId={hoveredEventId}
                        onBlockCreated={onBlockCreated}
                        onBlockUpdated={onBlockUpdated}
                        onBlockDeleted={onBlockDeleted}
                        onHoverEvent={onHoverEvent}
                    />
                ))}
            </div>
        </LayoutGroup>
    );
};

export default DayGrid;
