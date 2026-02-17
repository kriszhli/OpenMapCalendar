import React, { useRef, useState, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import type { TimeBlock } from '../types';
import { minutesToPx } from '../types';
import TimeRuler from './TimeRuler';
import DragSelectionLayer from './DragSelectionLayer';
import EventBlock from './EventBlock';
import './DayCard.css';

interface DayCardProps {
    dayIndex: number;
    startDate: Date;
    startHour: number;
    endHour: number;
    events: TimeBlock[];
    hoveredEventId: string | null;
    onBlockCreated: (block: TimeBlock) => void;
    onBlockUpdated: (block: TimeBlock) => void;
    onBlockDeleted: (block: TimeBlock) => void;
    onHoverEvent: (id: string | null) => void;
    showNavArrows?: boolean;
    onPrevDay?: () => void;
    onNextDay?: () => void;
}

const DAY_NAMES = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

const DayCard: React.FC<DayCardProps> = ({
    dayIndex,
    startDate,
    startHour,
    endHour,
    events,
    hoveredEventId,
    onBlockCreated,
    onBlockUpdated,
    onBlockDeleted,
    onHoverEvent,
    showNavArrows,
    onPrevDay,
    onNextDay,
}) => {
    const bodyRef = useRef<HTMLDivElement>(null);
    const [bodyHeight, setBodyHeight] = useState(0);
    const [hoveredMinute, setHoveredMinute] = useState<number | null>(null);
    const [dragMinutes, setDragMinutes] = useState<{ start: number; end: number } | null>(null);

    useEffect(() => {
        const el = bodyRef.current;
        if (!el) return;

        const observer = new ResizeObserver(([entry]) => {
            setBodyHeight(entry.contentRect.height);
        });
        observer.observe(el);
        return () => observer.disconnect();
    }, []);

    const handleBlockCreated = useCallback(
        (block: TimeBlock) => {
            onBlockCreated(block);
        },
        [onBlockCreated]
    );

    const handleHoverMinuteChange = useCallback((minute: number | null) => {
        setHoveredMinute(minute);
    }, []);

    const handleDragMinutesChange = useCallback((minutes: { start: number; end: number } | null) => {
        setDragMinutes(minutes);
    }, []);

    // Calculate display date
    const displayDate = new Date(startDate);
    displayDate.setDate(displayDate.getDate() + dayIndex);
    const dayName = DAY_NAMES[displayDate.getDay()];
    const dateNum = displayDate.getDate().toString().padStart(2, '0');

    const hoverLineY = hoveredMinute !== null
        ? minutesToPx(hoveredMinute, startHour, endHour, bodyHeight)
        : 0;

    return (
        <motion.div
            className="day-card"
            layout
            layoutId={`day-card-${dayIndex}`}
            transition={{
                layout: { type: 'spring', stiffness: 300, damping: 30 },
            }}
        >
            <div className={`day-card-header ${showNavArrows ? 'day-card-header-nav' : ''}`}>
                {showNavArrows ? (
                    <>
                        <button className="day-nav-arrow" onClick={onPrevDay}>
                            <svg width="7" height="12" viewBox="0 0 7 12" fill="currentColor"><path d="M6 1L1 6L6 11" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
                        </button>
                        <div className="day-card-header-center">
                            <span className="day-card-number">{dateNum}</span>
                            <span className="day-card-label">{dayName}</span>
                        </div>
                        <button className="day-nav-arrow" onClick={onNextDay}>
                            <svg width="7" height="12" viewBox="0 0 7 12" fill="currentColor"><path d="M1 1L6 6L1 11" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" /></svg>
                        </button>
                    </>
                ) : (
                    <>
                        <span className="day-card-number">{dateNum}</span>
                        <span className="day-card-label">{dayName}</span>
                    </>
                )}
            </div>

            <div className="day-card-body" ref={bodyRef}>
                {bodyHeight > 0 && (
                    <>
                        {/* Full-width hover line */}
                        {hoveredMinute !== null && (
                            <div
                                className="day-card-hover-line"
                                style={{ top: hoverLineY }}
                            />
                        )}

                        <TimeRuler
                            startHour={startHour}
                            endHour={endHour}
                            height={bodyHeight}
                            hoveredMinute={hoveredMinute}
                            events={events}
                            dragMinutes={dragMinutes}
                        />

                        <DragSelectionLayer
                            dayIndex={dayIndex}
                            startHour={startHour}
                            endHour={endHour}
                            containerHeight={bodyHeight}
                            onBlockCreated={handleBlockCreated}
                            onHoverMinuteChange={handleHoverMinuteChange}
                            onDragMinutesChange={handleDragMinutesChange}
                        />

                        {events.map((block) => (
                            <EventBlock
                                key={block.id}
                                block={block}
                                startHour={startHour}
                                endHour={endHour}
                                containerHeight={bodyHeight}
                                isHighlighted={hoveredEventId === block.id}
                                onUpdate={onBlockUpdated}
                                onDelete={onBlockDeleted}
                                onHoverEvent={onHoverEvent}
                            />
                        ))}
                    </>
                )}
            </div>
        </motion.div>
    );
};

export default React.memo(DayCard);
