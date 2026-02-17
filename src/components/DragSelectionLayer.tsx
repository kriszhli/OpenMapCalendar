import React, { useRef, useState, useCallback } from 'react';
import { snapToInterval, minutesToPx, pxToMinutes, generateId } from '../types';
import type { TimeBlock } from '../types';
import './DragSelectionLayer.css';

interface DragSelectionLayerProps {
    dayIndex: number;
    startHour: number;
    endHour: number;
    containerHeight: number;
    onBlockCreated: (block: TimeBlock) => void;
    onHoverMinuteChange: (minute: number | null) => void;
    onDragMinutesChange: (minutes: { start: number; end: number } | null) => void;
}

const DragSelectionLayer: React.FC<DragSelectionLayerProps> = ({
    dayIndex,
    startHour,
    endHour,
    containerHeight,
    onBlockCreated,
    onHoverMinuteChange,
    onDragMinutesChange,
}) => {
    const layerRef = useRef<HTMLDivElement>(null);
    const [isDragging, setIsDragging] = useState(false);
    const [dragStart, setDragStart] = useState(0);
    const [dragEnd, setDragEnd] = useState(0);

    const toMinutes = useCallback(
        (y: number) => snapToInterval(pxToMinutes(y, startHour, endHour, containerHeight)),
        [startHour, endHour, containerHeight]
    );

    const toPx = useCallback(
        (minutes: number) => minutesToPx(minutes, startHour, endHour, containerHeight),
        [startHour, endHour, containerHeight]
    );

    const handlePointerDown = useCallback(
        (e: React.PointerEvent) => {
            if (e.button !== 0) return;
            const rect = layerRef.current?.getBoundingClientRect();
            if (!rect) return;

            const y = e.clientY - rect.top;
            const minutes = toMinutes(y);

            setDragStart(minutes);
            setDragEnd(minutes);
            setIsDragging(true);
            onDragMinutesChange({ start: minutes, end: minutes });

            (e.target as HTMLElement).setPointerCapture(e.pointerId);
        },
        [toMinutes, onDragMinutesChange]
    );

    const handlePointerMove = useCallback(
        (e: React.PointerEvent) => {
            const rect = layerRef.current?.getBoundingClientRect();
            if (!rect) return;

            const y = Math.max(0, Math.min(containerHeight, e.clientY - rect.top));

            if (!isDragging) {
                const rawMinute = pxToMinutes(y, startHour, endHour, containerHeight);
                const snappedTo30 = Math.round(rawMinute / 30) * 30;
                onHoverMinuteChange(snappedTo30);
                return;
            }

            const minutes = toMinutes(y);
            setDragEnd(minutes);
            const startMin = Math.min(dragStart, minutes);
            const endMin = Math.max(dragStart, minutes);
            onDragMinutesChange({ start: startMin, end: endMin });
        },
        [isDragging, containerHeight, toMinutes, dragStart, startHour, endHour, onHoverMinuteChange, onDragMinutesChange]
    );

    const handlePointerUp = useCallback(
        (e: React.PointerEvent) => {
            if (!isDragging) return;
            setIsDragging(false);
            onDragMinutesChange(null);

            (e.target as HTMLElement).releasePointerCapture(e.pointerId);

            const startMin = Math.min(dragStart, dragEnd);
            const endMin = Math.max(dragStart, dragEnd);

            if (endMin - startMin < 30) return;

            const newBlock: TimeBlock = {
                id: generateId(),
                dayIndex,
                startMinutes: startMin,
                endMinutes: endMin,
                color: '',
                title: '',
                description: '',
            };

            onBlockCreated(newBlock);
        },
        [isDragging, dragStart, dragEnd, dayIndex, onBlockCreated, onDragMinutesChange]
    );

    const handlePointerLeave = useCallback(() => {
        if (!isDragging) {
            onHoverMinuteChange(null);
        }
    }, [isDragging, onHoverMinuteChange]);

    const selStartMin = Math.min(dragStart, dragEnd);
    const selEndMin = Math.max(dragStart, dragEnd);
    const topPx = toPx(selStartMin);
    const bottomPx = toPx(selEndMin);
    const heightPx = bottomPx - topPx;

    return (
        <div
            ref={layerRef}
            className="drag-layer"
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerLeave={handlePointerLeave}
        >
            {isDragging && heightPx > 0 && (
                <>
                    {/* Highlighted lines at exact snap positions */}
                    <div className="drag-snap-line" style={{ top: topPx }} />
                    <div className="drag-snap-line" style={{ top: bottomPx }} />

                    {/* Selection rectangle between the lines */}
                    <div
                        className="drag-selection"
                        style={{ top: topPx, height: heightPx }}
                    >
                        <div className="drag-selection-inner" />
                    </div>
                </>
            )}
        </div>
    );
};

export default DragSelectionLayer;
