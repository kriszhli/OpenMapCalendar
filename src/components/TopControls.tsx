import React, { useRef } from 'react';
import { motion } from 'framer-motion';
import type { ViewMode } from '../types';
import './TopControls.css';

interface TopControlsProps {
    numDays: number;
    startDate: Date;
    startHour: number;
    endHour: number;
    viewMode: ViewMode;
    onNumDaysChange: (n: number) => void;
    onStartDateChange: (d: Date) => void;
    onStartHourChange: (h: number) => void;
    onEndHourChange: (h: number) => void;
    onViewModeChange: (m: ViewMode) => void;
}

const ControlArrow = ({ direction, onClick }: { direction: 'up' | 'down'; onClick: () => void }) => (
    <button className="control-arrow" onClick={(e) => { e.stopPropagation(); onClick(); }}>
        <svg width="8" height="6" viewBox="0 0 8 6" fill="currentColor">
            {direction === 'up' ? (
                <path d="M4 0L8 6H0L4 0Z" />
            ) : (
                <path d="M4 6L0 0H8L4 6Z" />
            )}
        </svg>
    </button>
);

interface NumberControlProps {
    label: string;
    value: React.ReactNode;
    onIncrement: () => void;
    onDecrement: () => void;
    minWidth?: number;
}

const NumberControl = ({ label, value, onIncrement, onDecrement, minWidth = 40 }: NumberControlProps) => (
    <div className="control-group">
        <span className="control-label">{label}</span>
        <div className="custom-control-container" style={{ minWidth }}>
            <span className="control-value">{value}</span>
            <div className="control-arrows">
                <ControlArrow direction="up" onClick={onIncrement} />
                <ControlArrow direction="down" onClick={onDecrement} />
            </div>
        </div>
    </div>
);

const VIEW_MODES: ViewMode[] = ['day', 'row', 'grid'];

const TopControls: React.FC<TopControlsProps> = ({
    numDays,
    startDate,
    startHour,
    endHour,
    viewMode,
    onNumDaysChange,
    onStartDateChange,
    onStartHourChange,
    onEndHourChange,
    onViewModeChange,
}) => {
    const dateInputRef = useRef<HTMLInputElement>(null);

    // Format date as "Oct 14"
    const displayDate = startDate.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });

    const handleDateClick = () => {
        if (dateInputRef.current) {
            try {
                dateInputRef.current.showPicker();
            } catch (e) {
                dateInputRef.current.focus();
                dateInputRef.current.click();
            }
        }
    };

    const handleDateIncrement = () => {
        const next = new Date(startDate);
        next.setDate(next.getDate() + 1);
        onStartDateChange(next);
    };

    const handleDateDecrement = () => {
        const prev = new Date(startDate);
        prev.setDate(prev.getDate() - 1);
        onStartDateChange(prev);
    };

    const handleStartHourChange = (newHour: number) => onStartHourChange(Math.max(0, Math.min(23, newHour)));
    const handleEndHourChange = (newHour: number) => onEndHourChange(Math.max(0, Math.min(23, newHour)));
    const handleDaysChange = (newDays: number) => onNumDaysChange(Math.max(1, Math.min(31, newDays)));

    const formatTime = (h: number) => `${Math.floor(h).toString().padStart(2, '0')}:00`;

    const activeIndex = VIEW_MODES.indexOf(viewMode);

    const [theme, setTheme] = React.useState<'dark' | 'bright'>(() => {
        if (typeof window !== 'undefined') {
            const saved = localStorage.getItem('app-theme');
            if (saved === 'dark' || saved === 'bright') return saved;
            return window.matchMedia('(prefers-color-scheme: light)').matches ? 'bright' : 'dark';
        }
        return 'dark';
    });

    React.useEffect(() => {
        document.documentElement.setAttribute('data-theme', theme);
        localStorage.setItem('app-theme', theme);
    }, [theme]);

    const toggleTheme = () => setTheme(prev => prev === 'dark' ? 'bright' : 'dark');

    return (
        <div className="top-controls">
            {/* Start Date Selector */}
            <div className="control-group">
                <span className="control-label">Start Date</span>
                <div className="custom-control-container date-control">
                    <span className="control-value date-value" onClick={handleDateClick}>{displayDate}</span>
                    <input
                        ref={dateInputRef}
                        type="date"
                        className="hidden-date-input"
                        value={startDate.toISOString().split('T')[0]}
                        onChange={(e) => {
                            if (e.target.value) {
                                const d = new Date(e.target.value + 'T00:00:00');
                                onStartDateChange(d);
                            }
                        }}
                    />
                    <div className="control-arrows">
                        <ControlArrow direction="up" onClick={handleDateIncrement} />
                        <ControlArrow direction="down" onClick={handleDateDecrement} />
                    </div>
                </div>
            </div>


            <div className="control-divider" />

            {/* Days Selector */}
            <NumberControl
                label="Days"
                value={numDays}
                onIncrement={() => handleDaysChange(numDays + 1)}
                onDecrement={() => handleDaysChange(numDays - 1)}
            />

            <div className="control-divider" />

            {/* Time Selectors */}
            <NumberControl
                label="From"
                value={formatTime(startHour)}
                onIncrement={() => handleStartHourChange(Math.floor(startHour) + 1)}
                onDecrement={() => handleStartHourChange(Math.floor(startHour) - 1)}
                minWidth={60}
            />

            <NumberControl
                label="To"
                value={formatTime(endHour)}
                onIncrement={() => handleEndHourChange(Math.floor(endHour) + 1)}
                onDecrement={() => handleEndHourChange(Math.floor(endHour) - 1)}
                minWidth={60}
            />

            <div className="control-divider" />

            {/* View Toggle — 3 buttons */}
            <div className="view-toggle">
                <motion.div
                    className="view-toggle-indicator"
                    layout
                    layoutId="view-toggle-indicator"
                    style={{
                        left: `calc(${activeIndex} * (100% / 3) + 2px)`,
                        width: 'calc(100% / 3 - 2px)',
                    }}
                    transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                />
                {VIEW_MODES.map((mode) => (
                    <button
                        key={mode}
                        className={`view-toggle-btn ${viewMode === mode ? 'active' : ''}`}
                        onClick={() => onViewModeChange(mode)}
                    >
                        {mode.charAt(0).toUpperCase() + mode.slice(1)}
                    </button>
                ))}
            </div>

            {/* Theme Toggle */}
            <button className="theme-toggle-btn" onClick={toggleTheme}>
                {theme === 'dark' ? (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <circle cx="12" cy="12" r="5" />
                        <line x1="12" y1="1" x2="12" y2="3" />
                        <line x1="12" y1="21" x2="12" y2="23" />
                        <line x1="4.22" y1="4.22" x2="5.64" y2="5.64" />
                        <line x1="18.36" y1="18.36" x2="19.78" y2="19.78" />
                        <line x1="1" y1="12" x2="3" y2="12" />
                        <line x1="21" y1="12" x2="23" y2="12" />
                        <line x1="4.22" y1="19.78" x2="5.64" y2="18.36" />
                        <line x1="18.36" y1="5.64" x2="19.78" y2="4.22" />
                    </svg>
                ) : (
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
                    </svg>
                )}
            </button>
        </div>
    );
};

export default TopControls;
