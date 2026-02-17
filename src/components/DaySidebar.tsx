import React from 'react';
import { motion } from 'framer-motion';
import './DaySidebar.css';

interface DaySidebarProps {
    numDays: number;
    selectedDay: number;
    onSelectDay: (index: number) => void;
}

const DaySidebar: React.FC<DaySidebarProps> = ({ numDays, selectedDay, onSelectDay }) => {
    return (
        <div className="day-sidebar">
            <div className="day-sidebar-list">
                {Array.from({ length: numDays }, (_, i) => (
                    <button
                        key={i}
                        className={`day-sidebar-item ${selectedDay === i ? 'active' : ''}`}
                        onClick={() => onSelectDay(i)}
                    >
                        {selectedDay === i && (
                            <motion.div
                                className="day-sidebar-indicator"
                                layoutId="day-sidebar-indicator"
                                transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                            />
                        )}
                        <span className="day-sidebar-number">{i + 1}</span>
                    </button>
                ))}
            </div>
        </div>
    );
};

export default DaySidebar;
