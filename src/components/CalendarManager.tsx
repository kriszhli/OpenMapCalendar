import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import './CalendarManager.css';

interface CalendarEntry {
    id: string;
    name: string;
    updatedAt: string;
}

interface CalendarManagerProps {
    visible: boolean;
    currentCalendarId: string | null;
    onSelectCalendar: (id: string) => void;
    onCreateCalendar: (name: string) => void;
    onRenameCalendar: (id: string, name: string) => void;
    onDeleteCalendar: (id: string) => void;
    onClose: () => void;
}

const CalendarManager: React.FC<CalendarManagerProps> = ({
    visible,
    currentCalendarId,
    onSelectCalendar,
    onCreateCalendar,
    onRenameCalendar,
    onDeleteCalendar,
    onClose,
}) => {
    const [calendars, setCalendars] = useState<CalendarEntry[]>([]);
    const [loading, setLoading] = useState(true);
    const [creating, setCreating] = useState(false);
    const [newName, setNewName] = useState('');
    const [renamingId, setRenamingId] = useState<string | null>(null);
    const [renameValue, setRenameValue] = useState('');
    const [deleteConfirmId, setDeleteConfirmId] = useState<string | null>(null);
    const newNameRef = useRef<HTMLInputElement>(null);
    const renameRef = useRef<HTMLInputElement>(null);

    const fetchCalendars = useCallback(async (background = false) => {
        try {
            if (!background) setLoading(true);
            const res = await fetch('/api/calendars');
            const data: CalendarEntry[] = await res.json();
            setCalendars(data);
        } catch {
            if (!background) setCalendars([]);
        } finally {
            setLoading(false);
        }
    }, []);

    useEffect(() => {
        if (visible) {
            fetchCalendars();
            setCreating(false);
            setRenamingId(null);
            setDeleteConfirmId(null);
            const timer = window.setInterval(() => void fetchCalendars(true), 2000);
            return () => window.clearInterval(timer);
        }
    }, [visible, fetchCalendars]);

    useEffect(() => {
        if (creating) {
            setTimeout(() => newNameRef.current?.focus(), 100);
        }
    }, [creating]);

    useEffect(() => {
        if (renamingId) {
            setTimeout(() => renameRef.current?.focus(), 100);
        }
    }, [renamingId]);

    const handleCreate = useCallback(() => {
        if (!newName.trim()) return;
        onCreateCalendar(newName.trim());
        setNewName('');
        setCreating(false);
        // Refresh after a short delay to let the parent handle creation
        setTimeout(fetchCalendars, 200);
    }, [newName, onCreateCalendar, fetchCalendars]);

    const handleRename = useCallback((id: string) => {
        if (!renameValue.trim()) return;
        onRenameCalendar(id, renameValue.trim());
        setRenamingId(null);
        setRenameValue('');
        setTimeout(fetchCalendars, 200);
    }, [renameValue, onRenameCalendar, fetchCalendars]);

    const handleDelete = useCallback((id: string) => {
        onDeleteCalendar(id);
        setDeleteConfirmId(null);
        setTimeout(fetchCalendars, 200);
    }, [onDeleteCalendar, fetchCalendars]);

    const formatDate = (iso: string) => {
        const d = new Date(iso);
        return d.toLocaleDateString('en-US', {
            month: 'short', day: 'numeric', year: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    };

    const hasCalendars = calendars.length > 0;

    return (
        <AnimatePresence>
            {visible && (
                <motion.div
                    className="calendar-manager-backdrop"
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    exit={{ opacity: 0 }}
                    transition={{ duration: 0.25 }}
                    onClick={(e) => {
                        // Only close from backdrop if there are calendars to return to
                        if (hasCalendars && e.target === e.currentTarget) onClose();
                    }}
                >
                    <motion.div
                        className="calendar-manager-modal"
                        initial={{ opacity: 0, scale: 0.92, y: 24 }}
                        animate={{ opacity: 1, scale: 1, y: 0 }}
                        exit={{ opacity: 0, scale: 0.92, y: 24 }}
                        transition={{ type: 'spring', stiffness: 400, damping: 30 }}
                        onClick={(e) => e.stopPropagation()}
                    >
                        {/* Header */}
                        <div className="cm-header">
                            <div className="cm-header-left">
                                <svg className="cm-logo" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round">
                                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                                    <line x1="16" y1="2" x2="16" y2="6" />
                                    <line x1="8" y1="2" x2="8" y2="6" />
                                    <line x1="3" y1="10" x2="21" y2="10" />
                                </svg>
                                <h2 className="cm-title">Calendar Manager</h2>
                            </div>
                            <button aria-label="Close calendar manager" className="cm-close-btn" onClick={onClose}>
                                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                    <line x1="18" y1="6" x2="6" y2="18" />
                                    <line x1="6" y1="6" x2="18" y2="18" />
                                </svg>
                            </button>
                        </div>

                        {/* Calendar list */}
                        <div className="cm-body">
                            {loading ? (
                                <div className="cm-loading">Loading calendars...</div>
                            ) : (
                                <>
                                    {hasCalendars && (
                                        <div className="cm-section-label">Recent Calendars</div>
                                    )}
                                    <div className="cm-list">
                                        {calendars.map((cal) => (
                                            <div
                                                key={cal.id}
                                                className={`cm-calendar-row ${cal.id === currentCalendarId ? 'current' : ''}`}
                                            >
                                                {renamingId === cal.id ? (
                                                    <div className="cm-rename-row">
                                                        <input
                                                            ref={renameRef}
                                                            className="cm-rename-input"
                                                            value={renameValue}
                                                            onChange={(e) => setRenameValue(e.target.value)}
                                                            onKeyDown={(e) => {
                                                                if (e.key === 'Enter') handleRename(cal.id);
                                                                if (e.key === 'Escape') setRenamingId(null);
                                                            }}
                                                            placeholder="Calendar name..."
                                                        />
                                                        <button className="cm-action-btn save" onClick={() => handleRename(cal.id)}>Save</button>
                                                        <button className="cm-action-btn cancel" onClick={() => setRenamingId(null)}>Cancel</button>
                                                    </div>
                                                ) : deleteConfirmId === cal.id ? (
                                                    <div className="cm-delete-confirm">
                                                        <span className="cm-delete-msg">Delete "{cal.name}"?</span>
                                                        <button className="cm-action-btn delete" onClick={() => handleDelete(cal.id)}>Delete</button>
                                                        <button className="cm-action-btn cancel" onClick={() => setDeleteConfirmId(null)}>Cancel</button>
                                                    </div>
                                                ) : (
                                                    <>
                                                        <div
                                                            className="cm-calendar-info"
                                                            onClick={() => { onSelectCalendar(cal.id); onClose(); }}
                                                        >
                                                            <span className="cm-calendar-name">{cal.name}</span>
                                                            <span className="cm-calendar-date">{formatDate(cal.updatedAt)}</span>
                                                        </div>
                                                        <div className="cm-calendar-actions">
                                                            <button
                                                                className="cm-icon-btn"
                                                                title="Rename"
                                                                onClick={() => { setRenamingId(cal.id); setRenameValue(cal.name); }}
                                                            >
                                                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                                    <path d="M11 4H4a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h14a2 2 0 0 0 2-2v-7" />
                                                                    <path d="M18.5 2.5a2.121 2.121 0 0 1 3 3L12 15l-4 1 1-4 9.5-9.5z" />
                                                                </svg>
                                                            </button>
                                                            <button
                                                                className="cm-icon-btn delete"
                                                                title="Delete"
                                                                onClick={() => setDeleteConfirmId(cal.id)}
                                                            >
                                                                <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                                                    <polyline points="3 6 5 6 21 6" />
                                                                    <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                                                                </svg>
                                                            </button>
                                                        </div>
                                                    </>
                                                )}
                                            </div>
                                        ))}
                                    </div>

                                    {!hasCalendars && (
                                        <div className="cm-empty">
                                            <div className="cm-empty-icon">📅</div>
                                            <div className="cm-empty-text">No calendars yet</div>
                                            <div className="cm-empty-sub">Create your first calendar to get started</div>
                                        </div>
                                    )}
                                </>
                            )}
                        </div>

                        {/* Footer: create button */}
                        <div className="cm-footer">
                            {creating ? (
                                <div className="cm-create-row">
                                    <input
                                        ref={newNameRef}
                                        className="cm-create-input"
                                        value={newName}
                                        onChange={(e) => setNewName(e.target.value)}
                                        onKeyDown={(e) => {
                                            if (e.key === 'Enter') handleCreate();
                                            if (e.key === 'Escape') setCreating(false);
                                        }}
                                        placeholder="Calendar name..."
                                    />
                                    <button className="cm-create-confirm" onClick={handleCreate} disabled={!newName.trim()}>
                                        Create
                                    </button>
                                </div>
                            ) : (
                                <button className="cm-create-btn" onClick={() => setCreating(true)}>
                                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                        <line x1="12" y1="5" x2="12" y2="19" />
                                        <line x1="5" y1="12" x2="19" y2="12" />
                                    </svg>
                                    New Calendar
                                </button>
                            )}
                        </div>
                    </motion.div>
                </motion.div>
            )}
        </AnimatePresence>
    );
};

export default CalendarManager;
