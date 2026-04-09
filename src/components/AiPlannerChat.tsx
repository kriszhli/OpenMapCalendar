import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import './AiPlannerChat.css';

export interface AiChatMessage {
    id: string;
    role: 'user' | 'assistant';
    text: string;
    variant?: 'default' | 'success' | 'error';
}

export interface AiPlanDraftEvent {
    title: string;
    description?: string;
    date: string;
    startTime: string;
    endTime: string;
    origin?: string;
    destination?: string;
    color?: string;
}

export interface ClarificationChoice {
    label: string;
    date: string;
    start_time: string;
    end_time: string;
    description?: string;
}

export interface StagedAiPlan {
    status: 'ready' | 'needs_clarification';
    assistantMessage: string;
    events: AiPlanDraftEvent[];
    clarificationOptions: ClarificationChoice[];
    reasoning?: string;
    selectedClarificationOptionIndex?: number | null;
    isEditing?: boolean;
}

interface AiPlannerChatProps {
    messages: AiChatMessage[];
    loading: boolean;
    onSendMessage: (text: string) => void;
    canRollback?: boolean;
    onRollback?: () => void;
    disabled?: boolean;
    disabledHint?: string;
    stagedPlan?: StagedAiPlan | null;
    onCommit?: () => void;
    onToggleProposalEdit?: () => void;
    onUpdateProposalEvent?: (index: number, patch: Partial<AiPlanDraftEvent>) => void;
    onSelectClarificationOption?: (index: number) => void;
    onToggleReasoning?: () => void;
    showReasoning?: boolean;
    commitDisabled?: boolean;
    commitDisabledHint?: string;
}

const formatEventWindow = (event: AiPlanDraftEvent): string =>
    `${event.date} · ${event.startTime} - ${event.endTime}`;

const PlannerApprovalPanel: React.FC<AiPlannerChatProps> = ({
    messages,
    loading,
    onSendMessage,
    canRollback = false,
    onRollback,
    disabled = false,
    disabledHint = 'Create or select a calendar first.',
    stagedPlan = null,
    onCommit,
    onToggleProposalEdit,
    onUpdateProposalEvent,
    onSelectClarificationOption,
    onToggleReasoning,
    showReasoning = false,
    commitDisabled = false,
    commitDisabledHint = '',
}) => {
    const [open, setOpen] = useState(false);
    const [draft, setDraft] = useState('');
    const listRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!open) return;
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    }, [open, messages, loading, stagedPlan]);

    useEffect(() => {
        if (!open || disabled) return;
        const timer = window.setTimeout(() => inputRef.current?.focus(), 100);
        return () => window.clearTimeout(timer);
    }, [open, disabled]);

    const handleSubmit = (e: React.FormEvent) => {
        e.preventDefault();
        const text = draft.trim();
        if (!text || loading || disabled) return;
        setDraft('');
        onSendMessage(text);
    };

    const renderProposal = () => {
        if (!stagedPlan) return null;

        if (stagedPlan.status === 'needs_clarification') {
            return (
                <section className="ai-proposal-card">
                    <div className="ai-proposal-head">
                        <div>
                            <div className="ai-proposal-kicker">Needs clarification</div>
                            <div className="ai-proposal-title">{stagedPlan.assistantMessage}</div>
                        </div>
                        {onRollback && canRollback && (
                            <button type="button" className="ai-proposal-secondary" onClick={onRollback} disabled={loading}>
                                Clear
                            </button>
                        )}
                    </div>

                    <div className="ai-option-grid">
                        {(stagedPlan.clarificationOptions || []).slice(0, 2).map((option, index) => {
                            const selected = stagedPlan.selectedClarificationOptionIndex === index;
                            return (
                                <button
                                    key={`${option.label}-${option.date}-${index}`}
                                    type="button"
                                    className={`ai-option-card ${selected ? 'selected' : ''}`}
                                    onClick={() => onSelectClarificationOption?.(index)}
                                    disabled={loading}
                                >
                                    <div className="ai-option-label">{option.label}</div>
                                    <div className="ai-option-time">
                                        {option.date} · {option.start_time} - {option.end_time}
                                    </div>
                                    <div className="ai-option-desc">{option.description}</div>
                                </button>
                            );
                        })}
                    </div>
                </section>
            );
        }

        const selectedOptionText = stagedPlan.selectedClarificationOptionIndex !== null && stagedPlan.selectedClarificationOptionIndex !== undefined
            ? `Selected option ${stagedPlan.selectedClarificationOptionIndex + 1}`
            : '';

        return (
                <section className="ai-proposal-card">
                <div className="ai-proposal-head">
                    <div>
                        <div className="ai-proposal-kicker">Ready for approval</div>
                        <div className="ai-proposal-title">{stagedPlan.assistantMessage}</div>
                        {selectedOptionText && <div className="ai-proposal-meta">{selectedOptionText}</div>}
                    </div>
                    <div className="ai-proposal-actions">
                        {onToggleProposalEdit && (
                            <button
                                type="button"
                                className="ai-proposal-secondary"
                                onClick={onToggleProposalEdit}
                                disabled={loading}
                            >
                                {stagedPlan.isEditing ? 'Done' : 'Edit'}
                            </button>
                        )}
                        {onCommit && (
                            <button
                                type="button"
                                className="ai-proposal-primary"
                                onClick={onCommit}
                                disabled={loading || commitDisabled}
                                title={commitDisabled ? commitDisabledHint : undefined}
                            >
                                Commit
                            </button>
                        )}
                    </div>
                </div>

                <div className="ai-proposal-events">
                    {(stagedPlan.events || []).map((event, index) => {
                        const editable = !!stagedPlan.isEditing;
                        return (
                            <div className="ai-proposal-event" key={`${event.date}-${event.startTime}-${index}`}>
                                <div className="ai-event-row">
                                    {editable ? (
                                        <input
                                            className="ai-event-input ai-event-title"
                                            value={event.title}
                                            onChange={(e) => onUpdateProposalEvent?.(index, { title: e.target.value })}
                                            placeholder="Event title"
                                        />
                                    ) : (
                                        <div className="ai-event-title">{event.title}</div>
                                    )}
                                    <div className="ai-event-time">{formatEventWindow(event)}</div>
                                </div>

                                <div className="ai-event-grid">
                                    {editable ? (
                                        <>
                                            <input
                                                className="ai-event-input"
                                                value={event.date}
                                                onChange={(e) => onUpdateProposalEvent?.(index, { date: e.target.value })}
                                                placeholder="YYYY-MM-DD"
                                            />
                                            <input
                                                className="ai-event-input"
                                                value={event.startTime}
                                                onChange={(e) => onUpdateProposalEvent?.(index, { startTime: e.target.value })}
                                                placeholder="HH:MM"
                                            />
                                            <input
                                                className="ai-event-input"
                                                value={event.endTime}
                                                onChange={(e) => onUpdateProposalEvent?.(index, { endTime: e.target.value })}
                                                placeholder="HH:MM"
                                            />
                                            <input
                                                className="ai-event-input"
                                                value={event.origin || ''}
                                                onChange={(e) => onUpdateProposalEvent?.(index, { origin: e.target.value })}
                                                placeholder="Origin"
                                            />
                                            <input
                                                className="ai-event-input"
                                                value={event.destination || ''}
                                                onChange={(e) => onUpdateProposalEvent?.(index, { destination: e.target.value })}
                                                placeholder="Destination"
                                            />
                                        </>
                                    ) : (
                                        <div className="ai-event-summary">
                                            {event.description || 'No description provided.'}
                                        </div>
                                    )}
                                </div>
                            </div>
                        );
                    })}
                </div>

                {onToggleReasoning && (
                    <div className="ai-proposal-foot">
                        <button type="button" className="ai-proposal-link" onClick={onToggleReasoning} disabled={loading}>
                            {showReasoning ? 'Hide Reasoning' : 'Show Reasoning'}
                        </button>
                        {showReasoning && stagedPlan.reasoning && (
                            <pre className="ai-reasoning">
                                {stagedPlan.reasoning}
                            </pre>
                        )}
                    </div>
                )}
            </section>
        );
    };

    return (
        <div className="ai-panel-root">
            <button
                className={`ai-panel-fab ${open ? 'open' : ''}`}
                type="button"
                onClick={() => setOpen((prev) => !prev)}
                aria-label={open ? 'Close planner panel' : 'Open planner panel'}
            >
                {open ? (
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                    </svg>
                ) : (
                    <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8">
                        <path d="M21 15a4 4 0 0 1-4 4H8l-5 3V7a4 4 0 0 1 4-4h10a4 4 0 0 1 4 4z" />
                        <circle cx="9" cy="11" r="1" fill="currentColor" stroke="none" />
                        <circle cx="13" cy="11" r="1" fill="currentColor" stroke="none" />
                        <circle cx="17" cy="11" r="1" fill="currentColor" stroke="none" />
                    </svg>
                )}
            </button>

            <AnimatePresence>
                {open && (
                    <motion.section
                        className="ai-panel-panel"
                        initial={{ opacity: 0, y: 24, scale: 0.94 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 24, scale: 0.94 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                    >
                        <header className="ai-panel-header">
                            <div className="ai-panel-title-wrap">
                                <span className="ai-panel-title">AI Planner</span>
                                <span className="ai-panel-subtitle">Review proposals before approval</span>
                            </div>
                            {canRollback && onRollback && (
                                <button
                                    type="button"
                                    className="ai-panel-rollback"
                                    onClick={onRollback}
                                    disabled={loading}
                                >
                                    Rollback
                                </button>
                            )}
                        </header>

                        <div ref={listRef} className="ai-panel-messages">
                            {messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={[
                                        'ai-panel-message',
                                        msg.role === 'user' ? 'user' : 'system',
                                        msg.variant === 'error' ? 'error' : '',
                                        msg.variant === 'success' ? 'success' : '',
                                    ]
                                        .filter(Boolean)
                                        .join(' ')}
                                >
                                    {msg.text}
                                </div>
                            ))}

                            {loading && (
                                <div className="ai-panel-message system pending">
                                    Thinking...
                                </div>
                            )}

                            {disabled && (
                                <div className="ai-panel-status">
                                    {disabledHint}
                                </div>
                            )}
                        </div>

                        {renderProposal()}

                        <form className="ai-panel-input-row" onSubmit={handleSubmit}>
                            <input
                                ref={inputRef}
                                className="ai-panel-input"
                                value={draft}
                                onChange={(e) => setDraft(e.target.value)}
                                placeholder="Describe your plan, dates, and places..."
                                disabled={loading || disabled}
                            />
                            <button
                                type="submit"
                                className="ai-panel-send"
                                disabled={loading || disabled || !draft.trim()}
                            >
                                Send
                            </button>
                        </form>
                    </motion.section>
                )}
            </AnimatePresence>
        </div>
    );
};

export default PlannerApprovalPanel;
