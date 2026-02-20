import React, { useEffect, useRef, useState } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import './AiPlannerChat.css';

export interface AiChatMessage {
    id: string;
    role: 'user' | 'assistant';
    text: string;
    variant?: 'default' | 'success' | 'error';
}

interface AiPlannerChatProps {
    messages: AiChatMessage[];
    loading: boolean;
    onSendMessage: (text: string) => void;
    canRollback?: boolean;
    onRollback?: () => void;
    disabled?: boolean;
    disabledHint?: string;
}

const AiPlannerChat: React.FC<AiPlannerChatProps> = ({
    messages,
    loading,
    onSendMessage,
    canRollback = false,
    onRollback,
    disabled = false,
    disabledHint = 'Create or select a calendar first.',
}) => {
    const [open, setOpen] = useState(false);
    const [draft, setDraft] = useState('');
    const listRef = useRef<HTMLDivElement>(null);
    const inputRef = useRef<HTMLInputElement>(null);

    useEffect(() => {
        if (!open) return;
        listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: 'smooth' });
    }, [open, messages, loading]);

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

    return (
        <div className="ai-chat-root">
            <button
                className={`ai-chat-fab ${open ? 'open' : ''}`}
                type="button"
                onClick={() => setOpen((prev) => !prev)}
                aria-label={open ? 'Close AI assistant' : 'Open AI assistant'}
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
                        className="ai-chat-panel"
                        initial={{ opacity: 0, y: 24, scale: 0.94 }}
                        animate={{ opacity: 1, y: 0, scale: 1 }}
                        exit={{ opacity: 0, y: 24, scale: 0.94 }}
                        transition={{ duration: 0.2, ease: 'easeOut' }}
                    >
                        <header className="ai-chat-header">
                            <div className="ai-chat-title-wrap">
                                <span className="ai-chat-title">AI Planner</span>
                                <span className="ai-chat-subtitle">Schedule by plain language</span>
                            </div>
                            {canRollback && onRollback && (
                                <button
                                    type="button"
                                    className="ai-chat-rollback"
                                    onClick={onRollback}
                                    disabled={loading}
                                >
                                    Rollback
                                </button>
                            )}
                        </header>

                        <div ref={listRef} className="ai-chat-messages">
                            {messages.map((msg) => (
                                <div
                                    key={msg.id}
                                    className={[
                                        'ai-chat-bubble',
                                        msg.role === 'user' ? 'user' : 'assistant',
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
                                <div className="ai-chat-bubble assistant pending">
                                    Thinking...
                                </div>
                            )}

                            {disabled && (
                                <div className="ai-chat-status">
                                    {disabledHint}
                                </div>
                            )}
                        </div>

                        <form className="ai-chat-input-row" onSubmit={handleSubmit}>
                            <input
                                ref={inputRef}
                                className="ai-chat-input"
                                value={draft}
                                onChange={(e) => setDraft(e.target.value)}
                                placeholder="Describe your plan, dates, and places..."
                                disabled={loading || disabled}
                            />
                            <button
                                type="submit"
                                className="ai-chat-send"
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

export default AiPlannerChat;
