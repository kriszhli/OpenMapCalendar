import React, { useState, useCallback, useRef, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { TimeBlock, LocationData } from '../types';
import { minutesToPx } from '../types';
import ColorPalette from './ColorPalette';
import './EventBlock.css';

interface NominatimResult {
    display_name: string;
    lat: string;
    lon: string;
}

interface EventBlockProps {
    block: TimeBlock;
    startHour: number;
    endHour: number;
    containerHeight: number;
    isHighlighted: boolean;
    onUpdate: (block: TimeBlock) => void;
    onDelete: (block: TimeBlock) => void;
    onHoverEvent: (id: string | null) => void;
}

const EventBlock: React.FC<EventBlockProps> = ({
    block,
    startHour,
    endHour,
    containerHeight,
    isHighlighted,
    onUpdate,
    onDelete,
    onHoverEvent,
}) => {
    const [titleFocused, setTitleFocused] = useState(false);
    const [bodyFocused, setBodyFocused] = useState(false);
    const [locationFocused, setLocationFocused] = useState(false);
    const [destFocused, setDestFocused] = useState(false);
    const [colorPickerOpen, setColorPickerOpen] = useState(false);
    const [showEditPopout, setShowEditPopout] = useState(false);
    const [locationQuery, setLocationQuery] = useState('');
    const [searchResults, setSearchResults] = useState<NominatimResult[]>([]);
    const [isSearching, setIsSearching] = useState(false);
    const [destQuery, setDestQuery] = useState('');
    const [destResults, setDestResults] = useState<NominatimResult[]>([]);
    const [isDestSearching, setIsDestSearching] = useState(false);
    const titleRef = useRef<HTMLInputElement>(null);
    const descRef = useRef<HTMLTextAreaElement>(null);
    const popoutTitleRef = useRef<HTMLInputElement>(null);
    const searchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const destSearchTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
    const blockRef = useRef<HTMLDivElement>(null);
    const popoutRef = useRef<HTMLDivElement>(null);

    const topPx = minutesToPx(block.startMinutes, startHour, endHour, containerHeight);
    const bottomPx = minutesToPx(block.endMinutes, startHour, endHour, containerHeight);
    const heightPx = bottomPx - topPx;

    const durationMinutes = block.endMinutes - block.startMinutes;
    const isCompact = durationMinutes < 150;  // < 2.5 hours
    const isHalfHour = durationMinutes <= 30;

    // Show palette when editing or when no color chosen yet
    const isEditing = titleFocused || bodyFocused || locationFocused || destFocused || colorPickerOpen || showEditPopout;
    const showPalette = isEditing || !block.color;
    const showDelete = isEditing || !block.color;

    // Click outside handler to dismiss palette and popout
    // Uses capture phase so it fires even if child components call stopPropagation
    useEffect(() => {
        if (!showPalette) return;

        const handleClickOutside = (e: MouseEvent) => {
            const target = e.target as Node;

            // Don't dismiss if clicking inside the block
            if (blockRef.current?.contains(target)) return;

            // Don't dismiss if clicking inside the popout
            if (popoutRef.current?.contains(target)) return;

            // Don't dismiss if clicking on a color swatch or palette
            const paletteEl = document.querySelector('.color-palette');
            if (paletteEl?.contains(target)) return;

            // Don't dismiss if the native color picker is open
            if (colorPickerOpen) return;

            // If no color was chosen, assign default so palette hides
            if (!block.color) {
                onUpdate({ ...block, color: '#5B7FBF' });
            }

            // Blur any active input to reset focus states
            if (document.activeElement instanceof HTMLElement) {
                document.activeElement.blur();
            }

            // Dismiss everything
            setTitleFocused(false);
            setBodyFocused(false);
            setLocationFocused(false);
            setDestFocused(false);
            setShowEditPopout(false);
        };

        // Use timeout so the click that opened the popout doesn't immediately close it
        const timer = setTimeout(() => {
            document.addEventListener('pointerdown', handleClickOutside, true);
        }, 0);

        return () => {
            clearTimeout(timer);
            document.removeEventListener('pointerdown', handleClickOutside, true);
        };
    }, [showPalette, colorPickerOpen, block, onUpdate]);

    const handleColorSelect = useCallback(
        (color: string) => {
            onUpdate({ ...block, color });
            if (!block.color) {
                setTimeout(() => {
                    if (isCompact) {
                        popoutTitleRef.current?.focus();
                    } else {
                        titleRef.current?.focus();
                    }
                }, 150);
            }
        },
        [block, onUpdate, isCompact]
    );

    const handleDelete = useCallback(() => {
        onDelete(block);
    }, [block, onDelete]);

    const bgColor = block.color || 'rgba(255,255,255,0.06)';
    const bgSubtle = block.color
        ? `${block.color}28`
        : 'rgba(255,255,255,0.04)';

    const handleTitleZoneClick = useCallback(() => {
        if (isCompact) {
            setShowEditPopout(true);
            setTimeout(() => popoutTitleRef.current?.focus(), 100);
        } else {
            titleRef.current?.focus();
        }
    }, [isCompact]);

    const handleBodyZoneClick = useCallback(() => {
        descRef.current?.focus();
    }, []);

    // Open popout on click for compact blocks
    const handleCompactClick = useCallback(() => {
        setShowEditPopout(true);
        setTimeout(() => popoutTitleRef.current?.focus(), 100);
    }, []);

    // Nominatim search helper
    const nominatimSearch = useCallback(
        (query: string, setResults: (r: NominatimResult[]) => void, setSearching: (b: boolean) => void, timeoutRef: React.MutableRefObject<ReturnType<typeof setTimeout> | null>) => {
            if (timeoutRef.current) clearTimeout(timeoutRef.current);
            if (!query || query.length < 3) {
                setResults([]);
                setSearching(false);
                return;
            }
            setSearching(true);
            timeoutRef.current = setTimeout(async () => {
                try {
                    const res = await fetch(
                        `https://nominatim.openstreetmap.org/search?format=json&q=${encodeURIComponent(query)}&limit=5`,
                        { headers: { 'Accept-Language': 'en' } }
                    );
                    const data: NominatimResult[] = await res.json();
                    setResults(data);
                } catch {
                    setResults([]);
                } finally {
                    setSearching(false);
                }
            }, 500);
        },
        []
    );

    // Start location search
    useEffect(() => {
        nominatimSearch(locationQuery, setSearchResults, setIsSearching, searchTimeoutRef);
        return () => { if (searchTimeoutRef.current) clearTimeout(searchTimeoutRef.current); };
    }, [locationQuery, nominatimSearch]);

    // Destination search
    useEffect(() => {
        nominatimSearch(destQuery, setDestResults, setIsDestSearching, destSearchTimeoutRef);
        return () => { if (destSearchTimeoutRef.current) clearTimeout(destSearchTimeoutRef.current); };
    }, [destQuery, nominatimSearch]);

    const parseLocation = useCallback((result: NominatimResult): LocationData => ({
        name: result.display_name.split(',').slice(0, 2).join(','),
        lat: parseFloat(result.lat),
        lng: parseFloat(result.lon),
    }), []);

    const handleSelectLocation = useCallback(
        (result: NominatimResult) => {
            onUpdate({ ...block, location: parseLocation(result) });
            setLocationQuery('');
            setSearchResults([]);
        },
        [block, onUpdate, parseLocation]
    );

    const handleSelectDestination = useCallback(
        (result: NominatimResult) => {
            onUpdate({ ...block, destination: parseLocation(result) });
            setDestQuery('');
            setDestResults([]);
        },
        [block, onUpdate, parseLocation]
    );

    const handleClearLocation = useCallback(() => {
        onUpdate({ ...block, location: undefined });
    }, [block, onUpdate]);

    const handleClearDestination = useCallback(() => {
        onUpdate({ ...block, destination: undefined });
    }, [block, onUpdate]);

    const blockClasses = [
        'event-block',
        showDelete ? 'active' : '',
        isHighlighted ? 'highlighted' : '',
        isCompact ? 'compact' : '',
        isHalfHour ? 'half-hour' : '',
    ].filter(Boolean).join(' ');

    // ─── Single location input helper ───
    const renderLocationInput = (
        label: string,
        icon: string,
        value: LocationData | undefined,
        query: string,
        setQuery: (q: string) => void,
        results: NominatimResult[],
        searching: boolean,
        focused: boolean,
        setFocused: (f: boolean) => void,
        onSelect: (r: NominatimResult) => void,
        onClear: () => void,
    ) => (
        <div className={`location-input-col${focused ? ' focused' : ''}`}>
            <span className="location-icon">{icon}</span>
            {value ? (
                <div className="location-display">
                    <span className="location-name">{value.name}</span>
                    <button
                        className="location-clear"
                        onClick={onClear}
                        onMouseDown={(e) => e.stopPropagation()}
                    >
                        ×
                    </button>
                </div>
            ) : (
                <input
                    className="event-block-location-input"
                    placeholder={label}
                    value={query}
                    onChange={(e) => setQuery(e.target.value)}
                    onFocus={() => setFocused(true)}
                    onBlur={() => setTimeout(() => setFocused(false), 200)}
                    onMouseDown={(e) => e.stopPropagation()}
                    onPointerDown={(e) => e.stopPropagation()}
                />
            )}
            {focused && results.length > 0 && (
                <div className="location-search-results">
                    {results.map((r, i) => (
                        <button
                            key={i}
                            className="location-result"
                            onMouseDown={(e) => {
                                e.preventDefault();
                                e.stopPropagation();
                                onSelect(r);
                            }}
                        >
                            {r.display_name}
                        </button>
                    ))}
                </div>
            )}
            {focused && searching && (
                <div className="location-search-results">
                    <div className="location-result searching">Searching...</div>
                </div>
            )}
        </div>
    );

    const handleToggleRouteMode = useCallback(() => {
        const current = block.routeMode || 'simple';
        onUpdate({ ...block, routeMode: current === 'simple' ? 'precise' : 'simple' });
    }, [block, onUpdate]);

    // ─── Location zone (shared between inline and popout) ───
    const locationZone = (
        <div className="event-block-location-zone">
            <div className="location-row">
                {renderLocationInput(
                    'Start...', '📍',
                    block.location, locationQuery, setLocationQuery,
                    searchResults, isSearching, locationFocused, setLocationFocused,
                    handleSelectLocation, handleClearLocation,
                )}
                {renderLocationInput(
                    'Destination...', '🏁',
                    block.destination, destQuery, setDestQuery,
                    destResults, isDestSearching, destFocused, setDestFocused,
                    handleSelectDestination, handleClearDestination,
                )}
                {block.location && block.destination && (
                    <button
                        className={`route-mode-toggle ${(block.routeMode || 'simple') === 'precise' ? 'precise' : ''}`}
                        onClick={(e) => { e.stopPropagation(); handleToggleRouteMode(); }}
                        onMouseDown={(e) => e.stopPropagation()}
                        onPointerDown={(e) => e.stopPropagation()}
                        title={(block.routeMode || 'simple') === 'simple' ? 'Simple route (click for precise)' : 'Precise route (click for simple)'}
                    >
                        {(block.routeMode || 'simple') === 'simple' ? (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M4 19C4 19 8 14 12 12C16 10 20 5 20 5" />
                                <circle cx="4" cy="19" r="2" />
                                <circle cx="20" cy="5" r="2" />
                            </svg>
                        ) : (
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M3 17C3 17 5 15 7 13C9 11 11 13 13 11C15 9 17 7 19 7" />
                                <polyline points="15 7 19 7 19 11" />
                                <circle cx="3" cy="17" r="2" />
                            </svg>
                        )}
                    </button>
                )}
            </div>
        </div>
    );

    // ─── Popout edit panel (for compact blocks <2hr) ───
    const popoutPanel = isCompact && showEditPopout && (
        <motion.div
            ref={popoutRef}
            className="event-popout"
            style={{
                top: topPx,
                background: block.color
                    ? `radial-gradient(ellipse at 20% 10%, ${bgColor}44, ${bgSubtle})`
                    : 'var(--card)',
            }}
            initial={{ opacity: 0, x: -8, scale: 0.97 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            exit={{ opacity: 0, x: -8, scale: 0.97 }}
            transition={{ type: 'spring', stiffness: 400, damping: 28 }}
            onPointerDown={(e) => e.stopPropagation()}
            onMouseDown={(e) => e.stopPropagation()}
        >
            {/* Title */}
            <div className="event-block-title-zone">
                <input
                    ref={popoutTitleRef}
                    className="event-block-title"
                    placeholder="Add title..."
                    value={block.title}
                    onChange={(e) => onUpdate({ ...block, title: e.target.value })}
                    onFocus={() => setTitleFocused(true)}
                    onBlur={() => setTitleFocused(false)}
                    onMouseDown={(e) => e.stopPropagation()}
                    onPointerDown={(e) => e.stopPropagation()}
                />
            </div>

            {/* Location */}
            {locationZone}

            {/* Body */}
            <div className="event-block-body-zone">
                <textarea
                    className="event-block-description"
                    placeholder="Notes..."
                    value={block.description}
                    onChange={(e) => onUpdate({ ...block, description: e.target.value })}
                    onFocus={() => setBodyFocused(true)}
                    onBlur={() => setBodyFocused(false)}
                    onMouseDown={(e) => e.stopPropagation()}
                    onPointerDown={(e) => e.stopPropagation()}
                />
            </div>
        </motion.div>
    );

    return (
        <>
            <div
                ref={blockRef}
                className={blockClasses}
                style={{
                    top: topPx,
                    height: heightPx,
                    background: block.color
                        ? `radial-gradient(ellipse at 20% 10%, ${bgColor}44, ${bgSubtle})`
                        : `var(--event-default-bg)`,
                }}
                onPointerDown={(e) => e.stopPropagation()}
                onMouseDown={(e) => e.stopPropagation()}
                onMouseEnter={() => onHoverEvent(block.id)}
                onMouseLeave={() => onHoverEvent(null)}
                onClick={isCompact ? handleCompactClick : undefined}
            >
                {/* ─── Delete button ─── */}
                <AnimatePresence>
                    {showDelete && (
                        <motion.button
                            className="event-block-delete"
                            onClick={handleDelete}
                            onPointerDown={(e) => e.stopPropagation()}
                            onMouseDown={(e) => e.stopPropagation()}
                            initial={{ opacity: 0, scale: 0.5 }}
                            animate={{ opacity: 1, scale: 1 }}
                            exit={{ opacity: 0, scale: 0.5 }}
                            transition={{ type: 'spring', stiffness: 400, damping: 25 }}
                            aria-label="Delete event"
                        >
                            ×
                        </motion.button>
                    )}
                </AnimatePresence>

                {isCompact ? (
                    /* ─── Compact: title only ─── */
                    <div className="event-block-compact-title">
                        {block.title || 'Untitled'}
                        {(block.location || block.destination) && (
                            <span className="compact-location-badge">
                                {block.location && block.destination ? '📍→🏁' : block.location ? '📍' : '🏁'}
                            </span>
                        )}
                    </div>
                ) : (
                    <>
                        {/* ─── Title zone ─── */}
                        <div
                            className={`event-block-title-zone${titleFocused ? ' focused' : ''}`}
                            onClick={handleTitleZoneClick}
                        >
                            <input
                                ref={titleRef}
                                className="event-block-title"
                                placeholder="Add title..."
                                value={block.title}
                                onChange={(e) => onUpdate({ ...block, title: e.target.value })}
                                onFocus={() => setTitleFocused(true)}
                                onBlur={() => setTitleFocused(false)}
                                onMouseDown={(e) => e.stopPropagation()}
                                onPointerDown={(e) => e.stopPropagation()}
                            />
                        </div>

                        {/* ─── Location zone ─── */}
                        {locationZone}

                        {/* ─── Body zone ─── */}
                        <div
                            className={`event-block-body-zone${bodyFocused ? ' focused' : ''}`}
                            onClick={handleBodyZoneClick}
                        >
                            <textarea
                                ref={descRef}
                                className="event-block-description"
                                placeholder="Notes..."
                                value={block.description}
                                onChange={(e) => onUpdate({ ...block, description: e.target.value })}
                                onFocus={() => setBodyFocused(true)}
                                onBlur={() => setBodyFocused(false)}
                                onMouseDown={(e) => e.stopPropagation()}
                                onPointerDown={(e) => e.stopPropagation()}
                            />
                        </div>
                    </>
                )}
            </div>

            {/* ─── Popout edit panel for compact blocks ─── */}
            <AnimatePresence>
                {popoutPanel}
            </AnimatePresence>

            <ColorPalette
                visible={showPalette}
                topPx={bottomPx}
                currentColor={block.color}
                onSelectColor={handleColorSelect}
                onColorPickerFocus={() => setColorPickerOpen(true)}
                onColorPickerBlur={() => setColorPickerOpen(false)}
            />

            {/* Full-screen overlay to block interactions while native color picker is open */}
            {colorPickerOpen && (
                <div className="color-picker-overlay" />
            )}
        </>
    );
};

export default React.memo(EventBlock);
