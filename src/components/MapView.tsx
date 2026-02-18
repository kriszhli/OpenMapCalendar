import React, { useEffect, useRef, useMemo, useCallback } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import type { TimeBlock } from '../types';
import { minutesToTime } from '../types';
import './MapView.css';

interface MapViewProps {
    events: Record<number, TimeBlock[]>;
    hoveredEventId: string | null;
    onHoverEvent: (id: string | null) => void;
    preciseZoomEnabled?: boolean;
}

const DEFAULT_CENTER: L.LatLngTuple = [30, 0];
const DEFAULT_ZOOM = 2;

const escapeHtml = (value: string): string =>
    value
        .replaceAll('&', '&amp;')
        .replaceAll('<', '&lt;')
        .replaceAll('>', '&gt;')
        .replaceAll('"', '&quot;')
        .replaceAll("'", '&#39;');

// ─── OSRM route fetching ───
async function fetchRoute(
    startLat: number, startLng: number,
    endLat: number, endLng: number,
): Promise<L.LatLngTuple[] | null> {
    try {
        const url = `https://router.project-osrm.org/route/v1/driving/${startLng},${startLat};${endLng},${endLat}?overview=full&geometries=geojson`;
        const res = await fetch(url);
        const data = await res.json();
        if (data.code === 'Ok' && data.routes?.[0]?.geometry?.coordinates) {
            return data.routes[0].geometry.coordinates.map(
                ([lng, lat]: [number, number]) => [lat, lng] as L.LatLngTuple
            );
        }
    } catch {
        // Silently fail — no route drawn
    }
    return null;
}

const MapView: React.FC<MapViewProps> = ({ events, hoveredEventId, onHoverEvent, preciseZoomEnabled = false }) => {
    const mapRef = useRef<L.Map | null>(null);
    const containerRef = useRef<HTMLDivElement>(null);
    const markersRef = useRef<Map<string, L.CircleMarker[]>>(new Map());
    const glowMarkersRef = useRef<Map<string, L.CircleMarker[]>>(new Map());
    const routesRef = useRef<Map<string, L.Polyline>>(new Map());

    // Group located events by day and sort by start time.
    const dayLocatedEvents = useMemo(() => {
        return Object.entries(events)
            .map(([dayIndex, dayEvents]) => ({
                dayIndex: Number(dayIndex),
                events: [...dayEvents]
                    .filter((ev) => ev.location || ev.destination)
                    .sort((a, b) => a.startMinutes - b.startMinutes),
            }))
            .filter((group) => group.events.length > 0)
            .sort((a, b) => a.dayIndex - b.dayIndex);
    }, [events]);

    // Initialize map
    useEffect(() => {
        if (!containerRef.current || mapRef.current) return;

        const map = L.map(containerRef.current, {
            center: DEFAULT_CENTER,
            zoom: DEFAULT_ZOOM,
            zoomControl: true,
            attributionControl: true,
            scrollWheelZoom: false,
            touchZoom: false,
            zoomSnap: 1,
            zoomDelta: 1,
        });

        const updateTileLayers = () => {
            const theme = document.documentElement.getAttribute('data-theme') || 'dark';
            const isBright = theme === 'bright';

            // Base layer
            const baseUrl = isBright
                ? 'https://{s}.basemaps.cartocdn.com/light_nolabels/{z}/{x}/{y}{r}.png'
                : 'https://{s}.basemaps.cartocdn.com/dark_nolabels/{z}/{x}/{y}{r}.png';

            // Labels layer
            const labelsUrl = isBright
                ? 'https://{s}.basemaps.cartocdn.com/light_only_labels/{z}/{x}/{y}{r}.png'
                : 'https://{s}.basemaps.cartocdn.com/dark_only_labels/{z}/{x}/{y}{r}.png';

            map.eachLayer((layer) => {
                if (layer instanceof L.TileLayer) {
                    map.removeLayer(layer);
                }
            });

            L.tileLayer(baseUrl, {
                attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>',
                maxZoom: 20,
            }).addTo(map);

            L.tileLayer(labelsUrl, {
                maxZoom: 20,
            }).addTo(map);
        };

        // Initial load
        updateTileLayers();

        // Listen for theme changes
        const observer = new MutationObserver((mutations) => {
            mutations.forEach((mutation) => {
                if (mutation.type === 'attributes' && mutation.attributeName === 'data-theme') {
                    updateTileLayers();
                }
            });
        });

        observer.observe(document.documentElement, {
            attributes: true,
            attributeFilter: ['data-theme'],
        });

        mapRef.current = map;

        return () => {
            observer.disconnect();
            map.remove();
            mapRef.current = null;
        };
    }, []);

    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        if (preciseZoomEnabled) {
            map.scrollWheelZoom.enable();
            map.touchZoom.enable();
            map.options.zoomSnap = 0.25;
            map.options.zoomDelta = 0.25;
        } else {
            map.scrollWheelZoom.disable();
            map.touchZoom.disable();
            map.options.zoomSnap = 1;
            map.options.zoomDelta = 1;
        }
    }, [preciseZoomEnabled]);

    // Handle hover from calendar → highlight pins + route on map
    const handleCalendarHover = useCallback((eventId: string | null) => {
        // Reset all glow markers
        glowMarkersRef.current.forEach((glows) => {
            glows.forEach((g) => g.setStyle({ opacity: 0, fillOpacity: 0 }));
        });
        // Reset all routes to normal
        routesRef.current.forEach((route) => {
            route.setStyle({ weight: 3, opacity: 0.4 });
        });

        if (eventId) {
            const glows = glowMarkersRef.current.get(eventId);
            if (glows) {
                glows.forEach((g) => g.setStyle({ opacity: 0.6, fillOpacity: 0.3 }));
            }
            const route = routesRef.current.get(eventId);
            if (route) {
                route.setStyle({ weight: 4, opacity: 0.7 });
                route.bringToFront();
            }
        }
    }, []);

    useEffect(() => {
        handleCalendarHover(hoveredEventId);
    }, [hoveredEventId, handleCalendarHover]);

    // Sync markers + routes with events
    useEffect(() => {
        const map = mapRef.current;
        if (!map) return;

        let isCancelled = false;

        // Remove old markers & routes
        markersRef.current.forEach((ms) => ms.forEach((m) => m.remove()));
        glowMarkersRef.current.forEach((ms) => ms.forEach((m) => m.remove()));
        routesRef.current.forEach((r) => r.remove());
        markersRef.current.clear();
        glowMarkersRef.current.clear();
        routesRef.current.clear();

        const bounds: L.LatLng[] = [];

        // Helper to create a pin pair (main + glow)
        const createPin = (
            latlng: L.LatLng,
            color: string,
            eventId: string,
            tooltipText: string,
        ): { marker: L.CircleMarker; glow: L.CircleMarker } => {
            const glow = L.circleMarker(latlng, {
                radius: 20,
                color,
                fillColor: color,
                fillOpacity: 0,
                opacity: 0,
                weight: 2,
                className: 'map-pin-highlight',
            }).addTo(map);

            const marker = L.circleMarker(latlng, {
                radius: 8,
                color: '#fff',
                fillColor: color,
                fillOpacity: 1,
                weight: 1.5,
                opacity: 0.5,
            }).addTo(map);

            marker.bindTooltip(
                `<span class="map-tooltip-label" style="color: ${color}">${escapeHtml(tooltipText)}</span>`,
                {
                permanent: true,
                direction: 'top',
                offset: [0, -12],
                className: 'map-tooltip',
                }
            );

            marker.on('mouseover', () => {
                onHoverEvent(eventId);
                marker.setStyle({ fillOpacity: 1, radius: 11, opacity: 1 });
                glow.setStyle({ opacity: 0.6, fillOpacity: 0.3 });
            });
            marker.on('mouseout', () => {
                onHoverEvent(null);
                marker.setStyle({ fillOpacity: 1, radius: 8, opacity: 0.5 });
                glow.setStyle({ opacity: 0, fillOpacity: 0 });
            });

            return { marker, glow };
        };

        const drawRoute = (
            ev: TimeBlock,
            color: string,
            mode: 'simple' | 'precise',
            startLatLng: L.LatLng,
            endLatLng: L.LatLng
        ) => {
            if (mode === 'precise') {
                fetchRoute(
                    startLatLng.lat, startLatLng.lng,
                    endLatLng.lat, endLatLng.lng,
                ).then((coords) => {
                    if (isCancelled || !coords || !mapRef.current) return;
                    const polyline = L.polyline(coords, {
                        color,
                        weight: 3,
                        opacity: 0.5,
                        smoothFactor: 1,
                        className: 'map-route-line',
                    }).addTo(mapRef.current);

                    polyline.on('mouseover', () => {
                        onHoverEvent(ev.id);
                        polyline.setStyle({ weight: 4, opacity: 0.7 });
                        polyline.bringToFront();
                    });
                    polyline.on('mouseout', () => {
                        onHoverEvent(null);
                        polyline.setStyle({ weight: 3, opacity: 0.5 });
                    });

                    routesRef.current.set(ev.id, polyline);
                });
                return;
            }

            const midLat = (startLatLng.lat + endLatLng.lat) / 2;
            const midLng = (startLatLng.lng + endLatLng.lng) / 2;
            const dLat = endLatLng.lat - startLatLng.lat;
            const dLng = endLatLng.lng - startLatLng.lng;
            const dist = Math.sqrt(dLat * dLat + dLng * dLng) || 0.000001;
            const offset = dist * 0.2;
            const ctrlLat = midLat + (-dLng / dist) * offset;
            const ctrlLng = midLng + (dLat / dist) * offset;

            const NUM_POINTS = 30;
            const curvePoints: L.LatLngTuple[] = [];
            for (let i = 0; i <= NUM_POINTS; i++) {
                const t = i / NUM_POINTS;
                const invT = 1 - t;
                const lat = invT * invT * startLatLng.lat + 2 * invT * t * ctrlLat + t * t * endLatLng.lat;
                const lng = invT * invT * startLatLng.lng + 2 * invT * t * ctrlLng + t * t * endLatLng.lng;
                curvePoints.push([lat, lng]);
            }

            const polyline = L.polyline(curvePoints, {
                color,
                weight: 3,
                opacity: 0.5,
                smoothFactor: 1,
                dashArray: '8, 6',
                className: 'map-route-line',
            }).addTo(map);

            polyline.on('mouseover', () => {
                onHoverEvent(ev.id);
                polyline.setStyle({ weight: 4, opacity: 0.7 });
                polyline.bringToFront();
            });
            polyline.on('mouseout', () => {
                onHoverEvent(null);
                polyline.setStyle({ weight: 3, opacity: 0.5 });
            });

            routesRef.current.set(ev.id, polyline);
        };

        dayLocatedEvents.forEach(({ events: dayEvents }) => {
            dayEvents.forEach((ev, index) => {
                const color = ev.color || '#5B7FBF';
                const eventMarkers: L.CircleMarker[] = [];
                const eventGlows: L.CircleMarker[] = [];
                const startTimeLabel = minutesToTime(ev.startMinutes);
                const endTimeLabel = minutesToTime(ev.endMinutes);
                const eventTitle = (ev.title || 'Untitled').trim();

                if (ev.location) {
                    const latlng = L.latLng(ev.location.lat, ev.location.lng);
                    bounds.push(latlng);
                    const { marker, glow } = createPin(
                        latlng,
                        color,
                        ev.id,
                        `${startTimeLabel} | ${eventTitle}`
                    );
                    eventMarkers.push(marker);
                    eventGlows.push(glow);
                }

                if (ev.destination) {
                    const latlng = L.latLng(ev.destination.lat, ev.destination.lng);
                    bounds.push(latlng);
                    const { marker, glow } = createPin(
                        latlng,
                        color,
                        ev.id,
                        `${endTimeLabel} | ${eventTitle}`
                    );
                    eventMarkers.push(marker);
                    eventGlows.push(glow);
                }

                markersRef.current.set(ev.id, eventMarkers);
                glowMarkersRef.current.set(ev.id, eventGlows);

                const mode = ev.routeMode || 'simple';
                if (mode === 'hidden') return;

                if (ev.location && ev.destination) {
                    const startLatLng = L.latLng(ev.location.lat, ev.location.lng);
                    const endLatLng = L.latLng(ev.destination.lat, ev.destination.lng);
                    drawRoute(ev, color, mode, startLatLng, endLatLng);
                    return;
                }

                const currentEndpoint = ev.destination ?? ev.location;
                const previousEvent = dayEvents[index - 1];
                const previousEndpoint = previousEvent?.destination ?? previousEvent?.location;
                if (!currentEndpoint || !previousEndpoint) return;

                const startLatLng = L.latLng(previousEndpoint.lat, previousEndpoint.lng);
                const endLatLng = L.latLng(currentEndpoint.lat, currentEndpoint.lng);
                drawRoute(ev, color, mode, startLatLng, endLatLng);
            });
        });

        // Fit bounds
        if (bounds.length > 0) {
            if (bounds.length === 1) {
                map.setView(bounds[0], 12);
            } else {
                map.fitBounds(L.latLngBounds(bounds), { padding: [60, 60], maxZoom: 14 });
            }
        }

        return () => {
            isCancelled = true;
        };
    }, [dayLocatedEvents, onHoverEvent]);

    const hasLocations = dayLocatedEvents.length > 0;

    return (
        <div className="map-section">
            <div ref={containerRef} className="map-container" />
            {!hasLocations && (
                <div className="map-empty-state">
                    <div className="map-empty-text">
                        Add locations to your events to see them on the map
                    </div>
                </div>
            )}
        </div>
    );
};

export default React.memo(MapView);
