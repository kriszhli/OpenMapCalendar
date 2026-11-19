import { addCalendarBasemap } from '../map/basemap';
import { useCalendarTheme } from '../useCalendarTheme';
import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { countries } from './world';
import 'leaflet/dist/leaflet.css';
import { Crosshair, Minus, Plus, Moon, Sun, Layers } from 'lucide-react';
import { connections, type Stop } from './model';
import type { LocationData } from '../types';

const worldCenter: L.LatLngTuple = [24, 8];

interface Props {
  points: Stop[];
  tripId: string | null;
  focus: Stop | null;
  onSelect: (stop: Stop) => void;
  onWorld: () => void;
  onPick?: (place: Pick<LocationData, 'lat' | 'lng'>) => void;
  picked?: Pick<LocationData, 'lat' | 'lng'> | null;
}
function label(tag: string, text: string, cls?: string) {
  const el = document.createElement(tag);
  el.textContent = text;
  if (cls) el.className = cls;
  return el;
}

export default function AtlasMap({ points, tripId, focus, onSelect, onWorld, onPick, picked }: Props) {
  const { theme, toggleTheme } = useCalendarTheme();
  const host = useRef<HTMLDivElement>(null);
  const ref = useRef<L.Map | null>(null);
  const callbacks = useRef({ onSelect, onWorld, onPick });
  const [zoom, setZoom] = useState(2);
  const [streets, setStreets] = useState(false);
  const picking = !!onPick;
  const overview = !tripId && zoom < 5 && !streets;
  const [fitKey, setFitKey] = useState(0);
  const fitRef = useRef('');
  useEffect(() => {
    callbacks.current = { onSelect, onWorld, onPick };
  }, [onSelect, onWorld, onPick]);
  useEffect(() => {
    if (!host.current) return;
    const map = L.map(host.current, {
      zoomControl: false,
      minZoom: 0,
      maxZoom: 18,
      zoomSnap: 0.25,
      zoomDelta: 1,
      worldCopyJump: false,
      scrollWheelZoom: true,
      attributionControl: true,
    }).setView(worldCenter, 2);
    ref.current = map;
    map.attributionControl.setPrefix(false);
    L.control.scale({ position: 'bottomleft', imperial: false, maxWidth: 85 }).addTo(map);
    // The bundled coastline stays underneath the tiles if the device is offline.
    const fallback = map.createPane('atlasOffline');
    fallback.style.zIndex = '190';
    fallback.style.pointerEvents = 'none';
    L.geoJSON(countries, {
      pane: 'atlasOffline',
      interactive: false,
      style: feature => ({
        className: feature?.properties?.name === 'Antarctica' ? 'atlas-coastline atlas-antarctica' : 'atlas-coastline',
        weight: 0.65,
        fillOpacity: 1,
      }),
      attribution: 'Natural Earth',
    }).addTo(map);
    map.createPane('atlasRoutes').style.zIndex = '420';
    map.on('zoomend', () => setZoom(map.getZoom()));
    map.on('click', (e: L.LeafletMouseEvent) =>
      callbacks.current.onPick?.({ lat: e.latlng.lat, lng: e.latlng.wrap().lng }),
    );
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(host.current);
    return () => {
      observer.disconnect();
      map.remove();
      ref.current = null;
      fitRef.current = '';
    };
  }, []);
  useEffect(() => {
    if (!overview && ref.current) return addCalendarBasemap(ref.current);
  }, [overview]);
  // Fit on navigation or explicit recenter only. Refreshes and uploads preserve manual zoom.
  useEffect(() => {
    const map = ref.current;
    if (!map) return;
    const key = [tripId, fitKey, points.length > 0].join('/');
    if (key === fitRef.current) return;
    fitRef.current = key;
    if (points.length) {
      const size = map.getSize();
      const pad = tripId ? Math.min(90, size.x / 7) : size.x * 0.15;
      map.fitBounds(L.latLngBounds(points.map((s) => [s.place.lat, s.place.lng])), {
        paddingTopLeft: [pad, 48],
        paddingBottomRight: [pad, 60],
        maxZoom: tripId ? 15 : 3,
        animate: false,
      });
    } else map.setView(worldCenter, 2, { animate: false });
  }, [tripId, points, fitKey]);
  useEffect(() => {
    if (focus && ref.current)
      ref.current.setView([focus.place.lat, focus.place.lng], Math.max(13, ref.current.getZoom()), {
        animate: false,
      });
  }, [focus]);
  useEffect(() => {
    if (!picked || !ref.current) return;
    const map = ref.current;
    if (!map.getBounds().contains(picked)) map.setView(picked, 13, { animate: false });
  }, [picked]);
  useEffect(() => {
    const map = ref.current;
    if (!map || !tripId || onPick) return;
    const leave = () => {
      if (map.getZoom() <= 3) callbacks.current.onWorld();
    };
    map.on('zoomend', leave);
    return () => {
      map.off('zoomend', leave);
    };
  }, [tripId, onPick]);
  useEffect(() => {
    const map = ref.current;
    if (!map) return;
    const layer = L.layerGroup().addTo(map);
    const trips = new Set(points.map((s) => s.trip));
    for (const trip of trips) {
      const path = points.filter((p) => p.trip === trip);
      for (const leg of connections(path))
        if (leg.to.event.routeMode !== 'hidden') {
          L.polyline(leg.geometry, {
            pane: 'atlasRoutes',
            color: overview ? '#8f706e' : '#427a60',
            weight: overview ? 1.5 : 3,
            opacity: overview ? 0.55 : 0.85,
            dashArray: leg.geometry.length === 2 ? '5 6' : undefined,
          })
            .addTo(layer)
            .bindTooltip(
              label('span', leg.depart + ' → ' + leg.arrive + ' · ' + (leg.mode || 'Connection')),
            );
        }
    }
    // Group overlapping stops at this zoom. Every individual stop is also reachable in the itinerary.
    const clusters: Stop[][] = [];
    for (const p of points) {
      const px = map.project([p.place.lat, p.place.lng], zoom);
      const match = clusters.find((group) => {
        const other = group[0];
        const offset = px.subtract(map.project([other.place.lat, other.place.lng], zoom));
        if (map.distance(p.place, other.place) > 120_000) return false;
        return Math.abs(offset.x) < 30 && Math.abs(offset.y) < 24;
      });
      if (match) match.push(p);
      else clusters.push([p]);
    }
    for (const group of clusters) {
      const p = group.find((s) => s.key === focus?.key) || group.find((s) => s.photos.length) || group[0];
      const dot = !tripId;
      const marker = label('div', '', dot ? 'atlas-dot' : 'atlas-stop-marker');
      if (!dot) {
        marker.append(label('b', p.time), label('span', p.place.name));
        if (group.length > 1) marker.append(label('small', String(group.length), 'atlas-cluster-count'));
      }
      const pin = L.marker([p.place.lat, p.place.lng], {
        icon: L.divIcon({
          className: dot ? 'atlas-dot-target' : '',
          html: marker,
          iconSize: dot ? [32, 32] : [116, 34],
          iconAnchor: dot ? [16, 16] : [5, 17],
        }),
        keyboard: true,
        riseOnHover: true,
      }).addTo(layer);
      pin.getElement()?.setAttribute('aria-label', p.place.name + ', ' + p.time);
      pin.on('click', () => {
        if (picking) callbacks.current.onPick?.(p.place);
        else callbacks.current.onSelect(p);
      });
    }
    if (picked)
      L.circleMarker([picked.lat, picked.lng], {
        pane: 'atlasRoutes',
        radius: 8,
        color: '#743b2a',
        fillColor: '#eb7548',
        fillOpacity: 1,
      }).addTo(layer);
    return () => {
      map.removeLayer(layer);
    };
  }, [points, zoom, tripId, focus, picked, overview, picking]);
  return (
    <div className={'atlas-canvas' + (onPick ? ' picking' : '') + (overview ? ' atlas-overview' : '')}>
      <div ref={host} className="atlas-map calendar-basemap" aria-label="Interactive travel map" />
      <div className="atlas-map-tools">
        {!tripId && zoom < 5 && (
          <button aria-label="Street map" title="Street map" aria-pressed={streets} onClick={() => setStreets(!streets)}>
            <Layers />
          </button>
        )}
        {!overview && <button aria-label={theme === 'bright' ? 'Use dark map' : 'Use light map'} title={theme === 'bright' ? 'Use dark map' : 'Use light map'} onClick={toggleTheme}>
          {theme === 'bright' ? <Moon /> : <Sun />}
        </button>}
        <button aria-label="Zoom in" onClick={() => ref.current?.zoomIn()}>
          <Plus />
        </button>
        <button aria-label="Zoom out" onClick={() => ref.current?.zoomOut()}>
          <Minus />
        </button>
        <button aria-label="Fit all visible stops" onClick={() => setFitKey((n) => n + 1)}>
          <Crosshair />
        </button>
      </div>
      {onPick && <span className="map-pick-hint">Click the map to place your stop</span>}
    </div>
  );
}
