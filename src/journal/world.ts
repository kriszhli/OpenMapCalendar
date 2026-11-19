import { feature } from 'topojson-client';
import type { Topology, GeometryCollection } from 'topojson-specification';
import type { MultiPolygon, Polygon } from 'geojson';
import world from 'world-atlas/countries-110m.json';

const topology = world as unknown as Topology<{ countries: GeometryCollection }>;
export const countries = feature(topology, topology.objects.countries);

// Natural Earth joins rings at the date line on a sphere. Leaflet draws on a plane,
// so unwrap each ring before drawing it, including a copy at the opposite map edge.
for (const item of countries.features) {
  const geometry = item.geometry as Polygon | MultiPolygon;
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  const pieces = polygons.flatMap((rings) => {
    const ring = rings[0];
    if (!ring.some(([lng], i) => i > 0 && Math.abs(lng - ring[i - 1][0]) > 180)) return [rings];
    const unwrapped = rings.map((points) => {
      let previous = points[0][0];
      const coords = points.map(([lng, lat]) => {
        const x = lng + 360 * Math.round((previous - lng) / 360);
        previous = x;
        return [x, lat];
      });
      if (Math.abs(coords[0][0] - previous) > 180)
        coords.push([previous, -85], [coords[0][0], -85], coords[0]);
      return coords;
    });
    const xs = unwrapped[0].map((c) => c[0]);
    return [-360, 0, 360]
      .filter((dx) => Math.min(...xs) + dx < 180 && Math.max(...xs) + dx > -180)
      .map((dx) => unwrapped.map((points) => points.map(([lng, lat]) => [lng + dx, lat])));
  });
  item.geometry = { type: 'MultiPolygon', coordinates: pieces };
}
