import L from 'leaflet';
import './basemap.css';

const attribution = '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>';

// Both map views use the original CARTO base and transparent place-label tiles.
export function addCalendarBasemap(map: L.Map) {
  const style = () => document.documentElement.dataset.theme === 'bright' ? 'light' : 'dark';
  const url = (theme: string, layer: string) =>
    `/api/maps/carto/tiles/${theme}_${layer}/{z}/{x}/{y}{r}.png`;
  const original = L.layerGroup();
  const voyager = L.tileLayer('/api/maps/carto/tiles/voyager/{z}/{x}/{y}{r}.png', { attribution, maxZoom: 20 });
  let theme = style();
  function setOriginalTiles() {
    original.clearLayers();
    original.addLayer(L.tileLayer(url(theme, 'nolabels'), { attribution, maxZoom: 20, zIndex: 1 }));
    original.addLayer(L.tileLayer(url(theme, 'only_labels'), { maxZoom: 20, zIndex: 2 }));
  }
  setOriginalTiles();
  const useVoyager = localStorage.getItem('calendar-basemap') === 'voyager';
  (useVoyager ? voyager : original).addTo(map);
  map.getContainer().classList.toggle('voyager-basemap', useVoyager);
  const control = L.control.layers({ 'Original CARTO': original, 'Voyager': voyager }, {}, { position: 'topleft' }).addTo(map);
  const changeLayer = (event: L.LayersControlEvent) => {
    const selected = event.layer === voyager;
    localStorage.setItem('calendar-basemap', selected ? 'voyager' : 'original');
    map.getContainer().classList.toggle('voyager-basemap', selected);
  };
  map.on('baselayerchange', changeLayer);
  const observer = new MutationObserver(() => {
    const next = style();
    if (next === theme) return;
    theme = next;
    setOriginalTiles();
  });
  observer.observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
  const disconnect = () => observer.disconnect();
  map.once('unload', disconnect);
  return () => {
    disconnect();
    map.off('unload', disconnect);
    map.off('baselayerchange', changeLayer);
    control.remove();
    original.remove();
    voyager.remove();
    map.getContainer().classList.remove('voyager-basemap');
  };
}
