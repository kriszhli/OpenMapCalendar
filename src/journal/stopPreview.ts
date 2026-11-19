import L from 'leaflet';
import { dayLabel, type Stop } from './model';

// One transient card per map. Only the active stop requests a thumbnail.
export function stopPreview(map: L.Map, onSelect: (stop: Stop) => void) {
  const host = map.getContainer();
  const card = document.createElement('div');
  card.className = 'atlas-preview';
  card.hidden = true;
  host.append(card);
  L.DomEvent.disableClickPropagation(card);
  L.DomEvent.disableScrollPropagation(card);

  let active: HTMLElement | undefined;
  let country: SVGPathElement | undefined;
  let pending: number | undefined;
  const cancelClose = () => window.clearTimeout(pending);
  const close = () => {
    cancelClose();
    active?.classList.remove('preview-open');
    active?.removeAttribute('aria-describedby');
    active = undefined;
    country?.classList.remove('preview-country');
    country = undefined;
    card.hidden = true;
    card.replaceChildren();
  };
  const scheduleClose = () => {
    cancelClose();
    pending = window.setTimeout(() => {
      if (
        !card.contains(document.activeElement) &&
        !card.matches(':hover') &&
        !(active?.matches(':focus-visible') || active?.matches(':hover'))
      )
        close();
    }, 160);
  };
  card.addEventListener('pointerenter', cancelClose);
  card.addEventListener('pointerleave', scheduleClose);
  card.addEventListener('focusout', scheduleClose);
  const escape = (e: KeyboardEvent) => {
    if (e.key === 'Escape' && active) {
      if (card.contains(document.activeElement)) active.focus();
      close();
    }
  };
  document.addEventListener('keydown', escape);
  map.on('click movestart resize', close);

  function bind(marker: L.Marker, stop: Stop, overview: boolean) {
    const pin = marker.getElement()!;
    pin.setAttribute('aria-label', stop.place.name + ', ' + stop.time + '. Open itinerary');
    const show = () => {
      cancelClose();
      if (active === pin) return;
      close();
      active = pin;
      pin.classList.add('preview-open');
      const point = map.latLngToContainerPoint(marker.getLatLng());
      const size = map.getSize();
      if (overview) {
        const bounds = host.getBoundingClientRect();
        const screenPoint = new DOMPoint(bounds.left + point.x, bounds.top + point.y);
        for (const path of host.querySelectorAll<SVGPathElement>('.atlas-coastline')) {
          const matrix = path.getScreenCTM();
          if (matrix && path.isPointInFill(screenPoint.matrixTransform(matrix.inverse()))) {
            country = path;
            path.classList.add('preview-country');
            break;
          }
        }
      }
      const photo = stop.photos[0];
      if (photo) {
        const img = document.createElement('img');
        img.alt = '';
        img.src = photo.thumb;
        card.append(img);
      }
      const name = document.createElement('strong');
      name.textContent = stop.place.name;
      const time = document.createElement('span');
      time.id = 'atlas-preview-date-' + L.Util.stamp(map);
      time.textContent =
        dayLabel(stop.trip, stop.event.dayIndex, { month: 'short', day: 'numeric', year: 'numeric' }) +
        ' · ' +
        stop.time;
      pin.setAttribute('aria-describedby', time.id);
      const open = document.createElement('button');
      open.type = 'button';
      open.textContent = 'Open itinerary →';
      open.onclick = () => onSelect(stop);
      card.append(name, time, open);
      card.hidden = false;
      const left = Math.max(8, Math.min(point.x - card.offsetWidth / 2, size.x - card.offsetWidth - 8));
      const above = point.y - card.offsetHeight - 14;
      const top = above >= 8 ? above : Math.max(8, Math.min(point.y + 14, size.y - card.offsetHeight - 8));
      card.style.left = left + 'px';
      card.style.top = top + 'px';
    };
    let pointerType = '';
    pin.addEventListener('pointerdown', (e) => {
      pointerType = e.pointerType;
    });
    pin.addEventListener('pointerenter', (e) => {
      if (e.pointerType !== 'touch') show();
    });
    pin.addEventListener('pointerleave', scheduleClose);
    pin.addEventListener('focus', show);
    pin.addEventListener('blur', scheduleClose);
    pin.addEventListener('keydown', (e) => {
      if (e.key === ' ' || e.key === 'Enter') {
        e.preventDefault();
        e.stopPropagation();
        onSelect(stop);
      }
    });
    // Touch users open the card first and choose its itinerary action.
    marker.on('click', () => {
      if (pointerType === 'mouse') onSelect(stop);
      else show();
    });
  }
  return {
    bind,
    remove() {
      close();
      card.remove();
      document.removeEventListener('keydown', escape);
      map.off('click movestart resize', close);
    },
  };
}
