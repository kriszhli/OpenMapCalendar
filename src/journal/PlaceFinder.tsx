import { useState } from 'react';
import { Search } from 'lucide-react';

interface Result {
  display_name: string;
  lat: string;
  lon: string;
}
export default function PlaceFinder({ onPick }: { onPick: (p: { lat: number; lng: number }) => void }) {
  const [name, setName] = useState('');
  const [results, setResults] = useState<Result[]>([]);
  const [status, setStatus] = useState('');
  const [busy, setBusy] = useState(false);
  async function search() {
    setBusy(true);
    setStatus('');
    setResults([]);
    try {
      const params = new URLSearchParams({ q: name.trim(), format: 'jsonv2', limit: '4' });
      const res = await fetch('https://nominatim.openstreetmap.org/search?' + params, {
        signal: AbortSignal.timeout(7000),
      });
      if (!res.ok) throw new Error();
      const data: Result[] = await res.json();
      setResults(data);
      if (!data.length) setStatus('No places found. Try a city or pick the map.');
    } catch {
      setStatus('Place search is unavailable. Pick the map or enter coordinates.');
    } finally {
      setBusy(false);
    }
  }
  return (
    <div className="place-finder">
      <label>
        Place name
        <input
          name="place"
          placeholder="Museum or street address"
          required
          value={name}
          onChange={(e) => {
            setName(e.target.value);
            setResults([]);
            setStatus('');
          }}
        />
      </label>
      <button type="button" disabled={!name.trim() || busy} onClick={() => void search()}>
        <Search />
        {busy ? 'Finding places…' : 'Find on map'}
      </button>
      {results.map((r) => (
        <button
          type="button"
          key={r.lat + ',' + r.lon + r.display_name}
          onClick={() => {
            setName(r.display_name.split(',')[0]);
            onPick({ lat: Number(r.lat), lng: Number(r.lon) });
            setResults([]);
          }}
        >
          {r.display_name}
        </button>
      ))}
      {status && <p role="status">{status}</p>}
    </div>
  );
}
