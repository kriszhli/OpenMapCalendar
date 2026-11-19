import express from 'express';

const provider = 'https://basemaps.cartocdn.com';
const assetOrigins = new Set([provider, 'https://tiles.basemaps.cartocdn.com']);
const prefix = '/api/maps/carto';
const rasterLayers = new Set(['light_nolabels', 'dark_nolabels', 'light_only_labels', 'dark_only_labels', 'voyager']);
const tilePattern = /^(\d{1,2})\/(\d{1,7})\/(\d{1,7})(@2x)?\.png$/;
const version = '(?:/v/[A-Za-z0-9_-]+)?';
const allowedAsset = new RegExp('^' + version + '(?:/vectortiles/carto\\.streets/v2/\\d{1,2}/\\d{1,7}/\\d{1,7}\\.mvt|/fonts/[A-Za-z0-9 ,_.+-]{1,200}/\\d+-\\d+\\.pbf|/gl/voyager-gl-style/sprite(?:@2x)?\\.(?:png|json))$');

function localAsset(raw) {
  const url = new URL(raw);
  if (!assetOrigins.has(url.origin)) throw new Error('Unexpected asset provider');
  // Keep style placeholders intact and omit all upstream query parameters.
  return prefix + '/assets' + url.pathname.replace(/%7B/gi, '{').replace(/%7D/gi, '}');
}

export function cartoRouter({ key, fetchUpstream = fetch }) {
  const router = express.Router();
  router.use((req, res, next) => {
    if (req.get('Sec-Fetch-Site') === 'cross-site') return res.sendStatus(403);
    next();
  });

  async function serve(path, res, { asset = false, style = false } = {}) {
    const upstream = new URL((asset ? 'https://tiles.basemaps.cartocdn.com' : provider) + path);
    if (key) upstream.searchParams.set('key', key);
    try {
      const response = await fetchUpstream(upstream, { redirect: 'error', signal: AbortSignal.timeout(10000) });
      // Provider errors and headers can contain the authenticated request URL.
      if (!response.ok) {
        await response.body?.cancel();
        return res.status(response.status === 429 ? 429 : 502).json({ error: 'CARTO map resource unavailable' });
      }
      const cache = response.headers.get('cache-control') || 'private, max-age=3600';
      if (style) {
        const spec = await response.json();
        for (const source of Object.values(spec.sources)) {
          if (source.url || !Array.isArray(source.tiles)) throw new Error('Unexpected style source');
          source.tiles = source.tiles.map(localAsset);
        }
        spec.sprite = localAsset(spec.sprite);
        spec.glyphs = localAsset(spec.glyphs);
        const body = JSON.stringify(spec);
        if (key && body.includes(key)) throw new Error('Unexpected credential in style');
        return res.set('Cache-Control', cache).type('json').send(body);
      }
      const bytes = Buffer.from(await response.arrayBuffer());
      const type = path.endsWith('.png') ? 'image/png' : path.endsWith('.json') ? 'application/json' : 'application/octet-stream';
      if (key && bytes.includes(key)) throw new Error('Unexpected credential in resource');
      res.set('Cache-Control', cache).type(type).send(bytes);
    } catch {
      res.status(502).json({ error: 'CARTO map resource unavailable' });
    }
  }

  router.get('/voyager/style.json', (_req, res) => serve('/gl/voyager-gl-style/style.json', res, { style: true }));
  router.get(/^\/tiles\/([^/]+)\/(.+)$/, (req, res) => {
    const [layer, tile] = [req.params[0], req.params[1]];
    const match = tilePattern.exec(tile);
    if (!rasterLayers.has(layer) || !match || +match[1] > 20 || +match[2] >= 2 ** +match[1] || +match[3] >= 2 ** +match[1]) return res.sendStatus(404);
    return serve('/' + (layer === 'voyager' ? 'rastertiles/voyager' : layer) + '/' + tile, res);
  });
  router.get(/^\/assets(\/.+)$/, (req, res) => {
    const path = req.params[0];
    if (!allowedAsset.test(path)) return res.sendStatus(404);
    return serve(path, res, { asset: true });
  });
  return router;
}
