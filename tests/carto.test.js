import assert from 'node:assert/strict';
import { once } from 'node:events';
import test from 'node:test';
import express from 'express';
import { cartoRouter } from '../lib/carto.js';

test('CARTO credentials stay upstream, including nested styles and provider failures', async t => {
  const key = 'test-only-credential';
  const calls = [];
  const app = express();
  const asset = 'https://tiles.basemaps.cartocdn.com/v/1';
  app.use('/api/maps/carto', cartoRouter({ key, fetchUpstream: async url => {
    calls.push(url);
    assert.equal(url.searchParams.get('key'), key);
    if (!url.pathname.endsWith('style.json')) return new Response(url.href, { status: 403 });
    return Response.json({
      sources: { streets: { tiles: [asset + '/vectortiles/carto.streets/v2/{z}/{x}/{y}.mvt?key=' + key] } },
      sprite: asset + '/gl/voyager-gl-style/sprite?key=' + key,
      glyphs: asset + '/fonts/{fontstack}/{range}.pbf?key=' + key,
    });
  } }));
  const server = app.listen(0, '127.0.0.1');
  await once(server, 'listening');
  t.after(() => server.close());
  const origin = 'http://127.0.0.1:' + server.address().port;
  const result = await fetch(origin+'/api/maps/carto/voyager/style.json');
  const style = await result.json();
  assert.equal(result.status, 200);
  assert.ok(!JSON.stringify(style).includes(key));
  assert.match(style.glyphs, /^\/api\/maps\/carto\/assets\/v\/1\/fonts\/\{fontstack\}/);
  assert.ok(style.sources.streets.tiles[0].startsWith('/api/maps/carto/assets/'));
  const failed = await fetch(origin + style.sprite + '.png');
  assert.equal(failed.status, 502);
  assert.ok(!(await failed.text()).includes(key));
  assert.equal((await fetch(origin+'/api/maps/carto/tiles/voyager/0/9/9.png')).status,404);
  assert.equal(calls.length,2);
});
