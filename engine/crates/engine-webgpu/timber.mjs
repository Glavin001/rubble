// The timber yard: Shardwright's timber-frame house and two plank walls, free fire or a raze
// (engine-world-wasm `timber_yard`), on the range page (fracture/range-page.mjs).
import {rangePage} from './fracture/range-page.mjs';

await rangePage({
 scene:1,
 asset:'timber-yard.glb',
 loading:'Loading the timber yard…',
 lighter:'the lighter yard',
 prefs:'timber-hud',
 // The raze has the director's orbit; free fire starts on the page's own camera.
 director:take=>take===1,
 view:[[-9,4.5,-16],[-1,1.8,1.5]],
});
