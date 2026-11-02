// Broadside: the galleon stern shot apart around a walking captain (engine-world-wasm
// `broadside`), on the range page (fracture/range-page.mjs).
import {rangePage} from './fracture/range-page.mjs';

await rangePage({
 scene:0,
 asset:'galleon-stern.glb',
 loading:'Loading the galleon…',
 lighter:'the lighter ship',
 prefs:'broadside-hud',
 director:()=>true,
 view:[[-7.5,10.5,-15],[9.5,3,0]],
});
