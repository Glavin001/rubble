// Khronos glTF Validator on a .glb:  node validate.mjs <file.glb>
// (npm install in this folder first; gltf-validator is a dev dependency.)
import fs from 'node:fs';
import validator from 'gltf-validator';

const path = process.argv[2];
const data = new Uint8Array(fs.readFileSync(path));
const report = await validator.validateBytes(data, { maxIssues: 50 });
const { numErrors, numWarnings, numInfos, messages } = report.issues;
console.log(`${path}: ${numErrors} errors, ${numWarnings} warnings, ${numInfos} infos`);
for (const m of messages.filter((m) => m.severity <= 1).slice(0, 20)) {
  console.log(`  [${m.severity === 0 ? 'error' : 'warning'}] ${m.code} ${m.pointer ?? ''} ${m.message}`);
}
process.exit(numErrors ? 1 : 0);
