// npm run check-data: the committed studio data holds points and text only (no image or
// pixel data) and agrees with the structure schemas. Exits 1 with one line per problem.

import { checkData } from '../server/check-data.ts';
import { defaultPaths } from '../server/store.ts';

const dir = process.argv[2] ?? defaultPaths().data;
const problems = checkData(dir);
for (const p of problems) console.log(p);
console.log(problems.length ? `${problems.length} problem(s) in ${dir}` : `ok: ${dir} holds points and text only`);
process.exitCode = problems.length ? 1 : 0;
