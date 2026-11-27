// npm run check-data [DIR]: a workspace's committable data (default: the repo's
// apps/reference-studio/workspace/) holds points, text and numbers only (no image or pixel
// data) and agrees with the structure schemas. Exits 1 with one line per problem.

import { checkData } from '../shared/check-data.ts';
import { DiskWorkspace, WORKSPACE } from './node-workspace.ts';

const dir = process.argv[2] ?? WORKSPACE;
const problems = await checkData(new DiskWorkspace(dir));
for (const p of problems) console.log(p);
console.log(problems.length ? `${problems.length} problem(s) in ${dir}` : `ok: ${dir} holds points and text only (library/ and recordings/ are not committed)`);
process.exitCode = problems.length ? 1 : 0;
