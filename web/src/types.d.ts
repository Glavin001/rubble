/** Test hooks exposed on window for Playwright E2E tests */
interface RubbleTestHooks {
  ready: boolean;
  stepCount: number;
  bodyCount: number;
  getPositions?: () => Float32Array;
  getAngles?: () => Float32Array;
  getTransforms?: () => Float32Array;
  /** 7 floats: [upload, predict+aabb, broadphase, narrowphase, contact_fetch, solve, extract] */
  lastStepTimingsMs?: Float32Array;
  error: string | null;
}

interface Window {
  __rubble_test?: RubbleTestHooks;
}
