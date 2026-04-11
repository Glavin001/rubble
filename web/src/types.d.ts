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
  /** 4 floats: [step+transform_copy, matrix_update, render, total] */
  lastFrameTimingsMs?: Float32Array;
  error: string | null;
  /** Benchmark-only: stop the render loop and wait for any in-flight frame. */
  stopLoop?: () => Promise<void>;
  /** Benchmark-only: run one physics step without rendering. Returns 7-float timings. */
  benchStep?: () => Promise<number[]>;
  /** Run one full animation loop iteration (step + sync + render). */
  loopStep?: () => Promise<void>;
}

interface Window {
  __rubble_test?: RubbleTestHooks;
}
