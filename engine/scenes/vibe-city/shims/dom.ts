// Browser globals three's GLTFLoader expects, for running the GLB preset under bun.
// The export uses only geometry, so textures decode to a 1x1 placeholder.

const g = globalThis as Record<string, unknown>;

if (typeof g.ProgressEvent === "undefined") {
  g.ProgressEvent = class ProgressEvent extends Event {
    lengthComputable: boolean;
    loaded: number;
    total: number;
    constructor(type: string, init: { lengthComputable?: boolean; loaded?: number; total?: number } = {}) {
      super(type);
      this.lengthComputable = init.lengthComputable ?? false;
      this.loaded = init.loaded ?? 0;
      this.total = init.total ?? 0;
    }
  };
}

if (typeof g.createImageBitmap === "undefined") {
  g.createImageBitmap = async () => ({ width: 1, height: 1, close() {} });
}
