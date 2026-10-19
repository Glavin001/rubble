// Learned-impact bond mode: MGN-T forward (H=48, P=16) on the page's WebGPU device.
// Loads trained weights (learned-impact-slang/1), runs Slang→WGSL kernels, readbacks
// bond fail-bin / peak-util for the HUD. Does not drive bond breaks (follow-up).
import {loadShaderModules,pipelineBuilder} from '../shader-loader.mjs';

export const LEAN_H=48,LEAN_P=16,NODE_IN=46,EDGE_IN=28,FFN_MULT=4,ATTN_BLOCKS=1,MP_PRE=2,MP_POST=2,N_HEADS=4;
export const FAIL_BINS=5,OUT_PER_EDGE=FAIL_BINS+1;
export const WEIGHTS_FORMAT='learned-impact-slang/1';

const ENTRIES=[
  ['encode_nodes_l1',256],['encode_nodes_l2',256],['encode_edges_l1',256],['encode_edges_l2',256],
  ['mp_edges',256],['mp_nodes',256],
  ['attn_slice',256],['attn_tokens',16],['attn_qkv',256],['attn_mha',256],
  ['attn_deslice_mix',256],['attn_ffn_up',256],['attn_ffn_down',256],
  ['decode_bonds',256],
];

export function weightCount(h=LEAN_H,p=LEAN_P,attnBlocks=ATTN_BLOCKS,mpLayers=MP_PRE+MP_POST){
  const h2=FFN_MULT*h;
  const encN=h*NODE_IN+h+h*h+h,encE=h*EDGE_IN+h+h*h+h;
  const mpE=h*3*h+h+h*h+h,mpN=h*2*h+h+h*h+h;
  const attn=p*h+p+h+1+3*h*h+3*h+h*h+h+h2*h+h2+h*h2+h;
  const dec=FAIL_BINS*h+FAIL_BINS+h+1;
  return encN+encE+mpLayers*(mpE+mpN)+attnBlocks*attn+dec;
}

export function randomWeights(seed=1,h=LEAN_H){
  const n=weightCount();const w=new Float32Array(n);let s=seed|1;
  for(let i=0;i<n;i++){s=Math.imul(s,0x5851f42d)+1|0;const u=((s>>>0)/0xffffffff);w[i]=(u*2-1)/Math.sqrt(h)}
  return w;
}

/**
 * Load weights.bin + weights.json; verify format and byte size vs tensor shapes / n_floats.
 * @returns {Promise<{weights: Float32Array, meta: object}>}
 */
export async function loadSlangWeights(baseUrl,{status=()=>{}}={}){
  const jsonUrl=new URL('weights.json',baseUrl);
  const binUrl=new URL('weights.bin',baseUrl);
  status('Loading learned-impact weights…');
  const [jr,br]=await Promise.all([fetch(jsonUrl),fetch(binUrl)]);
  if(!jr.ok)throw new Error(`learned-impact weights.json: HTTP ${jr.status} (${jsonUrl})`);
  if(!br.ok)throw new Error(`learned-impact weights.bin: HTTP ${br.status} (${binUrl})`);
  const meta=await jr.json();
  const buf=await br.arrayBuffer();
  if(meta.format!==WEIGHTS_FORMAT){
    throw new Error(`learned-impact weights format ${meta.format||'?'} (want ${WEIGHTS_FORMAT}); re-export with pack_slang`);
  }
  const fromShapes=(meta.tensors||[]).reduce((n,t)=>{
    const shape=t.shape||[];return n+shape.reduce((a,b)=>a*b,1);
  },0);
  const expectFloats=meta.n_floats??fromShapes;
  const expectBytes=meta.n_bytes??expectFloats*4;
  if(buf.byteLength!==expectBytes){
    throw new Error(`learned-impact weights.bin size ${buf.byteLength} ≠ weights.json n_bytes ${expectBytes}`);
  }
  if(fromShapes&&fromShapes!==expectFloats){
    throw new Error(`learned-impact weights.json tensor shapes sum ${fromShapes} ≠ n_floats ${expectFloats}`);
  }
  const expectLayout=weightCount(meta.h??LEAN_H,meta.p??LEAN_P,meta.attn_blocks??ATTN_BLOCKS,meta.mp_layers??(MP_PRE+MP_POST));
  if(expectFloats!==expectLayout){
    throw new Error(`learned-impact weight count ${expectFloats} ≠ Slang layout ${expectLayout} (H=${meta.h} P=${meta.p})`);
  }
  if((meta.h??LEAN_H)!==LEAN_H||(meta.p??LEAN_P)!==LEAN_P){
    throw new Error(`learned-impact dims H=${meta.h} P=${meta.p}; kernels are H=${LEAN_H} P=${LEAN_P}`);
  }
  return {weights:new Float32Array(buf),meta};
}

/** Build undirected CSR + edgeIndex from `[E*2]` i32 endpoints. */
export function csrFromEndpoints(endpoints,n){
  const e=endpoints.length>>1;
  const deg=new Uint32Array(n);
  for(let i=0;i<e;i++){deg[endpoints[i*2]|0]++;deg[endpoints[i*2+1]|0]++}
  const csr=new Uint32Array(n+1+e*2);
  let at=n+1;
  const start=new Uint32Array(n);
  for(let i=0;i<n;i++){csr[i]=at;start[i]=at;at+=deg[i]}
  csr[n]=at;
  const edgeIndex=new Uint32Array(e*2);
  for(let ei=0;ei<e;ei++){
    const a=endpoints[ei*2]|0,b=endpoints[ei*2+1]|0;
    edgeIndex[ei*2]=a;edgeIndex[ei*2+1]=b;
    csr[start[a]++]=ei<<1;csr[start[b]++]=ei<<1;
  }
  return {csr,edgeIndex};
}

/** Build CSR + features from a stress-oracle pack JSON (`scenario.nodes` / `bonds`). */
export function graphFromPack(pack){
  const nodes=pack.scenario.nodes,bonds=pack.scenario.bonds;
  const n=nodes.length,e=bonds.length;
  const endpoints=new Int32Array(e*2);
  for(let i=0;i<e;i++){endpoints[i*2]=bonds[i].node0;endpoints[i*2+1]=bonds[i].node1}
  const {csr,edgeIndex}=csrFromEndpoints(endpoints,n);
  // Schema-sized features; pack JSON only fills a subset (rest zero). Full oracle
  // frames would populate joint state / load bins for parity with training.
  const nodeIn=new Float32Array(n*NODE_IN),edgeIn=new Float32Array(e*EDGE_IN);
  for(let i=0;i<n;i++){
    const c=nodes[i].centroid||{},base=i*NODE_IN;
    nodeIn[base]=nodes[i].mass||0;
    nodeIn[base+1]=nodes[i].inertia?.[0]||0;
    nodeIn[base+2]=nodes[i].inertia?.[1]||0;
    nodeIn[base+3]=nodes[i].inertia?.[2]||0;
    nodeIn[base+4]=nodes[i].half_extents?.[0]||c.x||0;
    nodeIn[base+5]=nodes[i].half_extents?.[1]||c.y||0;
    nodeIn[base+6]=nodes[i].half_extents?.[2]||c.z||0;
  }
  for(let i=0;i<e;i++){
    const b=bonds[i],nr=b.normal||{},base=i*EDGE_IN;
    edgeIn[base]=b.area||1;edgeIn[base+1]=nr.x||0;edgeIn[base+2]=nr.y||0;edgeIn[base+3]=nr.z||1;
  }
  const islands=new Uint32Array([0,n,0,e]);
  const nodeIsland=new Uint32Array(n);
  return {nNodes:n,nEdges:e,nIslands:1,islands,csr,edgeIndex,nodeIn,edgeIn,nodeIsland};
}

/**
 * Graph from stress-web live capture (`stress_learned_*`).
 * @param {{nNodes:number,nEdges:number,nodeIn:Float32Array,edgeIn:Float32Array,endpoints:Int32Array|number[]}} snap
 */
export function graphFromLive(snap){
  const n=snap.nNodes|0,e=snap.nEdges|0;
  const endpoints=snap.endpoints instanceof Int32Array?snap.endpoints:new Int32Array(snap.endpoints);
  const {csr,edgeIndex}=csrFromEndpoints(endpoints,n);
  const nodeIn=snap.nodeIn instanceof Float32Array?snap.nodeIn:new Float32Array(snap.nodeIn);
  const edgeIn=snap.edgeIn instanceof Float32Array?snap.edgeIn:new Float32Array(snap.edgeIn);
  if(nodeIn.length!==n*NODE_IN)throw new Error(`live node_in ${nodeIn.length} ≠ ${n}*${NODE_IN}`);
  if(edgeIn.length!==e*EDGE_IN)throw new Error(`live edge_in ${edgeIn.length} ≠ ${e}*${EDGE_IN}`);
  return {nNodes:n,nEdges:e,nIslands:1,islands:new Uint32Array([0,n,0,e]),csr,edgeIndex,nodeIn,edgeIn,nodeIsland:new Uint32Array(n)};
}

/** Train-time RMS scales (mirrors `learned_impact.normalize.normalize_batch`) on flat features. */
export function normalizeLiveFeatures(nodeIn,edgeIn,n,e){
  const he=rmsSlice(nodeIn,n,NODE_IN,4,3,1e-3);
  const mass=rmsSlice(nodeIn,n,NODE_IN,0,1,1e-3);
  const load=rmsSlice(nodeIn,n,NODE_IN,19,24,1.0);
  const gravFloor=9.81;
  const joint=rmsSlice(edgeIn,e,EDGE_IN,0,22,1.0);
  const qs=rmsSlice(edgeIn,e,EDGE_IN,22,6,1.0);
  const nOut=new Float32Array(nodeIn),eOut=new Float32Array(edgeIn);
  for(let i=0;i<n;i++){
    const b=i*NODE_IN;
    nOut[b]/=mass;
    for(let k=0;k<3;k++){nOut[b+1+k]/=(mass*he*he);nOut[b+4+k]/=he}
    for(let k=0;k<3;k++){nOut[b+7+k]/=he;nOut[b+13+k]/=he}
    for(let k=0;k<24;k++)nOut[b+19+k]/=load;
    for(let k=0;k<3;k++)nOut[b+43+k]/=gravFloor;
  }
  for(let i=0;i<e;i++){
    const b=i*EDGE_IN;
    for(let k=0;k<22;k++)eOut[b+k]/=joint;
    for(let k=0;k<6;k++)eOut[b+22+k]/=qs;
  }
  return {nodeIn:nOut,edgeIn:eOut,scales:{he,mass,load,joint,qs}};
}

function rmsSlice(arr,count,stride,off,width,floor){
  let s=0,n=0;
  for(let i=0;i<count;i++){
    const b=i*stride+off;
    for(let k=0;k<width;k++){const v=arr[b+k];s+=v*v;n++}
  }
  return Math.max(Math.sqrt(s/Math.max(n,1)),floor);
}

/** Dense edge indices whose argmax fail-bin is ≠ 0. */
export function failEdgeIndices(pred,nEdges,{maxBreaks=64}={}){
  const out=[];
  for(let e=0;e<nEdges&&out.length<maxBreaks;e++){
    const base=e*OUT_PER_EDGE;
    let best=0,bestV=pred[base];
    for(let c=1;c<FAIL_BINS;c++){const v=pred[base+c];if(v>bestV){bestV=v;best=c}}
    if(best>0)out.push(e);
  }
  return out;
}

function packGraphU(g){
  const offIslands=0,offCsr=4,offEdges=offCsr+g.csr.length,offNodeIsl=offEdges+g.edgeIndex.length;
  const u=new Uint32Array(offNodeIsl+g.nNodes);
  u.set(g.islands,offIslands);u.set(g.csr,offCsr);u.set(g.edgeIndex,offEdges);u.set(g.nodeIsland,offNodeIsl);
  return {u,offIslands,offCsr,offEdges,offNodeIsl};
}

function workLayout(g){
  const H=LEAN_H,P=LEAN_P,h2=FFN_MULT*H;
  let o=0;
  const offTokens=o;o+=g.nIslands*P*H;
  const offSlice=o;o+=g.nNodes*P;
  const offQkv=o;o+=g.nIslands*P*3*H;
  const offOutput=o;o+=Math.max(g.nNodes*H,g.nEdges*OUT_PER_EDGE,g.nIslands*P*H);
  const offFfn=o;o+=Math.max(g.nNodes*h2,g.nEdges*H);
  return {size:o,offTokens,offSlice,offQkv,offOutput,offFfn};
}

/** std140 Params matching Slang's Params_std140_0 (80 bytes). */
function writeParams(block,g,gu,wl){
  const b=new ArrayBuffer(80),v=new DataView(b);
  const u32=(o,x)=>v.setUint32(o,x,true);
  u32(0,g.nNodes);u32(4,g.nEdges);u32(8,g.nIslands);u32(12,block);
  u32(16,MP_PRE);u32(20,MP_POST);u32(24,ATTN_BLOCKS);u32(28,N_HEADS);
  u32(32,gu.offIslands);u32(36,gu.offCsr);u32(40,gu.offEdges);u32(44,gu.offNodeIsl);
  u32(48,wl.offTokens);u32(52,wl.offSlice);u32(56,wl.offQkv);u32(60,wl.offOutput);
  u32(64,wl.offFfn);u32(68,0);u32(72,0);u32(76,0);
  return b;
}

function groups(n,wg){return Math.max(1,Math.ceil(n/wg))}

function summarizePreds(pred,nEdges){
  const failHist=new Array(FAIL_BINS).fill(0);
  let utilSum=0,utilMax=-Infinity,utilMin=Infinity,failish=0;
  for(let e=0;e<nEdges;e++){
    const base=e*OUT_PER_EDGE;
    let best=0,bestV=pred[base];
    for(let c=1;c<FAIL_BINS;c++){const v=pred[base+c];if(v>bestV){bestV=v;best=c}}
    failHist[best]++;
    if(best>0)failish++;
    const u=pred[base+FAIL_BINS];
    utilSum+=u;utilMax=Math.max(utilMax,u);utilMin=Math.min(utilMin,u);
  }
  return {
    failHist,
    failish,
    failishFrac:nEdges?failish/nEdges:0,
    utilMean:nEdges?utilSum/nEdges:0,
    utilMax:Number.isFinite(utilMax)?utilMax:0,
    utilMin:Number.isFinite(utilMin)?utilMin:0,
  };
}

/**
 * @param {GPUDevice} device
 * @param {{packUrl?: string|URL, pack?: object, graph?: object, weightsUrl?: string|URL, status?: (s:string)=>void, normalize?: boolean}} opts
 */
export async function createLearnedImpact(device,opts={}){
  const status=opts.status||(()=>{});
  let g=opts.graph;
  if(!g){
    let pack=opts.pack;
    if(!pack){
      // Served under fracture/assets (HTTP root is engine-webgpu; cannot escape to stress-oracle/).
      const url=opts.packUrl||new URL('./assets/house-1story.json',import.meta.url);
      status('Loading learned-impact pack…');
      const r=await fetch(url);if(!r.ok)throw new Error(`learned-impact pack: HTTP ${r.status}`);
      pack=await r.json();
    }
    g=graphFromPack(pack);
  }
  if(opts.normalize){
    const n=normalizeLiveFeatures(g.nodeIn,g.edgeIn,g.nNodes,g.nEdges);
    g={...g,nodeIn:n.nodeIn,edgeIn:n.edgeIn};
  }
  let gu=packGraphU(g),wl=workLayout(g);

  const weightsUrl=opts.weightsUrl||new URL('./assets/learned-impact/',import.meta.url);
  let weights,weightMeta;
  try{
    ({weights,meta:weightMeta}=await loadSlangWeights(weightsUrl,{status}));
  }catch(e){
    status(`Learned-impact weights missing (${e.message}); using random`);
    weights=randomWeights(1);
    weightMeta={format:'random',h:LEAN_H,p:LEAN_P,n_floats:weights.length};
  }

  const [{module}]=await loadShaderModules(device,[{name:'learned_impact',file:'learned_impact'}],{
    urlOf:f=>`../generated/wgsl/${f}.wgsl`,status,missing:f=>`Missing ${f}.wgsl; slangc engine/slang/kernels/learned_impact.slang -target wgsl -o engine/crates/engine-webgpu/generated/wgsl/learned_impact.wgsl`,
  });

  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  const mk=(label,bytes,extra=0)=>device.createBuffer({label,size:Math.max(4,bytes),usage:usage|extra});
  const bufWeights=mk('li_weights',weights.byteLength);device.queue.writeBuffer(bufWeights,0,weights);
  let bufGraph=mk('li_graph_u',gu.u.byteLength);device.queue.writeBuffer(bufGraph,0,gu.u);
  let bufNodeIn=mk('li_node_in',g.nodeIn.byteLength);device.queue.writeBuffer(bufNodeIn,0,g.nodeIn);
  let bufEdgeIn=mk('li_edge_in',g.edgeIn.byteLength);device.queue.writeBuffer(bufEdgeIn,0,g.edgeIn);
  let bufNodeX=mk('li_node_x',g.nNodes*LEAN_H*4);
  let bufEdgeX=mk('li_edge_x',g.nEdges*LEAN_H*4);
  let bufWork=mk('li_work',wl.size*4,GPUBufferUsage.COPY_SRC);
  let predBytes=g.nEdges*OUT_PER_EDGE*4;
  let bufPred=device.createBuffer({label:'li_pred_read',size:Math.max(4,predBytes),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});

  const layout=device.createBindGroupLayout({label:'li',entries:[
    {binding:0,visibility:GPUShaderStage.COMPUTE,buffer:{type:'uniform'}},
    {binding:1,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
    {binding:2,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
    {binding:3,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
    {binding:4,visibility:GPUShaderStage.COMPUTE,buffer:{type:'read-only-storage'}},
    {binding:5,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
    {binding:6,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
    {binding:7,visibility:GPUShaderStage.COMPUTE,buffer:{type:'storage'}},
  ]});
  const pipeLayout=device.createPipelineLayout({bindGroupLayouts:[layout]});
  const builder=pipelineBuilder(device,ENTRIES.length,status);
  const pipes={};
  for(const [name,wg] of ENTRIES){
    pipes[name]={wg,pipeline:await builder.create(name,{layout:pipeLayout,compute:{module,entryPoint:name}})};
  }
  await builder.finish();

  let paramsBufs=[];
  let binds=[];
  let bind0,bindMp,bindAttn;
  function rebuildBinds(){
    for(const b of paramsBufs)b.destroy();
    paramsBufs=[];binds=[];
    const mkBind=block=>{
      const pb=device.createBuffer({label:`li_params_${block}`,size:80,usage:GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST});
      device.queue.writeBuffer(pb,0,writeParams(block,g,gu,wl));
      paramsBufs.push(pb);
      const bg=device.createBindGroup({layout,entries:[
        {binding:0,resource:{buffer:pb}},
        {binding:1,resource:{buffer:bufWeights}},
        {binding:2,resource:{buffer:bufGraph}},
        {binding:3,resource:{buffer:bufNodeIn}},
        {binding:4,resource:{buffer:bufEdgeIn}},
        {binding:5,resource:{buffer:bufNodeX}},
        {binding:6,resource:{buffer:bufEdgeX}},
        {binding:7,resource:{buffer:bufWork}},
      ]});
      binds.push(bg);return bg;
    };
    bind0=mkBind(0);
    bindMp=[0,1,2,3].map(mkBind);
    bindAttn=[0].map(mkBind);
  }
  rebuildBinds();

  function encodeForward(encoder){
    const nh=g.nNodes*LEAN_H,eh=g.nEdges*LEAN_H,nh2=g.nNodes*FFN_MULT*LEAN_H,tok=g.nIslands*LEAN_P;
    const pass=encoder.beginComputePass({label:'learned_impact'});
    const go=(name,bind,n)=>{const p=pipes[name];pass.setPipeline(p.pipeline);pass.setBindGroup(0,bind);pass.dispatchWorkgroups(name==='attn_tokens'?Math.max(1,g.nIslands):groups(n,p.wg))};
    go('encode_nodes_l1',bind0,g.nNodes);go('encode_nodes_l2',bind0,nh);
    go('encode_edges_l1',bind0,g.nEdges);go('encode_edges_l2',bind0,eh);
    for(let i=0;i<MP_PRE;i++){go('mp_edges',bindMp[i],g.nEdges);go('mp_nodes',bindMp[i],g.nNodes)}
    for(let a=0;a<ATTN_BLOCKS;a++){
      const b=bindAttn[a];
      go('attn_slice',b,g.nNodes);go('attn_tokens',b,g.nIslands);
      go('attn_qkv',b,tok*3*LEAN_H);go('attn_mha',b,tok);
      go('attn_deslice_mix',b,g.nNodes);go('attn_ffn_up',b,nh2);go('attn_ffn_down',b,nh);
    }
    for(let i=0;i<MP_POST;i++){go('mp_edges',bindMp[MP_PRE+i],g.nEdges);go('mp_nodes',bindMp[MP_PRE+i],g.nNodes)}
    go('decode_bonds',bind0,g.nEdges);
    pass.end();
  }

  let lastMs=0,runs=0,lastSummary=null,lastPred=null;
  return {
    get graph(){return g},
    params:weights.length,
    weightSource:weightMeta.format||WEIGHTS_FORMAT,
    get lastSummary(){return lastSummary},
    get lastPred(){return lastPred},
    /**
     * Replace graph/features (e.g. stress-web live capture). Rebuilds buffers when N/E change.
     * @param {object} next graphFromLive / graphFromPack shape
     * @param {{normalize?: boolean}} o
     */
    resync(next,o={}){
      let ng=next;
      if(o.normalize??opts.normalize){
        const n=normalizeLiveFeatures(ng.nodeIn,ng.edgeIn,ng.nNodes,ng.nEdges);
        ng={...ng,nodeIn:n.nodeIn,edgeIn:n.edgeIn};
      }
      const same=ng.nNodes===g.nNodes&&ng.nEdges===g.nEdges;
      g=ng;gu=packGraphU(g);wl=workLayout(g);predBytes=g.nEdges*OUT_PER_EDGE*4;
      if(same){
        device.queue.writeBuffer(bufGraph,0,gu.u);
        device.queue.writeBuffer(bufNodeIn,0,g.nodeIn);
        device.queue.writeBuffer(bufEdgeIn,0,g.edgeIn);
        rebuildBinds();
      }else{
        for(const b of [bufGraph,bufNodeIn,bufEdgeIn,bufNodeX,bufEdgeX,bufWork,bufPred])b.destroy();
        bufGraph=mk('li_graph_u',gu.u.byteLength);device.queue.writeBuffer(bufGraph,0,gu.u);
        bufNodeIn=mk('li_node_in',g.nodeIn.byteLength);device.queue.writeBuffer(bufNodeIn,0,g.nodeIn);
        bufEdgeIn=mk('li_edge_in',g.edgeIn.byteLength);device.queue.writeBuffer(bufEdgeIn,0,g.edgeIn);
        bufNodeX=mk('li_node_x',g.nNodes*LEAN_H*4);
        bufEdgeX=mk('li_edge_x',g.nEdges*LEAN_H*4);
        bufWork=mk('li_work',wl.size*4,GPUBufferUsage.COPY_SRC);
        bufPred=device.createBuffer({label:'li_pred_read',size:Math.max(4,predBytes),usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});
        rebuildBinds();
      }
    },
    /** Run one forward; await GPU; optionally read back bond preds. */
    async run({readback=true}={}){
      const t0=performance.now();
      const enc=device.createCommandEncoder();
      encodeForward(enc);
      if(readback)enc.copyBufferToBuffer(bufWork,wl.offOutput*4,bufPred,0,predBytes);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      lastMs=performance.now()-t0;runs++;
      if(readback){
        await bufPred.mapAsync(GPUMapMode.READ);
        lastPred=new Float32Array(bufPred.getMappedRange().slice(0));
        bufPred.unmap();
        lastSummary=summarizePreds(lastPred,g.nEdges);
      }
      return lastMs;
    },
    get lastMs(){return lastMs},
    get runs(){return runs},
    dispose(){
      for(const b of [bufWeights,bufGraph,bufNodeIn,bufEdgeIn,bufNodeX,bufEdgeX,bufWork,bufPred,...paramsBufs])b.destroy();
    },
  };
}
