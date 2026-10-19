// Learned-impact bond mode (compute only): lean MGN-T forward (H=48, P=64) on the
// page's WebGPU device. Slang → WGSL (`generated/wgsl/learned_impact.wgsl`). Outputs
// are discarded; the HUD shows ms/forward so we can verify town-scale latency in-browser.
import {loadShaderModules,pipelineBuilder} from '../shader-loader.mjs';

export const LEAN_H=48,LEAN_P=64,FEAT_IN=16,FFN_MULT=4,ATTN_BLOCKS=2,MP_PRE=2,MP_POST=2,N_HEADS=4;

const ENTRIES=[
  ['encode_nodes_l1',256],['encode_nodes_l2',256],['encode_edges_l1',256],['encode_edges_l2',256],
  ['mp_edges',256],['mp_nodes',256],
  ['attn_slice',256],['attn_tokens',64],['attn_qkv',256],['attn_mha',256],
  ['attn_deslice_mix',256],['attn_ffn_up',256],['attn_ffn_down',256],
  ['decode',256],
];

export function weightCount(h=LEAN_H,p=LEAN_P,attnBlocks=ATTN_BLOCKS,mpLayers=MP_PRE+MP_POST){
  const enc=h*FEAT_IN+h+h*h+h,mpE=h*3*h+h+h*h+h,mpN=h*2*h+h+h*h+h,h2=FFN_MULT*h;
  const attn=p*h+h+1+3*h*h+3*h+h*h+h+h2*h+h2+h*h2+h,dec=h*h+h+h*h+h;
  return 2*enc+mpLayers*(mpE+mpN)+attnBlocks*attn+dec;
}

export function randomWeights(seed=1,h=LEAN_H){
  const n=weightCount();const w=new Float32Array(n);let s=seed|1;
  for(let i=0;i<n;i++){s=Math.imul(s,0x5851f42d)+1|0;const u=((s>>>0)/0xffffffff);w[i]=(u*2-1)/Math.sqrt(h)}
  return w;
}

/** Build CSR + features from a stress-oracle pack JSON (`scenario.nodes` / `bonds`). */
export function graphFromPack(pack){
  const nodes=pack.scenario.nodes,bonds=pack.scenario.bonds;
  const n=nodes.length,e=bonds.length;
  const deg=new Uint32Array(n);
  for(const b of bonds){deg[b.node0]++;deg[b.node1]++}
  const csr=new Uint32Array(n+1+e*2); // row ptrs then adjacency edge ids (<<1 | side unused)
  let at=n+1;
  const start=new Uint32Array(n);
  for(let i=0;i<n;i++){csr[i]=at;start[i]=at;at+=deg[i]}
  csr[n]=at;
  const edgeIndex=new Uint32Array(e*2);
  for(let ei=0;ei<e;ei++){
    const a=bonds[ei].node0,b=bonds[ei].node1;
    edgeIndex[ei*2]=a;edgeIndex[ei*2+1]=b;
    csr[start[a]++]=ei<<1;csr[start[b]++]=ei<<1;
  }
  const nodeIn=new Float32Array(n*FEAT_IN),edgeIn=new Float32Array(e*FEAT_IN);
  for(let i=0;i<n;i++){
    const c=nodes[i].centroid||{};const base=i*FEAT_IN;
    nodeIn[base]=c.x||0;nodeIn[base+1]=c.y||0;nodeIn[base+2]=c.z||0;
    nodeIn[base+3]=nodes[i].mass||0;nodeIn[base+4]=nodes[i].volume||1;
  }
  for(let i=0;i<e;i++){
    const b=bonds[i],nr=b.normal||{},base=i*FEAT_IN;
    edgeIn[base]=b.area||1;edgeIn[base+1]=nr.x||0;edgeIn[base+2]=nr.y||0;edgeIn[base+3]=nr.z||1;
  }
  // One island covering all nodes/edges (pack_town disjoint copies would be multiple).
  const islands=new Uint32Array([0,n,0,e]);
  const nodeIsland=new Uint32Array(n); // all 0
  return {nNodes:n,nEdges:e,nIslands:1,islands,csr,edgeIndex,nodeIn,edgeIn,nodeIsland};
}

function packGraphU(g){
  // islands (4) + csr + edge pairs + node_island
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
  const offOutput=o;o+=Math.max(g.nNodes*H,g.nIslands*P*H);
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

/**
 * @param {GPUDevice} device
 * @param {{packUrl?: string, pack?: object, status?: (s:string)=>void}} opts
 */
export async function createLearnedImpact(device,opts={}){
  const status=opts.status||(()=>{});
  let pack=opts.pack;
  if(!pack){
    const url=opts.packUrl||new URL('../../../stress-oracle/packs/house-1story.json',import.meta.url);
    status('Loading learned-impact pack…');
    const r=await fetch(url);if(!r.ok)throw new Error(`learned-impact pack: HTTP ${r.status}`);
    pack=await r.json();
  }
  const g=graphFromPack(pack);
  const gu=packGraphU(g),wl=workLayout(g);
  const weights=randomWeights(1);

  const [{module}]=await loadShaderModules(device,[{name:'learned_impact',file:'learned_impact'}],{
    urlOf:f=>`../generated/wgsl/${f}.wgsl`,status,missing:f=>`Missing ${f}.wgsl; slangc engine/slang/kernels/learned_impact.slang -target wgsl -o engine/crates/engine-webgpu/generated/wgsl/learned_impact.wgsl`,
  });

  const usage=GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST;
  const mk=(label,bytes,extra=0)=>device.createBuffer({label,size:Math.max(4,bytes),usage:usage|extra});
  const bufWeights=mk('li_weights',weights.byteLength);device.queue.writeBuffer(bufWeights,0,weights);
  const bufGraph=mk('li_graph_u',gu.u.byteLength);device.queue.writeBuffer(bufGraph,0,gu.u);
  const bufNodeIn=mk('li_node_in',g.nodeIn.byteLength);device.queue.writeBuffer(bufNodeIn,0,g.nodeIn);
  const bufEdgeIn=mk('li_edge_in',g.edgeIn.byteLength);device.queue.writeBuffer(bufEdgeIn,0,g.edgeIn);
  const bufNodeX=mk('li_node_x',g.nNodes*LEAN_H*4);
  const bufEdgeX=mk('li_edge_x',g.nEdges*LEAN_H*4);
  const bufWork=mk('li_work',wl.size*4);

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

  const paramsBufs=[];
  const binds=[];
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
  const bind0=mkBind(0);
  const bindMp=[0,1,2,3].map(mkBind);
  const bindAttn=[0,1].map(mkBind);

  const nh=g.nNodes*LEAN_H,eh=g.nEdges*LEAN_H,nh2=g.nNodes*FFN_MULT*LEAN_H,tok=g.nIslands*LEAN_P;

  function encodeForward(encoder){
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
    go('decode',bind0,g.nNodes);
    pass.end();
  }

  let lastMs=0,runs=0;
  return {
    graph:g,
    params:weightCount(),
    /** Run one forward; await GPU; return wall ms. Output not read. */
    async run(){
      const t0=performance.now();
      const enc=device.createCommandEncoder();
      encodeForward(enc);
      device.queue.submit([enc.finish()]);
      await device.queue.onSubmittedWorkDone();
      lastMs=performance.now()-t0;runs++;
      return lastMs;
    },
    get lastMs(){return lastMs},
    get runs(){return runs},
    dispose(){
      for(const b of [bufWeights,bufGraph,bufNodeIn,bufEdgeIn,bufNodeX,bufEdgeX,bufWork,...paramsBufs])b.destroy();
    },
  };
}
