"""Lossless native stress trajectories with an explicit frame-input boundary."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .joint_history import FIELDS as JOINT_FIELDS,validate_values
from .response_history import FIELDS as RESPONSE_FIELDS
from .runtime_snapshot import NativeGraph

FORMAT='learned-stress-phase/1'

def _equal(a,b,label):
    a=np.asarray(a,dtype='<f8');b=np.asarray(b,dtype='<f8')
    if not np.array_equal(a.view('u8'),b.view('u8')):raise ValueError(label)

def _history(snapshot):
    return np.array([[np.inf if x is None else x for x in row] for row in snapshot['joint_values']],dtype='<f8')

class PhaseHistory:
    def __init__(self,root):
        self.root=Path(root);self.directory=self.root/'stress_phases'
        self.meta=json.loads((self.directory/'manifest.json').read_text())
        self.report=json.loads((self.root/'report.json').read_text())
        self.graph=json.loads((self.root/'graph.json').read_text())
        m=self.meta
        if m.get('format')!=FORMAT or m.get('status')!='completed' or self.report.get('status')!='completed':
            raise ValueError('incomplete or unsupported stress phase recording')
        if (m['input_width'],m['target_width'],m['chunk_width'])!=(51,42,18) or m['joint_fields']!=JOINT_FIELDS or m['response_fields']!=RESPONSE_FIELDS:
            raise ValueError('stress phase field contract mismatch')
        if m['bond_keys']!=self.graph['bond_keys'] or m['chunk_keys']!=self.graph['chunk_keys']:
            raise ValueError('phase graph keys mismatch')
        self.phases=m['phases'];self.frames=int(self.report['frames']);self.rows=int(m['rows'])
        n=len(m['chunk_keys']);e=len(m['bond_keys']);p=len(self.phases)
        if self.rows!=self.report['oracle_response_rows'] or p!=self.report['phases'] or self.report['engine_fallback_evaluations']!=0:
            raise ValueError('phase report count/fallback mismatch')
        self.inputs=self._map('inputs.f64','<f8',(self.rows,51))
        self.targets=self._map('targets.f64','<f8',(self.rows,42))
        self.slots=self._map('bond_slots.u32','<u4',(self.rows,))
        self.input_flags=self._map('input_flags.i32','<i4',(self.rows,4))
        self.target_flags=self._map('target_flags.i32','<i4',(self.rows,4))
        self.chunks=self._map('chunks.f64','<f8',(p,n,18))
        self.chunk_flags=self._map('chunk_flags.u64','<u8',(p,n,2))
        self.frame_phases=[[] for _ in range(self.frames)];offset=0;previous=None
        for i,phase in enumerate(self.phases):
            frame=phase['frame'];count=phase['rows']
            if type(frame)!=int or not 0<=frame<self.frames or type(count)!=int or count<0 or phase['row_start']!=offset:
                raise ValueError('invalid phase index/offset')
            if type(phase['fracture'])!=bool or not np.isfinite([phase['time'],phase['dt']]).all() or phase['dt']<=0:
                raise ValueError('invalid phase clock/flags')
            if previous is not None and (phase['substep']!=previous['substep']+1 or phase['time']!=previous['time']+previous['dt']
                or frame not in (previous['frame'],previous['frame']+1)):
                raise ValueError('native phases are not contiguous')
            self.frame_phases[frame].append(i);offset+=count;previous=phase
        if offset!=self.rows or not p or any(not indices for indices in self.frame_phases):raise ValueError('missing native phases')
        scene=self.graph['scene'];enabled=scene['sim']['features']['rebar']
        self.has_rebar=np.array([
            enabled and scene['bodies'][s]['bonds'][b].get('rebar') is not None for s,b in m['bond_keys']],dtype=bool)

    def _map(self,name,dtype,shape):
        path=self.directory/name;size=int(np.prod(shape))*np.dtype(dtype).itemsize
        if path.stat().st_size!=size:raise ValueError(f'phase byte count mismatch: {name}')
        return np.memmap(path,dtype=dtype,mode='r',shape=shape) if size else np.empty(shape,dtype=dtype)

    def snapshot(self,index):
        if not 0<=index<=self.frames:raise IndexError('snapshot outside recorded frames')
        return json.loads((self.root/'snapshots'/f'{index:06}.json').read_text())

    def frame_inputs(self,index,*,include_projectiles=True):
        """Only the selected frame start; never reads targets or phase observations."""
        if not 0<=index<self.frames:raise IndexError('frame outside recorded trajectory')
        return NativeGraph(self.graph).batch(self.snapshot(index),include_projectiles=include_projectiles)

    def phase_rows(self,index):
        p=self.phases[index];return slice(p['row_start'],p['row_start']+p['rows'])

    def audit(self,source=None):
        initial=self.snapshot(0);current=_history(initial);response=np.array(initial['response_values'],dtype='<f8')
        breaks=0;negative=0;minimum=0.;by_frame=[];source_replay=None
        if source is not None:source_replay=json.loads((Path(source)/'replay.json').read_text())['frames']
        for frame,indices in enumerate(self.frame_phases):
            start=self.snapshot(frame);end=self.snapshot(frame+1);first=self.phases[indices[0]];last=self.phases[indices[-1]]
            if start['world']['time']!=first['time'] or end['world']['time']!=last['time']+last['dt']:
                raise ValueError('phase/frame clock mismatch')
            if source_replay is not None and (start['world']!=source_replay[frame] or end['world']!=source_replay[frame+1]):
                raise ValueError('phase recording changed original world trajectory')
            _equal(current,_history(start),'frame-start joint history mismatch');_equal(response,start['response_values'],'frame-start response mismatch')
            frame_breaks=0
            for i in indices:
                p=self.phases[i];sl=self.phase_rows(i);ids=np.asarray(self.slots[sl]);x=self.inputs[sl];y=self.targets[sl]
                a=self.input_flags[sl];b=self.target_flags[sl]
                if len(np.unique(ids))!=len(ids) or (ids>=len(current)).any():raise ValueError('invalid phase bond slots')
                validate_values(x[:,12:33],a)
                validate_values(y[:,:21],np.column_stack([b[:,:2],np.ones(len(b),dtype='i4'),b[:,2]]))
                if not (a[:,2]==1).all() or not np.isin(b[:,3],[0,1]).all():raise ValueError('invalid phase participation flags')
                if not np.isfinite(x[:,:12]).all() or not np.isfinite(x[:,33:]).all() or not np.isfinite(y[:,21:]).all():raise ValueError('nonfinite phase values')
                _equal(x[:,12:33],current[ids],'phase history chain mismatch');_equal(x[:,33:],response[ids],'phase response chain mismatch')
                old_connected=(x[:,12]<1)|(self.has_rebar[ids]&(a[:,0]==0))
                new_connected=(y[:,0]<1)|(self.has_rebar[ids]&(b[:,0]==0))
                if not np.array_equal(a[:,3],old_connected) or not np.array_equal(b[:,2],new_connected) or not np.array_equal(b[:,3],old_connected&~new_connected):
                    raise ValueError('phase connectivity contradiction')
                irreversible=[0,1,2,3,6,16]
                if (y[:,irreversible]<x[:,np.array(irreversible)+12]).any() or ((a[:,0]==1)&(b[:,0]==0)).any():raise ValueError('phase history healing')
                if (y[:,[33,39,41]]<0).any():raise ValueError('negative stored/dissipated/damping work')
                _equal(y[:,19],x[:,31]+y[:,39],'phase dissipation accumulation mismatch')
                current[ids]=y[:,:21];response[ids]=y[:,21:39]
                frame_breaks+=int(b[:,3].sum());negative+=int((y[:,40]<0).sum())
                if len(y):minimum=min(minimum,float(y[:,40].min()))
                if not np.isfinite(self.chunks[i]).all() or not np.isin(self.chunk_flags[i,:,0],[0,1]).all():raise ValueError('invalid phase chunk observations')
            _equal(current,_history(end),'frame-end joint history mismatch');_equal(response,end['response_values'],'frame-end response mismatch')
            breaks+=frame_breaks;by_frame.append({'frame':frame,'phases':len(indices),'breaks':frame_breaks})
        files=sorted(p for p in self.root.rglob('*') if p.is_file() and p.name not in ['audit.json'])
        return {'status':'completed_verified','frames':self.frames,'phases':len(self.phases),'rows':self.rows,
            'breaks':breaks,'negative_signed_residuals':negative,'minimum_signed_residual_j':minimum,'by_frame':by_frame,
            'original_world_replay_exact':source_replay is not None,
            'files_sha256':{str(p.relative_to(self.root)):self._hash(p) for p in files},
            'scope':'Lossless native-phase training labels, exact history/response chain and optional original world replay parity. No learned inference or runtime speed claim.'}

    @staticmethod
    def _hash(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1<<20),b''):h.update(block)
        return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);p.add_argument('--source',type=Path)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    if a.out.exists():raise FileExistsError(a.out)
    report=PhaseHistory(a.root).audit(a.source);a.out.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files_sha256'}))

if __name__=='__main__':main()
