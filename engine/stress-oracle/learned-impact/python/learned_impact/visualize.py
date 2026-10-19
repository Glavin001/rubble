"""Recorded oracle vs exact PyTorch checkpoint comparison, served on localhost."""
from __future__ import annotations
import argparse
import hashlib
import json
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
import torch
from .dataset import event_batches
from .model import LearnedImpactNet
from .physics import PhysicsImpactNet, PreviewImpactNet, StatePreviewImpactNet, WorldStateImpactNet, WorldInputControlNet, WorldMotionImpactNet, ForcingPreviewImpactNet, ForcingNativeImpactNet
from .constitutive_model import ConstitutiveNativeImpactNet
from .history_model import HistoryCarryImpactNet, ResidualHistoryImpactNet
from .projectile_model import ProjectileHistoryImpactNet, ProjectileResponseImpactNet
from .physical_response_model import PhysicalResponseImpactNet
from .consistent_model import ConsistentHistoryImpactNet
from .response_model import ResponseHistoryImpactNet
from .normalize import normalize_batch
from .schema import load_event
from .world_targets import reconstruct, rotation_log


def world_comparison(start, end, residual, valid, dt):
    """One-step endpoint inspection, with the exact recorded start pose."""
    predicted = reconstruct(start, residual, dt)
    baseline = reconstruct(start, np.zeros_like(residual), dt)
    valid = np.asarray(valid, dtype=bool)
    error = predicted['centers'] - np.asarray(end['centers'])
    baseline_error = baseline['centers'] - np.asarray(end['centers'])
    angle = rotation_log(predicted['rotations'] @ np.swapaxes(np.asarray(end['rotations']), -1, -2))
    def values(a):
        return [float(x) if ok else None for x, ok in zip(np.linalg.norm(a, axis=1), valid)]
    return {'centers': predicted['centers'].tolist(), 'rotations': predicted['rotations'].tolist(),
            'valid': valid.tolist(), 'position_error': values(error),
            'baseline_position_error': values(baseline_error),
            'truth_travel': values(np.asarray(end['centers']) - np.asarray(start['centers'])),
            'predicted_travel': values(predicted['centers'] - np.asarray(start['centers'])),
            'position_rmse_m': float(np.sqrt(np.mean(error[valid] ** 2))) if valid.any() else None,
            'baseline_position_rmse_m': float(np.sqrt(np.mean(baseline_error[valid] ** 2))) if valid.any() else None,
            'orientation_rmse_rad': float(np.sqrt(np.mean(np.sum(angle[valid] ** 2, axis=1)))) if valid.any() else None,
            'scored_pieces': int(valid.sum())}


def scores(truth, predicted):
    tp = int(np.count_nonzero(truth & predicted))
    fp = int(np.count_nonzero(~truth & predicted))
    fn = int(np.count_nonzero(truth & ~predicted))
    return counts_metrics(tp, fp, fn)


def counts_metrics(tp, fp, fn):
    return {'tp': tp, 'fp': fp, 'fn': fn, 'precision': tp / max(1, tp + fp),
            'recall': tp / max(1, tp + fn),
            'f1': 2 * tp / (2 * tp + fp + fn) if (tp + fp + fn) else None}


def load_checkpoint(path):
    sd_path = path / 'state_dict.pt'
    sd = torch.load(sd_path, map_location='cpu', weights_only=True)
    h, p = sd['attn.assign.weight'].shape[1], sd['attn.assign.weight'].shape[0]
    run_path = path / 'run.json'
    run = json.loads(run_path.read_text()) if run_path.exists() else {}
    constructors = {'physics-v2':PhysicsImpactNet, 'physics-preview-v3':PreviewImpactNet,
                    'physics-preview-state-v4':StatePreviewImpactNet,
                    'physics-preview-world-v5':WorldStateImpactNet,'physics-preview-world-control-v5':WorldInputControlNet,'physics-preview-world-motion-v6':WorldMotionImpactNet,'physics-forcing-preview-v7':ForcingPreviewImpactNet,'physics-forcing-native-v7':ForcingNativeImpactNet,'physics-constitutive-native-v10':ConstitutiveNativeImpactNet,'physics-history-native-v11':HistoryCarryImpactNet,'physics-history-residual-native-v12':ResidualHistoryImpactNet,'physics-projectile-history-native-v13':ProjectileHistoryImpactNet,'physics-consistent-history-native-v15':ConsistentHistoryImpactNet,'physics-response-history-native-v18':ResponseHistoryImpactNet,'physics-projectile-response-native-v22':ProjectileResponseImpactNet,'physics-projectile-response-si-v23':PhysicalResponseImpactNet}
    arch = run.get('architecture')
    if arch and arch not in (*constructors, 'legacy'):
        raise ValueError(f'unsupported checkpoint architecture {arch}')
    model = constructors[arch](h=h, p=p, threshold=run['threshold']) if arch in constructors else LearnedImpactNet(h=h, p=p)
    model.load_state_dict(sd)
    model.eval()
    metrics_path = path/'metrics.json'
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    brief = {part:{k:v for k,v in metrics[part].items() if k in ('frames','precision','recall','f1','util_log_mae','state_rmse','zero_increment_rmse')}
             for part in ('train','validation','test','all_recorded_frames') if isinstance(metrics.get(part),dict)}
    return model, {'name': run.get('display_name', f'{path.parent.name}/{path.name}'), 'path': str(path.resolve()), 'h': h, 'p': p,
                   'sha256': hashlib.sha256(sd_path.read_bytes()).hexdigest(), 'run': run, 'metrics':brief}


@torch.no_grad()
def compare_event(path, model, geometry):
    ev = load_event(path,world_inputs=getattr(model,'requires_world_inputs',False),world_targets=getattr(model,'requires_world_targets',False),forcing_inputs=getattr(model,'requires_forcing_inputs',False),exact_joint_history=getattr(model,'requires_exact_joint_history',False),material_inputs=getattr(model,'requires_material_inputs',False),projectile_inputs=getattr(model,'requires_projectile_inputs',False),bond_response_history=getattr(model,'requires_bond_response_history',False))
    world_replay = json.loads((path/'replay.json').read_text()) if getattr(model,'requires_world_targets',False) else None
    input_contract=getattr(model,'input_contract','causal_preview' if isinstance(model,PreviewImpactNet) else 'oracle_interval')
    frames = []
    tp = fp = fn = 0
    for i, raw in enumerate(event_batches(ev,bond_policy=getattr(model,"bond_policy","first_failure"))):
        batch = model.prepare_batch(raw) if isinstance(model, PhysicsImpactNet) else normalize_batch(raw)
        pred = model(batch)
        if any(not torch.isfinite(v).all() for v in pred.values()):
            raise ValueError(f'nonfinite prediction {path.name} frame {i}')
        ids = raw['bond_ids'].numpy()
        truth = raw['bond_fail_class'].numpy()
        predicted = (model.predict_classes(pred) if isinstance(model, PhysicsImpactNet) else pred['bond_fail_logits'].argmax(-1)).numpy()
        sm = scores(truth > 0, predicted > 0)
        tp += sm['tp']; fp += sm['fp']; fn += sm['fn']
        delta = model.decode_delta(pred).numpy() if isinstance(model, PhysicsImpactNet) else pred['chunk_state_delta'].numpy().copy()
        if not isinstance(model, PhysicsImpactNet):
            delta[:, :3] *= batch['_scales']['he']
            delta[:, 6:9] *= batch['_scales']['he']
        truth_delta = ev.frames[i]['chunk_state_delta']
        motion_valid = raw.get('delta_valid',torch.ones(len(truth_delta),dtype=torch.bool)).numpy()
        def motion_values(values):
            return [float(v) if valid else None for v,valid in zip(np.linalg.norm(values[:,:3],axis=1),motion_valid)]
        # The head learns log1p utilization. Metrics in log space remain finite
        # even for an extreme model prediction; display exponent is capped only
        # at float32's range, never applied to the solver or training labels.
        log_util = pred['bond_peak_util'].numpy()
        util = np.expm1(np.clip(log_util, 0, 80))
        truth_load = raw['chunk_load_bins'].numpy()
        model_load = np.zeros_like(truth_load) if input_contract=='frame_start_forcing' else raw['chunk_preview_load_bins'].numpy() if isinstance(model, PreviewImpactNet) else truth_load
        frames.append({'truth_load': np.linalg.norm(truth_load[:,:,:3],axis=2).max(axis=1).tolist(),
                       'model_load': np.linalg.norm(model_load[:,:,:3],axis=2).max(axis=1).tolist(),
                       'load_error': np.linalg.norm(model_load[:,:,:3]-truth_load[:,:,:3],axis=2).max(axis=1).tolist(),
                       'truth_torque': np.linalg.norm(truth_load[:,:,3:],axis=2).max(axis=1).tolist(),
                       'model_torque': np.linalg.norm(model_load[:,:,3:],axis=2).max(axis=1).tolist(),
                       'index': i, 'time': (i + 1) * ev.manifest['dt'],
                       'active': ids.tolist(), 'truth': truth.tolist(), 'predicted': predicted.tolist(),
                       'truth_util': raw['bond_peak_util'].tolist(), 'predicted_util': util.tolist(),
                       'motion_valid':motion_valid.tolist(),
                       'truth_delta': motion_values(truth_delta),
                       'predicted_delta': motion_values(delta),
                       'delta_error': motion_values(delta-truth_delta),
                       'metrics': {**sm, 'util_log_mae': float(np.abs(log_util - np.log1p(raw['bond_peak_util'].numpy().clip(0))).mean()) if len(ids) else 0,
                                   'motion_scored_chunk_intervals':int(motion_valid.sum()),
                                   'motion_fragment_reset_excluded_chunk_intervals':int((~motion_valid).sum()),
                                   'displacement_rmse_m': float(np.sqrt(np.mean((delta[motion_valid, :3] - truth_delta[motion_valid, :3]) ** 2))) if motion_valid.any() else None}})
        if world_replay is not None:
            frames[-1]['world'] = world_comparison(world_replay['frames'][i], world_replay['frames'][i+1],
                model.decode_world_motion(pred).numpy(), raw['world_motion_valid'].numpy(), ev.manifest['dt'])
        if getattr(model,'requires_bond_response_history',False):
            from .response_view import comparison as response_comparison
            history=ev.bond_response_history
            frames[-1]['response']=response_comparison(history.values[i+1,ids],model.decode_response(pred),
                history.values[i,ids],history.flags[i+1,ids,0].astype(bool),raw['bond_endpoints'].numpy(),ev.manifest['n_chunks'])
    return {'input_contract': input_contract,
            'response_available':getattr(model,'requires_bond_response_history',False),
            'manifest': ev.manifest, 'geometry': geometry, 'context': ev.context,
            'half_extents': ev.static['chunk_half_extents'].tolist(),
            'endpoints': ev.static['bond_endpoints'].tolist(),
            'frames': frames, 'metrics': counts_metrics(tp, fp, fn),
            'world_motion_available': world_replay is not None,
            'evaluation': ('No interval contact forces are supplied. Prescribed force/pressure/blast descriptors come from the known scene schedule. Zero input-load panels mean unused input slots, not predicted zero force. ' if input_contract=='frame_start_forcing' else 'Contact loads come from a frame-start rigid preview. ' if isinstance(model, PreviewImpactNet) else 'Contact loads come from the oracle response over the interval. ') + 'Every prediction starts from the recorded oracle frame. ' + ('Cracked interfaces remain in the graph while mechanically active; only still-connected bonds can fracture again. ' if getattr(model,'bond_policy',None)=='runtime_alive' else 'Previously failed bonds are excluded using oracle history. ') + ('World endpoint inspection interpolates from the oracle start pose to the predicted endpoint, then resets at the next interval. Projectiles and active-piece masks remain recorded. This is not an autonomous trajectory.' if world_replay is not None else 'Both panels share oracle motion; the model predicts peak bond utilization and new breaks, not the displayed trajectories.')}


def load_replay(path, entry, manifest):
    """Require exact provenance and stable chunk indexing before animating labels."""
    replay = json.loads(path.read_text())
    if replay.get('format') != 'learned-impact-replay/1':
        raise ValueError('unknown replay format')
    if replay.get('source_sha256') != entry['sha256']:
        raise ValueError('replay does not match the recorded event fingerprint')
    frames = replay['frames']
    if len(frames) != manifest['n_frames'] + 1:
        raise ValueError('replay must contain initial pose plus every recorded frame')
    n = manifest['n_chunks']
    for i, frame in enumerate(frames):
        if not np.isclose(frame['time'] - frames[0]['time'], i * manifest['dt'], atol=1e-8, rtol=1e-6):
            raise ValueError('replay time does not match oracle frame')
        for key, shape in [('centers', (n, 3)), ('rotations', (n, 3, 3)), ('active', (n,)), ('clusters', (n,))]:
            a = np.asarray(frame[key])
            if a.shape != shape or not np.isfinite(a).all():
                raise ValueError(f'invalid replay {key}')
    return replay


class Comparison:
    def __init__(self, dataset, weights, geometry):
        self.dataset = dataset
        self.manifest = json.loads((dataset / 'manifest.json').read_text())
        self.events = {e['path']: e for e in self.manifest['events']}
        self.geometry = json.loads(geometry.read_text())['events']
        self.replays = geometry.parent / 'replays'
        for name in self.events:
            context = dataset / name / 'context.json'
            if name not in self.geometry and context.exists():
                c = json.loads(context.read_text())
                self.geometry[name] = {k: c[k] for k in ('centers', 'rotations', 'source_scene')}
        missing = self.events.keys() - self.geometry.keys()
        if missing:
            raise ValueError(f'missing verified geometry for {sorted(missing)}')
        self.models = [load_checkpoint(w) for w in weights]
        self.partitions = [json.loads((w/'split.json').read_text()).get('events',{}) if (w/'split.json').exists() else {} for w in weights]
        self.dataset_sha = hashlib.sha256((dataset / 'manifest.json').read_bytes()).hexdigest()

    def index(self):
        coverage_path = self.dataset.parent/'coverage.json'
        coverage = json.loads(coverage_path.read_text()) if coverage_path.exists() else None
        if coverage:
            coverage = {k:v for k,v in coverage.items() if k not in ('rows','parameter_values')}
        return {'events': [{**e, 'motion': self.replay_path(e['path']).exists()} for e in self.events.values()], 'models': [meta for _, meta in self.models],
                'dataset_sha256': self.dataset_sha, 'frames': sum(e['n_frames'] for e in self.events.values()), 'coverage':coverage}

    def replay_path(self, name):
        native = self.dataset / name / 'replay.json'
        return native if native.exists() else self.replays / (name + '.json')

    @lru_cache(maxsize=24)
    def event(self, name, model):
        if name not in self.events or model < 0 or model >= len(self.models):
            raise ValueError('unknown event or checkpoint')
        result = compare_event(self.dataset / name, self.models[model][0], self.geometry[name])
        run = self.models[model][1]['run']
        result['partition'] = self.partitions[model].get(name, 'unseen' if self.partitions[model] else
            'train' if run.get('manifest_sha256') == self.dataset_sha else 'unknown')
        path = self.replay_path(name)
        result['replay'] = load_replay(path, self.events[name], result['manifest']) if path.exists() else None
        return result

    def report(self):
        reports = []
        for model, (_, meta) in enumerate(self.models):
            rows = []
            for name, entry in self.events.items():
                data = self.event(name, model)
                rows.append({'event': name, 'structure_type': entry['structure_type'], **data['metrics']})
            tp, fp, fn = (sum(r[k] for r in rows) for k in ('tp', 'fp', 'fn'))
            families = {}
            for family in sorted({r['structure_type'] for r in rows}):
                members = [r for r in rows if r['structure_type'] == family]
                families[family] = counts_metrics(*(sum(r[k] for r in members) for k in ('tp', 'fp', 'fn')))
            reports.append({'model': meta, 'metrics': counts_metrics(tp, fp, fn), 'families': families, 'events': rows})
        return {'scope': 'Recorded-frame comparison. For the all-data checkpoint these are in-sample metrics, not generalization.',
                'dataset_sha256': self.dataset_sha, 'models': reports}


def serve(comparison, port, experiments=None, rollout_catalog=None):
    comparisons = {'main': comparison, **(experiments or {})}
    asset_dir = Path(__file__).parent / 'viewer'
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            parsed = urlparse(self.path)
            try:
                q = parse_qs(parsed.query)
                selected = comparisons[q.get('dataset', ['main'])[0]]
                if parsed.path in ('/api/rollouts', '/api/rollout'):
                    catalog = json.loads(rollout_catalog.read_text()) if rollout_catalog else []
                    if parsed.path == '/api/rollouts':
                        payload = [{k: item[k] for k in ('id', 'label')} for item in catalog]
                    else:
                        item = next((item for item in catalog if item['id'] == q['event'][0]), None)
                        if item is None: raise ValueError('Unknown independent rollout')
                        payload = json.loads(Path(item['comparison']).read_text())
                        # Augment older immutable recordings without rewriting their provenance.
                        from .interaction_metrics import compare_interactions
                        payload['metrics']['interactions'] = compare_interactions(payload['oracle'], payload['model'])
                    data = json.dumps(payload, allow_nan=False).encode(); content_type = 'application/json'
                elif parsed.path == '/api/index':
                    data = json.dumps({**selected.index(), 'datasets': list(comparisons), 'dataset_labels': {'main':'All recorded scenarios','preview':'Causal input experiment','tails':'Longer impact recordings · 120 ms','scale':'Physical scale test · 10,400 chunks','layouts':'New wall layouts · 48 impacts','responses':'Corrected oracle · experimental response models','impact-responses':'Corrected oracle · 128 scenarios · full response model'}}, allow_nan=False).encode(); content_type = 'application/json'
                elif parsed.path == '/api/audit':
                    path = selected.dataset.parent / 'audit.json'
                    if not path.exists():
                        self.send_error(404, 'No audit has been generated for this dataset'); return
                    data = path.read_bytes(); content_type = 'application/json'
                elif parsed.path == '/api/event':
                    q = parse_qs(parsed.query)
                    data = json.dumps(selected.event(q['event'][0], int(q.get('model', ['0'])[0])), allow_nan=False).encode(); content_type = 'application/json'
                elif parsed.path in ('/', '/rollout', '/rollout.js', '/viewer.js', '/viewer.css', '/scene.js'):
                    name = 'index.html' if parsed.path == '/' else 'rollout.html' if parsed.path == '/rollout' else parsed.path[1:]
                    data = (asset_dir / name).read_bytes()
                    content_type = 'text/javascript' if name.endswith('.js') else 'text/html' if name.endswith('.html') else 'text/css'
                else:
                    self.send_error(404); return
                self.send_response(200)
                self.send_header('Content-Type', content_type + '; charset=utf-8')
                self.send_header('Content-Length', str(len(data)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers(); self.wfile.write(data)
            except (KeyError, ValueError, OSError) as e:
                self.send_error(400, str(e))
        def log_message(self, fmt, *args):
            pass
    server = HTTPServer(('127.0.0.1', port), Handler)
    print(f'Stress learned comparison: http://127.0.0.1:{server.server_port}', flush=True)
    server.serve_forever()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--weights', type=Path, action='append', required=True, help='repeat to enable checkpoint switching')
    p.add_argument('--geometry', type=Path, required=True)
    p.add_argument('--preview-dataset', type=Path)
    p.add_argument('--preview-weights', type=Path, action='append')
    p.add_argument('--tail-dataset', type=Path)
    p.add_argument('--tail-weights', type=Path, action='append')
    p.add_argument('--scale-dataset', type=Path)
    p.add_argument('--scale-weights', type=Path, action='append')
    p.add_argument('--layout-dataset', type=Path)
    p.add_argument('--layout-weights', type=Path, action='append')
    p.add_argument('--response-dataset', type=Path)
    p.add_argument('--response-weights', type=Path, action='append')
    p.add_argument('--impact-response-dataset', type=Path)
    p.add_argument('--impact-response-weights', type=Path, action='append')
    p.add_argument('--port', type=int, default=8791)
    p.add_argument('--rollout-catalog', type=Path, help='Explicit catalog of audited independent trajectory comparisons')
    p.add_argument('--report', type=Path, help='write all-event metrics and exit')
    args = p.parse_args()
    torch.set_num_threads(4)
    app = Comparison(args.dataset, args.weights, args.geometry)
    if args.report:
        args.report.write_text(json.dumps(app.report(), indent=2, allow_nan=False) + '\n')
        print(f'wrote {args.report}')
    else:
        experiments = {'preview': Comparison(args.preview_dataset, args.preview_weights, args.geometry)} if args.preview_dataset and args.preview_weights else {}
        if args.tail_dataset and args.tail_weights:
            experiments['tails'] = Comparison(args.tail_dataset, args.tail_weights, args.geometry)
        if args.scale_dataset and args.scale_weights:
            experiments['scale'] = Comparison(args.scale_dataset, args.scale_weights, args.geometry)
        if args.layout_dataset and args.layout_weights:
            experiments['layouts'] = Comparison(args.layout_dataset, args.layout_weights, args.geometry)
        if args.response_dataset and args.response_weights:
            experiments['responses'] = Comparison(args.response_dataset, args.response_weights, args.geometry)
        if args.impact_response_dataset and args.impact_response_weights:
            experiments['impact-responses'] = Comparison(args.impact_response_dataset, args.impact_response_weights, args.geometry)
        serve(app, args.port, experiments, args.rollout_catalog)


if __name__ == '__main__':
    main()
