"""Physical actor motion in two independent worlds at shared physical times.

This does not infer penetration from displacement, or contact impulse from total
momentum change. No event with missing actors or a short observation horizon is
silently counted as a successful interaction. Tolerances are reporting settings,
never changes to the physics engine.
"""
import numpy as np


def _actors(frame):
    result = {}
    for actor in frame['impactors']:
        name = actor['name']
        if name in result:
            raise ValueError('duplicate actor identity')
        result[name] = actor
    return result


def compare_interactions(oracle, model, velocity_tolerance_mps=0.01):
    if not np.isfinite(velocity_tolerance_mps) or velocity_tolerance_mps < 0:
        raise ValueError('invalid direction reporting tolerance')
    if len(oracle) < 2 or len(oracle) != len(model):
        raise ValueError('need matching independent trajectories')
    times = np.array([f['time'] for f in oracle], dtype=float)
    other_times = np.array([f['time'] for f in model], dtype=float)
    if (not np.isfinite(times).all() or not np.isfinite(other_times).all()
            or (np.diff(times) <= 0).any()
            or not np.allclose(times, other_times, atol=1e-9, rtol=0)):
        raise ValueError('interaction clocks must match and increase')
    worlds = {side: [_actors(f) for f in frames]
              for side, frames in [('oracle', oracle), ('model', model)]}
    names = set(worlds['oracle'][0])
    if any(set(frame) != names for frames in worlds.values() for frame in frames):
        raise ValueError('actor identities changed or are missing')
    duration = float(times[-1] - times[0])

    def mean_square(values):
        return float(np.trapezoid(np.square(values).sum(axis=1), times) / duration)

    def direction(value):
        return ('forward' if value > velocity_tolerance_mps else
                'backward' if value < -velocity_tolerance_mps else 'near stationary')

    actors = []
    for name in sorted(names):
        initial = worlds['oracle'][0][name]
        if initial != worlds['model'][0][name]:
            raise ValueError('actor initial states differ')
        mass = float(initial['mass'])
        if not np.isfinite(mass) or mass <= 0:
            raise ValueError('invalid actor mass')
        series = {}
        for side, frames in worlds.items():
            entries = [f[name] for f in frames]
            if any(a['mass'] != mass or a['driven'] != initial['driven'] for a in entries):
                raise ValueError('actor mass or driven status changed')
            x = np.array([a['center'] for a in entries], dtype=float)
            v = np.array([a['velocity'] for a in entries], dtype=float)
            if x.shape != (len(times), 3) or v.shape != x.shape or not np.isfinite([x, v]).all():
                raise ValueError('invalid actor motion')
            series[side] = (x, v)
        x0, v0 = series['oracle'][0][0], series['oracle'][1][0]
        speed = float(np.linalg.norm(v0))
        axis = v0 / speed if speed > velocity_tolerance_mps else None
        ox, ov = series['oracle']; mx, mv = series['model']
        report = {
            'name': name, 'mass_kg': mass, 'driven': initial['driven'],
            'axis': None if axis is None else axis.tolist(),
            'axis_definition': 'initial oracle velocity; unavailable for initially stationary actors',
            'position_vector_rmse_m': mean_square(mx - ox) ** .5,
            'velocity_vector_rmse_mps': mean_square(mv - ov) ** .5,
            'final_position_error_m': float(np.linalg.norm(mx[-1] - ox[-1])),
            'final_velocity_error_mps': float(np.linalg.norm(mv[-1] - ov[-1])),
        }
        for side, (x, v) in series.items():
            item = {'translational_ke_j': (.5 * mass * np.square(v).sum(axis=1)).tolist()}
            if axis is not None:
                travel = (x - x0) @ axis; forward = v @ axis
                reverse = np.flatnonzero(forward < -velocity_tolerance_mps)
                item.update(
                    forward_displacement_m=travel.tolist(), forward_velocity_mps=forward.tolist(),
                    final_direction=direction(float(forward[-1])),
                    maximum_sampled_forward_displacement_m=float(travel.max()),
                    first_saved_reverse_time_s=float(times[reverse[0]]) if len(reverse) else None,
                    net_forward_momentum_change_kg_mps=float(mass * (forward[-1] - forward[0])),
                )
            report[side] = item
        report['final_direction_disagreement'] = (None if axis is None else
            report['oracle']['final_direction'] != report['model']['final_direction'])
        actors.append(report)
    return {
        'format': 'independent-actor-motion/1', 'times_s': times.tolist(),
        'duration_s': duration, 'velocity_tolerance_mps': velocity_tolerance_mps,
        'actors': actors, 'status': 'measured' if actors else 'no_external_actors',
        'scope': ('Saved-frame motion, trapezoidal time-weighted squared vector errors. '
                  'Negative forward velocity means motion opposite the initial velocity, '
                  'not proof of a particular contact event. No geometry-based penetration '
                  'classification. Total momentum change includes all external forces; '
                  'kinetic energy excludes rotation. Driven actors are prescribed inputs '
                  'and cannot validate structural resistance through motion alone.'),
    }
