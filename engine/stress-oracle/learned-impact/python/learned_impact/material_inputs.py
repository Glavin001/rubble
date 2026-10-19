"""Versioned constitutive descriptors recoverable from the saved source scene.

Legacy models omit friction, shear caps, material kind and rate/fatigue laws.
These descriptors preserve those distinctions without re-running trajectories.
They are source-law inputs, not a reconstructed World checkpoint. Existing
architectures do not consume them until explicitly wired into a new version.
"""
import numpy as np

CONTRACT='constitutive-material-descriptors/1'
FEATURE_FLAGS=('rate_effects','static_fatigue','weibull','softening','crack_contact','crushing',
               'buckling','rebar','rigid_motion_loads','blast_shadowing','blast_clearing','damping')
METHOD_FLAGS=('corotational_bonds','exact_frame_transfer','exact_joint_patch','exact_rate_filter',
              'patch_after_damage','rayleigh_damping')
MATERIAL_FIELDS=('density','youngs_modulus','poisson_ratio','damping_ratio','tensile_strength',
                 'compressive_strength','cohesion','friction','fracture_tension','fracture_shear',
                 'fracture_compression','shear_cap','has_shear_cap','weibull_modulus','has_weibull',
                 'dif_reference_rate','dif_exponent','dif_transition_rate','dif_exponent_high','dif_max',
                 'has_dif','fatigue_exponent','fatigue_test_time','has_fatigue','brittle','ductile')


def material_features(material):
    """26 finite fields; optional laws have explicit masks, never implicit zeros."""
    keys=('density','youngs_modulus','poisson_ratio','damping_ratio','tensile_strength','compressive_strength','cohesion','friction')
    values=[material[k] for k in keys]
    values += [material['fracture_energy'][k] for k in ('tension','shear','compression')]
    scales=[1000.,1e9,1.,.01,1e6,1e6,1e6,1.,100.,100.,100.]
    for key,scale in (('shear_cap',1e6),('weibull_modulus',1.)):
        value=material[key];values.extend([0. if value is None else value,float(value is not None)]);scales.extend([scale,1.])
    for key,fields in (('dif',('reference_rate','exponent','transition_rate','exponent_high','max')),
                       ('static_fatigue',('exponent','test_time'))):
        law=material[key]
        values.extend([0. if law is None else law[f] for f in fields]);values.append(float(law is not None))
        scales.extend([1.]*(len(fields)+1))
    kind=material['kind']
    if kind not in ('brittle','ductile'):raise ValueError('unsupported material kind')
    values.extend([float(kind=='brittle'),float(kind=='ductile')]);scales.extend([1.,1.])
    raw=np.asarray(values,dtype=np.float64)
    if raw.shape!=(26,) or not np.isfinite(raw).all():raise ValueError('invalid constitutive material values')
    encoded=np.sign(raw)*np.log1p(np.abs(raw)/np.asarray(scales))
    # Masks/indicators remain exact 0/1; numeric fields retain their magnitude.
    for i in (12,14,20,23,24,25):encoded[i]=raw[i]
    return encoded.astype(np.float32)


def observations(scene,context,manifest,static):
    if scene['name']!=context['source_scene']:raise ValueError('material source scene identity mismatch')
    n=int(manifest['n_chunks']);e=int(manifest['n_bonds'])
    chunk_keys=context.get('chunk_keys');bond_keys=context.get('bond_keys')
    if chunk_keys is None or bond_keys is None or len(chunk_keys)!=n or len(bond_keys)!=e:
        raise ValueError('material descriptors require aligned chunk/bond keys')
    if len({tuple(k) for k in chunk_keys})!=n or len({tuple(k) for k in bond_keys})!=e:raise ValueError('duplicate material mapping keys')
    def checked_key(key):
        if len(key)!=2 or any(type(v) is not int or v<0 for v in key):raise ValueError('invalid source geometry key')
        return key
    materials={name:material_features(value) for name,value in scene['materials'].items()}
    nodes=[];edges=[];rebar=[];slots={tuple(k):i for i,k in enumerate(chunk_keys)}
    try:
        for key in chunk_keys:
            s,c=checked_key(key);nodes.append(materials[scene['bodies'][s]['chunks'][c]['material']])
        for i,key in enumerate(bond_keys):
            s,b=checked_key(key);bond=scene['bodies'][s]['bonds'][b]
            expected=[slots[(s,bond['a'])],slots[(s,bond['b'])]]
            if expected!=static['bond_endpoints'][i].tolist():raise ValueError('bond material/endpoints misaligned')
            edges.append(materials[bond['material']])
            # Scene BondDesc::rebar defaults to None. Structure construction
            # retains a tie exactly when the source has one and features.rebar.
            rebar.append(bond.get('rebar') is not None and scene['sim']['features']['rebar'])
    except (IndexError,KeyError) as error:raise ValueError('source material mapping missing or stale') from error
    sim=scene['sim'];flags=[]
    for mapping,names in ((sim['features'],FEATURE_FLAGS),(sim['methods'],METHOD_FLAGS)):
        for name in names:
            value=mapping[name]
            if type(value) is not bool:raise ValueError('constitutive flags must be explicit booleans')
            flags.append(float(value))
    if context.get('solver_methods')!=sim['methods']:raise ValueError('resolved solver methods mismatch')
    if type(sim['fracture']) is not bool:raise ValueError('fracture enablement must be explicit')
    flags.append(float(sim['fracture']))
    scale=float(sim['stiffness_scale'])
    if not np.isfinite(scale) or scale<=0:raise ValueError('invalid stiffness scale')
    flags.append(np.log1p(scale))
    return {'chunk_material_features':np.asarray(nodes,dtype=np.float32).reshape(n,26),
            'bond_material_features':np.asarray(edges,dtype=np.float32).reshape(e,26),
            'bond_has_rebar':np.asarray(rebar,dtype=bool),
            'constitutive_flags':np.asarray(flags,dtype=np.float32)}
