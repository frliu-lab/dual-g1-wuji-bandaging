"""Sensitivity sweep for the proxy-guided, prewrapped flexible strip."""
import json
import argparse
from itertools import product
import xml.etree.ElementTree as ET
import numpy as np
import mujoco

from guided_flex_pilot import make_guided_xml, OUT


def one(mass_kg, friction, solref):
    root = ET.fromstring(make_guided_xml())
    root.find("worldbody/flexcomp").set('mass', str(mass_kg))
    root.find("worldbody/geom[@name='forearm_proxy']").set('friction', f'{friction} .005 .0001')
    root.find('worldbody/flexcomp/contact').set('friction', f'{friction} .005 .0001')
    root.find('worldbody/flexcomp/contact').set('solref', f'{solref} 1')
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding='unicode'))
    data = mujoco.MjData(model)
    forearm = model.geom('forearm_proxy').id
    flex_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_FLEX, 'bandage_strip')
    guide_id = model.body('guide_proxy').mocapid[0]
    penetration = 0.
    finite = True
    for step in range(round(.5/model.opt.timestep)):
        u = min(1., max(0., (step*model.opt.timestep-.05)/.40))
        theta = .5*np.pi*(3*u*u-2*u*u*u)
        data.mocap_pos[guide_id] = [.055, .049*np.cos(theta), .12+.049*np.sin(theta)]
        mujoco.mj_step(model, data)
        finite &= bool(np.all(np.isfinite(data.qpos)))
        for c in data.contact:
            if forearm in c.geom and flex_id in c.flex:
                penetration = max(penetration, max(0., -c.dist))
    centers = np.asarray(data.flexvert_xpos)[:66].reshape(33, 2, 3).mean(axis=1)
    radial = np.hypot(centers[:, 1], centers[:, 2]-.12)
    near_arm = (.047 <= radial) & (radial <= .065)
    angle = np.mod(np.arctan2(centers[:, 2]-.12, centers[:, 1]), 2*np.pi)
    bins = np.floor(angle/(2*np.pi)*12).astype(int).clip(0, 11)
    coverage = [len(set(bins[lo:hi][near_arm[lo:hi]]))/12 for lo, hi in ((0, 17), (16, 33))]
    error = float(np.linalg.norm(centers[-1]-data.mocap_pos[guide_id]))
    return {'mass_kg': mass_kg, 'sliding_friction': friction,
            'max_forearm_penetration_m': penetration,
            'endpoint_error_m': error, 'coverage_per_turn_12_bins': coverage,
            'finite': finite,
            'passes_proxy_gate': bool(finite and penetration < .003 and error < .005
                                      and min(coverage) >= .75)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--solref', type=float, default=.0015)
    parser.add_argument('--output', default='guided_flex_sweep.json')
    args = parser.parse_args()
    cases = [one(mass, friction, args.solref)
             for mass, friction in product((.015, .025, .05), (.4, .8, 1.2))]
    report = {'scope': 'Nine prewrapped, kinematic-guide material cases; no robot grasp or calibrated tension',
              'mass_kg': [.015, .025, .05], 'sliding_friction': [.4, .8, 1.2],
              'contact_solref': [args.solref, 1.],
              'proxy_gate': {'max_forearm_penetration_m': .003, 'max_endpoint_error_m': .005,
                             'min_coverage_per_turn': .75},
              'cases': cases, 'passed_cases': sum(c['passes_proxy_gate'] for c in cases)}
    (OUT / args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed_cases': report['passed_cases'], 'total_cases': len(cases),
                      'worst_penetration_m': max(c['max_forearm_penetration_m'] for c in cases),
                      'worst_endpoint_error_m': max(c['endpoint_error_m'] for c in cases),
                      'minimum_coverage': min(min(c['coverage_per_turn_12_bins']) for c in cases)}))
