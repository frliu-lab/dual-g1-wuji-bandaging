"""Isolated MuJoCo flexible-strip contact pilot; no robots or wrap policy."""
import json
from pathlib import Path
import numpy as np
import mujoco

OUT = Path(__file__).resolve().parents[2] / 'outputs/wuji_multiagent/bandage_v0'
XML = '''<mujoco model="bandage_flex_contact_pilot">
  <option timestep="0.0005" iterations="100"/>
  <worldbody>
    <geom name="forearm_proxy" type="capsule" fromto="-.16 0 .08 .16 0 .08"
          size=".05" friction=".8 .005 .0001"/>
    <geom name="pilot_floor" type="plane" pos="0 0 -.05" size="1 1 .01"/>
    <flexcomp name="bandage_strip" type="grid" dim="2" count="15 3 1"
              spacing=".02 .02 .02" mass=".03" radius=".003"
              pos="-.14 -.02 .18" rgba=".95 .9 .75 1">
      <edge damping=".2"/>
      <contact solref=".005 1" friction=".8 .005 .0001"/>
    </flexcomp>
  </worldbody>
</mujoco>'''


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'flex_pilot.xml').write_text(XML, encoding='utf-8')
    model = mujoco.MjModel.from_xml_string(XML)
    data = mujoco.MjData(model)
    maximum_contacts = 0
    maximum_penetration = 0.
    forearm_contacts = 0
    forearm_penetration = 0.
    floor_contacts = 0
    floor_penetration = 0.
    final_forearm_penetration = 0.
    final_floor_penetration = 0.
    nonfinite = False
    for _ in range(round(.5 / model.opt.timestep)):
        mujoco.mj_step(model, data)
        maximum_contacts = max(maximum_contacts, data.ncon)
        maximum_penetration = max(maximum_penetration, max((max(0., -c.dist) for c in data.contact), default=0.))
        for c in data.contact:
            if model.geom('forearm_proxy').id in c.geom:
                forearm_contacts += 1
                forearm_penetration = max(forearm_penetration, max(0., -c.dist))
            if model.geom('pilot_floor').id in c.geom:
                floor_contacts += 1
                floor_penetration = max(floor_penetration, max(0., -c.dist))
        nonfinite |= not np.all(np.isfinite(data.qpos))
    for c in data.contact:
        if model.geom('forearm_proxy').id in c.geom:
            final_forearm_penetration = max(final_forearm_penetration, max(0., -c.dist))
        if model.geom('pilot_floor').id in c.geom:
            final_floor_penetration = max(final_floor_penetration, max(0., -c.dist))
    result = {
        'scope': 'Isolated deformable strip drop test only; no gripping or wrapping',
        'flex_count': model.nflex,
        'flex_vertices': model.nflexvert,
        'maximum_contacts': maximum_contacts,
        'maximum_penetration_m': maximum_penetration,
        'forearm_contact_samples': forearm_contacts,
        'forearm_maximum_penetration_m': forearm_penetration,
        'floor_contact_samples': floor_contacts,
        'floor_maximum_penetration_m': floor_penetration,
        'final_forearm_penetration_m': final_forearm_penetration,
        'final_floor_penetration_m': final_floor_penetration,
        'finite_after_half_second': not nonfinite,
    }
    (OUT / 'flex_pilot_check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))
    if model.nflex < 1 or maximum_contacts < 1 or nonfinite:
        raise RuntimeError('Flexible strip contact pilot failed')


if __name__ == '__main__':
    main()
