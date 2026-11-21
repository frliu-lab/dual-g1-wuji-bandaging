"""Measured-contact gate for the dressing transport phase."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PickupGate:
    ready: bool
    thumb_n: float
    index_n: float
    robot_up_n: float
    stage_n: float
    drift_m: float
    reasons: tuple[str, ...]


def evaluate_pickup_gate(frame, *, min_digit_n=0.04, min_up_n=0.10,
                         max_stage_n=0.06, max_drift_m=0.02):
    """Use measured MuJoCo forces; an absent contact counts as zero force."""
    digit = {'thumb': 0.0, 'index': 0.0}
    stage = 0.0
    for contact in frame.get('pad_contacts', ()):
        normal = max(0.0, float(contact['normal_force_n']))
        geoms = contact['geoms']
        bodies = contact['bodies']
        if 'dressing_stage' in geoms or 'equipment_tray' in geoms:
            stage += normal
        for name, prefix in (('thumb', 'g1_a/wuji_right/r_thumb_'),
                             ('index', 'g1_a/wuji_right/r_index_finger_')):
            if any(body.startswith(prefix) for body in bodies):
                digit[name] += normal
    up = sum(float(value) for body, value in
             frame.get('actual_pad_up_force_by_body_n', {}).items()
             if body.startswith('g1_a/wuji_right/'))
    drift = float(frame['pad_lateral_shift_m'])
    reasons = tuple(name for name, fails in (
        ('thumb_contact', digit['thumb'] < min_digit_n),
        ('index_contact', digit['index'] < min_digit_n),
        ('upward_load', up < min_up_n),
        ('stage_loaded', stage > max_stage_n),
        ('lateral_drift', drift > max_drift_m),
    ) if fails)
    return PickupGate(not reasons, digit['thumb'], digit['index'], up,
                      stage, drift, reasons)
