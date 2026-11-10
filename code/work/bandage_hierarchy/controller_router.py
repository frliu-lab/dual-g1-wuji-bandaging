"""Control arbitration for a future two-G1 bandaging simulation.

WBC remains responsible for whole-body dynamics and contact support. A hand
specialist may supply finger joint targets only. This module sends no commands
to hardware and does not claim a trained bandaging policy exists.
"""
from dataclasses import dataclass
from enum import Enum
import numpy as np


class Mode(str, Enum):
    BODY = 'body_wbc'
    HAND = 'hand_specialist'
    HOLD = 'hold_last_safe_fingers'


@dataclass(frozen=True)
class Observation:
    time_s: float
    phase: str
    body_valid: bool
    feet_supported: bool
    arm_pose_stable: bool
    hand_contacts: tuple[bool, bool, bool, bool]
    hand_feedback_fresh: bool
    dressing_grasp_valid: bool = False
    forearm_support_valid: bool = False


@dataclass(frozen=True)
class HandProposal:
    time_s: float
    positions_rad: np.ndarray
    confidence: float


@dataclass(frozen=True)
class Decision:
    mode: Mode
    finger_targets_rad: np.ndarray
    specialist_weight: float
    reason: str


class HandExpertRouter:
    """Blend a hand expert in only after contact and posture are stable."""

    FINE_PHASES = frozenset({'place_dressing', 'wrap', 'secure_end'})
    # Order: A-left support, A-right dressing, B-left spool, B-right guide.
    REQUIRED_CONTACTS = {
        'place_dressing': (0, 1),
        'wrap': (0, 1, 2, 3),
        'secure_end': (0, 1, 3),
    }

    def __init__(self, finger_count: int, *, blend_s: float = .20,
                 max_joint_speed_rad_s: float = .8, max_age_s: float = .04,
                 stable_s: float = .25):
        if finger_count <= 0 or min(blend_s, max_joint_speed_rad_s, max_age_s, stable_s) <= 0:
            raise ValueError('All router dimensions and limits must be positive')
        self.n = finger_count
        self.blend_s = blend_s
        self.max_speed = max_joint_speed_rad_s
        self.max_age_s = max_age_s
        self.stable_s = stable_s
        self.last_t: float | None = None
        self.stable_since: float | None = None
        self.weight = 0.
        self.last_safe: np.ndarray | None = None

    def step(self, obs: Observation, wbc_fingers_rad: np.ndarray,
             proposal: HandProposal | None) -> Decision:
        base = np.asarray(wbc_fingers_rad, dtype=float)
        if base.shape != (self.n,) or not np.all(np.isfinite(base)):
            raise ValueError('WBC finger target shape or values invalid')
        if self.last_t is not None and obs.time_s <= self.last_t:
            raise ValueError('Observation time must increase')
        dt = 0. if self.last_t is None else obs.time_s-self.last_t
        self.last_t = obs.time_s
        if len(obs.hand_contacts) != 4 or any(type(c) is not bool for c in obs.hand_contacts):
            raise ValueError('Four per-hand contact booleans are required')
        required = self.REQUIRED_CONTACTS.get(obs.phase, ())
        contact_ready = all(obs.hand_contacts[i] for i in required)
        grasp_ready = obs.dressing_grasp_valid if obs.phase in self.FINE_PHASES else True
        support_ready = obs.forearm_support_valid if obs.phase in self.FINE_PHASES else True
        stable = (obs.body_valid and obs.feet_supported and obs.arm_pose_stable
                  and contact_ready and grasp_ready and support_ready)
        if stable:
            if self.stable_since is None:
                self.stable_since = obs.time_s
        else:
            self.stable_since = None
        valid_proposal = (proposal is not None and proposal.positions_rad.shape == (self.n,)
                          and np.all(np.isfinite(proposal.positions_rad))
                          and 0. <= obs.time_s-proposal.time_s <= self.max_age_s
                          and proposal.confidence >= .8 and obs.hand_feedback_fresh)
        ready = (obs.phase in self.FINE_PHASES and stable and self.stable_since is not None
                 and obs.time_s-self.stable_since >= self.stable_s and valid_proposal)
        if not obs.body_valid or not obs.feet_supported:
            self.weight = 0.
            target = self.last_safe.copy() if self.last_safe is not None else base.copy()
            return Decision(Mode.HOLD, target, 0., 'body_or_support_guard')
        if obs.phase in self.FINE_PHASES and self.weight > 0 and not stable:
            self.weight = 0.
            target = self.last_safe.copy() if self.last_safe is not None else base.copy()
            return Decision(Mode.HOLD, target, 0., 'fine_contact_or_arm_pose_lost')
        if obs.phase in self.FINE_PHASES and self.weight > 0 and not valid_proposal:
            self.weight = 0.
            target = self.last_safe.copy() if self.last_safe is not None else base.copy()
            return Decision(Mode.HOLD, target, 0., 'hand_proposal_or_feedback_stale')
        goal_weight = 1. if ready else 0.
        delta = min(1., dt/self.blend_s) if dt else 0.
        self.weight += np.clip(goal_weight-self.weight, -delta, delta)
        expert = np.asarray(proposal.positions_rad, dtype=float) if ready else base
        target = (1.-self.weight)*base+self.weight*expert
        if self.last_safe is not None:
            target = np.clip(target, self.last_safe-self.max_speed*dt,
                             self.last_safe+self.max_speed*dt)
        self.last_safe = target.copy()
        return Decision(Mode.HAND if self.weight > 0 else Mode.BODY,
                        target, float(self.weight), 'fine_contact_ready' if ready else 'wbc_only')
