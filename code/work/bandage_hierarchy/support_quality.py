"""Conservative online gate for simulated training-forearm support.

Limits are provisional research screens. They are not clinical or hardware limits.
"""
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SupportSample:
    time_s: float
    elbow_rad: float
    elbow_speed_rad_s: float
    peak_force_n: float
    impulse_by_body_ns: dict[str, float]
    both_feet_supported: bool
    deep_nonfloor_contact: bool


class SupportQualityMonitor:
    def __init__(self, *, window_s=.5, max_peak_force_n=20.,
                 min_palm_share=.7, max_finger_share=.1):
        self.window_s=window_s
        self.max_peak_force_n=max_peak_force_n
        self.min_palm_share=min_palm_share
        self.max_finger_share=max_finger_share
        self.samples=deque()
        self.faults=set()
        self.valid=False
        self.last_report={}

    def update(self, sample: SupportSample) -> bool:
        if self.samples and sample.time_s<=self.samples[-1].time_s:
            raise ValueError('Support samples must have increasing time')
        if sample.peak_force_n>self.max_peak_force_n:
            self.faults.add('peak_force_exceeded')
        if sample.deep_nonfloor_contact:
            self.faults.add('deep_nonfloor_contact')
        if sample.elbow_rad<0.:
            self.faults.add('forearm_overlift')
        self.samples.append(sample)
        while self.samples and sample.time_s-self.samples[0].time_s>self.window_s:
            self.samples.popleft()
        full=bool(self.samples and sample.time_s-self.samples[0].time_s>=self.window_s-.025)
        total={}
        for frame in self.samples:
            for body,impulse in frame.impulse_by_body_ns.items():
                total[body]=total.get(body,0.)+impulse
        impulse=sum(total.values())
        palm=total.get('g1_a/wuji_left/l_wrist',0.)
        finger=sum(v for k,v in total.items() if '/l_' in k and 'finger' in k)
        window_ok=(full and impulse>=.5 and palm/impulse>=self.min_palm_share
                   and finger/impulse<=self.max_finger_share
                   and all(s.both_feet_supported for s in self.samples)
                   and all(0.<=s.elbow_rad<=.30 for s in self.samples)
                   and all(abs(s.elbow_speed_rad_s)<=.25 for s in self.samples)
                   and all(s.peak_force_n<=self.max_peak_force_n for s in self.samples))
        self.valid=bool(window_ok and not self.faults)
        self.last_report={'valid':self.valid,'window_full':full,
                          'window_palm_impulse_share':palm/max(impulse,1e-9),
                          'window_finger_impulse_share':finger/max(impulse,1e-9),
                          'window_total_impulse_ns':impulse,
                          'latched_faults':sorted(self.faults)}
        return self.valid
