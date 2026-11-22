"""Verify direct QP honors arbitrary task rows and per-row weights."""
import numpy as np
from direct_wbc_qp import solve_direct

base={
    'M':np.eye(1),'B':np.eye(1),'J':np.zeros((0,1)),
    'mask':np.array([],dtype=int),'pad_flags':(),
    'target':np.array([]),'weights':np.ones(1),
    'desired':np.zeros(1),'taskJ':np.ones((2,1)),
    'tasktarget':np.array([0.,1.]),
    'fixed_hand':np.array([],dtype=int),'forceweight':np.array([]),
    'force_goal_weighted':np.array([]),'h':np.zeros(1),
    'lo':np.array([-10.]),'hi':np.array([10.]),
    'acc_lo':np.array([-10.]),'acc_hi':np.array([10.]),
    'mu':np.array([]),
}
out=[]
for weights in ([1.,1.],[1.,10.]):
    data=base|{'task_weight':np.array(weights)}
    acceleration,torque,_,_=solve_direct(data)
    # The single acceleration minimizes a² + w0² a² + w1²(a-1)².
    expected=weights[1]**2/(1+weights[0]**2+weights[1]**2+1e-4)
    assert abs(acceleration[0]-expected)<1e-5,(acceleration,expected)
    assert abs(torque[0]-acceleration[0])<1e-6
    out.append(float(acceleration[0]))
assert out[1]>out[0]+.5,out
print({'passed':True,'two_row_accelerations':out})
