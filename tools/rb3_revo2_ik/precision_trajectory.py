"""Frozen joint-space precision benchmark, no simulator or policy dependency."""
import numpy as np


def single_joint_config(config, amplitude=.05):
    """Six isolated, slow out-and-back quintics; no online IK or ZOH stretch."""
    import copy
    result=copy.deepcopy(config)
    base=np.asarray(config['poses']['A'])
    result['poses']={'A':base.tolist()}
    sequence=[]
    for index,name in enumerate(config['joint_names']):
        target=base.copy();target[index]+=amplitude
        result['poses'][name]=target.tolist()
        result['poses'][name+'_return']=base.tolist()
        sequence.extend([name,name+'_return'])
    result.update(initial_offset_from_A_rad=[0.]*6,segment_duration_s=1.,
                  hold_duration_s=1.,selection_sequence=sequence,validation_sequence=sequence)
    return result


def quintic(start, end, time, duration):
    u=np.clip(np.asarray(time)/duration,0,1)
    delta=np.asarray(end)-np.asarray(start)
    s=10*u**3-15*u**4+6*u**5
    ds=(30*u**2-60*u**3+30*u**4)/duration
    dds=(60*u-180*u**2+120*u**3)/duration**2
    return np.asarray(start)+s[...,None]*delta,ds[...,None]*delta,dds[...,None]*delta


def build(config, held_out=False):
    dt=config['physics_dt'];t=0.;q0=np.asarray(config['poses']['A'])+config['initial_offset_from_A_rad']
    rows=[];phases=[]
    def append(label, duration, start, end, moving):
        nonlocal t
        count=round(duration/dt)
        if not np.isclose(count*dt,duration):raise ValueError('Duration must align with dt')
        phase=dict(name=label,start_s=t,end_s=t+duration,moving=moving,start_q=np.asarray(start).tolist(),end_q=np.asarray(end).tolist())
        phases.append(phase)
        for k in range(1,count+1):
            q,v,a=quintic(start,end,k*dt,duration)
            rows.append((t+k*dt,q,v,a,len(phases)-1))
        t+=duration
    append('initial_settle',config['initial_hold_s'],q0,q0,False)
    previous=q0
    for name in config['validation_sequence' if held_out else 'selection_sequence']:
        target=np.asarray(config['poses'][name])
        append('move_'+name,config['segment_duration_s'],previous,target,True)
        append('hold_'+name,config['hold_duration_s'],target,target,False)
        previous=target
    # Analytic maxima of the quintic, not merely sampled extrema.
    for phase in phases:
        d=np.abs(np.asarray(phase['end_q'])-phase['start_q']);duration=phase['end_s']-phase['start_s']
        if np.max(1.875*d/duration)>config['max_command_speed_rad_s'] or np.max((10/np.sqrt(3))*d/duration**2)>config['max_command_acceleration_rad_s2']:
            raise ValueError('Frozen motion exceeds engineering bounds')
    return q0,rows,phases


def command_rows(rows, q0, mode, substeps=4):
    """Offline command comparison, sharing one control-interval causal delay.

    At control boundary b the latest waypoint is q(b*dt). Linear mode ramps
    from the previous boundary waypoint to it over the next substeps ticks.
    Smooth mode reconstructs the same interval from the frozen quintic's
    analytical boundary derivatives (not available from arbitrary RL output).
    No joint-angle wrapping, finite-difference velocity feedforward or lookahead.
    """
    if mode == 'analytical':
        return rows
    if mode not in ('linear_zero', 'smooth_pv') or substeps < 1:
        raise ValueError('Invalid command mode/substeps')
    q0=np.asarray(q0);zero=np.zeros_like(q0)
    def sample(tick):
        return (q0,zero,zero) if tick <= 0 else rows[tick-1][1:4]
    result=[]
    for index,(t,_,_,_,phase) in enumerate(rows):
        tick=index+1
        if mode == 'smooth_pv':
            q,v,a=sample(tick-substeps)
        else:
            boundary=(index//substeps)*substeps
            start=sample(boundary-substeps)[0];end=sample(boundary)[0]
            q=start+(end-start)*(tick-boundary)/substeps
            v=zero
            # Linear corners have undefined instantaneous acceleration;
            # store path finite differences separately, not a fictitious a.
            a=zero
        result.append((t,q,v,a,phase))
    return result
