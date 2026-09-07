"""Compare the opt-in IK against recorded verified IK inputs and solutions."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.warm_start_ik import WarmStartIK


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--stride',type=int,default=10)
    a=p.parse_args()
    if a.stride<1:raise ValueError('Stride must be positive')
    if a.output.exists():raise FileExistsError(a.output)
    m=json.loads((a.source/'metadata.json').read_text())
    with Path(m['state_bank']).open() as f:bank=json.loads(next(f))
    kin=RB3730Kinematics(base_position=bank['base_position'],base_quaternion_xyzw=bank['base_quaternion_xyzw'])
    fast=WarmStartIK(kin);records=[];last_episode=None
    with (a.source/'physics.jsonl').open() as f:
        for index,line in enumerate(f):
            r=json.loads(line);ep=r['episode']
            if ep!=last_episode:
                previous=np.asarray(m['initial_states'][ep]['all_q'])[m['arm_ids']];last_episode=ep
            if index%a.stride==0:
                qactual=np.asarray(r['controller']['pre_state']['all_q'])[m['arm_ids']]
                start=time.perf_counter()
                result=fast.inverse(r['ik_input_pos'],r['ik_input_quat'],initial_q=previous,neutral_q=qactual,max_nfev=300)
                records.append(dict(episode=ep,step=r['episode_step'],time_s=time.perf_counter()-start,
                    success=result.success,position_error_m=result.position_error_m,rotation_error_rad=result.orientation_error_rad,
                    max_joint_difference_rad=float(np.max(abs(result.q-r['q_ik']))),candidates=result.candidates_evaluated))
            previous=np.asarray(r['q_ik'])
    report=dict(source=str(a.source),stride=a.stride,records=records,fallbacks=fast.fallbacks)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print('samples',len(records),'failures',sum(not r['success'] for r in records),
          'max joint difference rad',max(r['max_joint_difference_rad'] for r in records),
          'mean ms',np.mean([r['time_s'] for r in records])*1000,'fallbacks',fast.fallbacks)


if __name__=='__main__':main()
