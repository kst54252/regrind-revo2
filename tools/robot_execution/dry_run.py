"""Exercise real FK/IK and the I/O contract with a virtual clock and mock device."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from tools.rb3_revo2_ik.rb3_kinematics import RB3730Kinematics
from tools.rb3_revo2_ik.warm_start_ik import WarmStartIK
from tools.revo2_kinematics.revo2_kinematics import Revo2Kinematics
from .backends import MockBackend
from .contracts import DecodedTarget, Pose
from .session import ExecutionLimits, ExecutionSession


def main():
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=root / "config/robot_execution/mock.json")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--output", type=Path, required=True, help="New JSONL path; never overwritten")
    args = parser.parse_args()
    if args.steps < 60:
        parser.error("--steps must be >=60 for the fixed small smooth test motion")
    config = json.loads(args.config.read_text())
    if config["backend"] != "mock":
        parser.error("Only mock backend is available. Hardware execution is not commissioned.")
    kin, hand = RB3730Kinematics(), Revo2Kinematics()
    names = tuple(kin.joint_names) + tuple(hand.joint_names)
    arm_low, arm_high = kin.get_joint_limits()
    hand_low, hand_high = hand.get_joint_limits()
    limits = ExecutionLimits(np.r_[arm_low, hand_low], np.r_[arm_high, hand_high],
        np.r_[np.full(6, config['arm_max_speed_rad_s']), np.full(6, config['hand_max_speed_rad_s'])],
        np.r_[np.full(6, config['arm_max_acceleration_rad_s2']), np.full(6, config['hand_max_acceleration_rad_s2'])],
        np.full(12, config['max_tracking_error_rad']), config['control_period_s'], config['max_state_age_s'])
    q0 = np.r_[np.array([0., .5, -1., .5, .8, 0.]), (hand_low + hand_high) / 2]
    tick = [10.]
    clock = lambda: tick[0]
    backend = MockBackend(kin, names, q0, clock, frame=config['world_frame'])
    session = ExecutionSession(backend, kin, names, limits, frame=config['world_frame'],
                               solver=WarmStartIK(kin), clock=clock, require_object=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    wall_start = time.monotonic()
    max_pose_error = 0.
    def serial(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        raise TypeError(type(value).__name__)
    # Exclusive open BEFORE running; preserve previous logs.
    with args.output.open('x', encoding='utf-8') as log:
        log.write(json.dumps(dict(kind='mock_metadata', config=config, joint_names=names,
            hardware_connected=False, policy_executed=False, simulated_dynamics=False,
            virtual_clock=True, model=str(kin.model_config_path))) + '\n')
        try:
            session.start()
            for index in range(args.steps):
                tick[0] = 10. + (index + 1) * limits.period
                backend.advance(limits.period)
                u = (index + 1) / args.steps
                blend = 10*u**3 - 15*u**4 + 6*u**5
                desired = q0 + blend * np.array([.002, 0, -.002, 0, .002, 0] + [.002]*6)
                p, quat = kin.forward(desired[:6])
                def target_from_actual(state):
                    # Contract exercise only. A real policy adapter must consume
                    # this ACTUAL state, plus tracked object and its own history.
                    return DecodedTarget(Pose(p, quat, state.wrist.frame), desired[6:], names[6:],
                                         tick[0] + limits.period, index // 4)
                command = session.step(target_from_actual)
                max_pose_error = max(max_pose_error, session.last_result.position_error_m)
                row = dict(command=asdict(command), actual_before_send=asdict(session.last_state),
                           path_speed_rad_s=session.last_path_speed,
                           ik_position_error_m=session.last_result.position_error_m,
                           ik_rotation_error_rad=session.last_result.orientation_error_rad)
                log.write(json.dumps(row, default=serial) + '\n')
        except Exception as exc:
            log.write(json.dumps(dict(kind='fault', error=str(exc), status=session.status,
                                      stop_error=session.stop_error)) + '\n')
            raise
        finally:
            session.close()
    print(json.dumps(dict(status=session.status, commands=len(backend.sent),
        virtual_duration_s=args.steps * limits.period, wall_elapsed_s=time.monotonic()-wall_start,
        max_ik_position_error_m=max_pose_error, output=str(args.output),
        hardware_connected=False, policy_executed=False, physics_validated=False), indent=2))


if __name__ == '__main__':
    main()
