import numpy as np
from pydrake.all import ClarabelSolver

from regrind.retargeting import revo2_constants as rc
from regrind.retargeting.retargeter import HandInteractionMeshOneStageRetargeter


def _make_retargeter():
    object_config = rc.OBJECT_CONFIGS["tuna_fish_can"]
    return HandInteractionMeshOneStageRetargeter(
        robot_model_path=rc.ROBOT_URDF_FILE,
        robot_name=rc.ROBOT_NAME,
        object_model_path=object_config["object_urdf_file"],
        object_name="tuna_fish_can",
        object_body_name=object_config["object_body_name"],
        q_a_init_idx=-7,
        activate_joint_limits=True,
        activate_obj_non_penetration=False,
        demo_joints=rc.MANO_JOINTS,
        laplacian_match_links=rc.MANO_TO_REVO2_MAPPING,
        solver=ClarabelSolver(),
        visualize=False,
        semantic_keypoints=rc.SEMANTIC_KEYPOINTS,
        independent_joint_names=rc.ACTUATED_JOINT_NAMES,
        mimic_joints=rc.MIMIC_JOINTS,
        robot_model_dof=rc.ROBOT_MODEL_DOF,
    )


def test_failed_frame_is_nan_and_next_frame_uses_last_success(monkeypatch):
    retargeter = _make_retargeter()
    object_points = np.load(
        rc.OBJECT_CONFIGS["tuna_fish_can"]["object_keypoints_paths"]["bottom"]
    )
    object_poses = np.tile(
        np.array([1.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0]), (3, 1)
    )
    initial = np.zeros(retargeter.nq)
    initial[0] = 1.0
    initial[retargeter.robot_position_count :] = object_poses[0]
    human = np.tile(
        retargeter._get_robot_link_positions(
            initial, rc.MANO_TO_REVO2_MAPPING
        )[None],
        (3, 1, 1),
    )

    warm_starts = []

    def fake_iterate(**kwargs):
        frame = len(warm_starts)
        warm_starts.append(np.array(kwargs["q_n"], copy=True))
        if frame == 1:
            retargeter.last_solver_success = False
            retargeter.solver_failure_count += 1
            raise RuntimeError("intentional test failure")
        q = np.array(kwargs["q_locked"], copy=True)
        q[retargeter.q_a_indices] = kwargs["q_n"][retargeter.q_a_indices]
        retargeter._apply_mimic_positions(q)
        retargeter.last_solver_success = True
        retargeter.last_objective = float(frame)
        return q, float(frame)

    monkeypatch.setattr(retargeter, "iterate", fake_iterate)
    q_a_init = np.concatenate(
        (np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.zeros(6))
    )
    motions, *_ = retargeter.retarget_motion(
        human_joint_motions=human,
        object_poses=object_poses,
        object_poses_augmented=object_poses.copy(),
        object_points_local_demo=object_points,
        object_points_local=object_points,
        q_a_init=q_a_init,
    )

    assert motions.shape == (3, retargeter.nq)
    assert np.isfinite(motions[0]).all()
    assert np.isnan(motions[1]).all()
    assert np.isfinite(motions[2]).all()
    assert retargeter.failure_indices == [1]
    assert [item["warm_start_frame"] for item in retargeter.frame_diagnostics] == [
        None,
        0,
        0,
    ]
    np.testing.assert_allclose(
        warm_starts[2][retargeter.q_a_indices],
        motions[0][retargeter.q_a_indices],
    )
