import numpy as np
import time
from tqdm import tqdm

from pydrake.all import (
    StartMeshcat,
    MathematicalProgram,
    SolverOptions,
    CommonSolverOption,
    MeshcatVisualizer,
    MeshcatVisualizerParams,
    Role,
    JacobianWrtVariable,
    MosekSolver,
    ClarabelSolver,
    RigidTransform,
    InitializeAutoDiff,
    ExtractGradient,
    ExtractValue,
    Quaternion,
    Sphere,
    Rgba,
)


def _robot_resources(robot_name: str):
    """Return default joint names list, actuated DOF count, and collision setup for a robot."""
    if robot_name == "leaphand":
        from regrind.retargeting.leaphand_constants import MANO_JOINTS, ROBOT_DOF
        from regrind.retargeting.drake_utils import _setup_leaphand_collision_exclusions

        return MANO_JOINTS, ROBOT_DOF, _setup_leaphand_collision_exclusions
    if robot_name == "wujihand":
        from regrind.retargeting.wujihand_constants import MANO_JOINTS, ROBOT_DOF
        from regrind.retargeting.drake_utils import _setup_wujihand_collision_exclusions

        return MANO_JOINTS, ROBOT_DOF, _setup_wujihand_collision_exclusions
    if robot_name == "revo2":
        from regrind.retargeting.revo2_constants import MANO_JOINTS, ROBOT_DOF

        return MANO_JOINTS, ROBOT_DOF, None
    raise ValueError(
        f"Unknown robot_name {robot_name!r}; expected 'leaphand', 'wujihand', "
        "or 'revo2'."
    )


from regrind.retargeting.drake_utils import (
    create_plant,
    transform_points_world_to_local,
    transform_points_local_to_world,
    rotate_points_along_axis,
    get_adjacency_list,
    calculate_laplacian_coordinates,
    create_interaction_mesh,
    calculate_laplacian_matrix,
)


class HandInteractionMeshOneStageRetargeter:
    """
    A class to perform kinematic retargeting from human motion to a robot,
    preserving spatial relationships using an interaction mesh.
    """

    def __init__(
        self,
        robot_model_path: str,
        robot_name: str = "wujihand",
        object_model_path: str | None = None,
        object_name: str = "largebox",
        table_height: float | None = None,
        q_a_init_idx: int = -7,
        activate_obj_non_penetration: bool = True,
        activate_joint_limits: bool = True,
        step_size: float = 0.2,
        collision_detection_threshold: float = 0.1,
        penetration_tolerance: float = 1e-3,
        solver: MosekSolver | ClarabelSolver = MosekSolver(),
        visualize: bool = False,
        demo_joints: list[str] | None = None,
        laplacian_match_links: dict[str, str] = {},
        smplh: bool = True,
        debug: bool = False,
        w_nominal_tracking_init: float = 5.0,
        nominal_tracking_tau: float = 10.0,
        track_nominal_indices: np.ndarray | None = None,
        hand_keypoint_weight: dict[str, float] | None = None,
        semantic_keypoints: dict[str, dict] | None = None,
        independent_joint_names: tuple[str, ...] | None = None,
        mimic_joints: dict[str, tuple[str, float, float]] | None = None,
        robot_model_dof: int | None = None,
        object_body_name: str | None = None,
    ):
        """This kinematic retargeter solves the diffIK problem with hard constraints in SQP style.
        During each SQP iteration, the problem is solved with the following constraints and costs:
            1. [Cost] Minimize the Laplacian deformation in the object frame.
            2. [Constraint] Enforce the non-penetration constraints w/ the ground and (if activated) the object.
            3. [Constraint] Enforce the foot sticking constraints if activated.
            4. [Constraint] Enforce the joint limits if activated.
            5. [Constraint] Enforce trust region of dq.
        The constraints are linearized and the costs are quadratic with a trust region.

        Args:
            q_a_init_idx: the index in robot's configuration where the optimization variables start. -7: starts from the
            floating base, -3: starts from the translation of the floating base, 0: starts from the actuated DOF,
            12: starts from waist, 15: starts from left shoulder
            step_size: trust region for each SQP iteration.
            collision_detection_threshold: only start to detect collision when the distance is smaller than this threshold.
            penetration_tolerance: tolerance for penetration when enforcing non-penetration constraints.
            contact_matching_threshold: only start to match contact when the distance is smaller than this threshold.
            ik_match_links: match keypoints between robot and human (IK).
            contact_match_links : match the links that making contact.
            nominal_tracking_tau: the time constant for the nominal tracking cost.
            track_nominal_indices: Row indices into ``q_a`` for the nominal-tracking cost.
                ``None`` (default) uses ``np.arange(self.nq_a)``, i.e. every optimized scalar;
                ``self.nq_a`` follows ``robot_dof`` and ``q_a_init_idx``. For joint-only tracking
                with ``q_a_init_idx=-7``, pass ``np.arange(7, 7 + robot_dof)``.
            robot_name: selects kinematic constants and collision exclusions
                (``'leaphand'`` or ``'wujihand'``).
            hand_keypoint_weight: optional multipliers (per MANO keypoint id string,
                e.g. ``"0"`` through ``"20"``) for hand vertices in the Laplacian cost
                and in the Laplacian operator (neighbor centroid weights). Keys must be
                a subset of ``laplacian_match_links``; missing keys default to ``1.0``.
                Object vertices use weight ``1.0``.
        """
        default_demo_joints, robot_dof, setup_collision_exclusions = _robot_resources(
            robot_name
        )
        self.robot_name = robot_name
        self.robot_dof = robot_dof
        if demo_joints is None:
            demo_joints = list(default_demo_joints)

        self.robot_model_path = robot_model_path
        self.object_model_path = object_model_path
        self.object_name = object_name
        self.object_body_name = object_body_name
        self.collision_detection_threshold = collision_detection_threshold
        self.activate_obj_non_penetration = activate_obj_non_penetration
        self.activate_joint_limits = activate_joint_limits
        self.penetration_tolerance = penetration_tolerance
        self.step_size = step_size
        self.solver = solver
        self.visualize = visualize
        self.debug = debug
        self.smplh = smplh
        self.semantic_keypoints = semantic_keypoints or {}
        self.independent_joint_names = tuple(independent_joint_names or ())
        self.mimic_joints = dict(mimic_joints or {})
        self.robot_model_dof = robot_model_dof or self.robot_dof
        self.robot_position_count = 7 + self.robot_model_dof
        self.last_solver_success = None
        self.last_objective = np.nan
        self.solver_failure_count = 0
        self.frame_diagnostics = []
        self.failure_indices = []

        # Setup Drake system
        self.plant, scene_graph, builder = create_plant(
            robot_model_path,
            object_model_path,
            collision_exclusion_setter=setup_collision_exclusions,
            table_height=table_height,
        )
        self.nq = self.plant.num_positions()

        # Initialize foot links and joint mappings
        self._initialize_mappings(laplacian_match_links, demo_joints)

        if hand_keypoint_weight is None:
            self._hand_keypoint_laplacian_mult = None
        else:
            mesh_keys = set(self.laplacian_match_links.keys())
            unknown = set(hand_keypoint_weight) - mesh_keys
            if unknown:
                raise ValueError(
                    "hand_keypoint_weight has MANO ids not in laplacian_match_links: "
                    f"{sorted(unknown)}"
                )
            mult = []
            for k in self.laplacian_match_links:
                w = float(hand_keypoint_weight.get(k, 1.0))
                if not np.isfinite(w) or w < 0:
                    raise ValueError(
                        f"hand_keypoint_weight[{k!r}] must be finite and >= 0, got {w}"
                    )
                mult.append(w)
            self._hand_keypoint_laplacian_mult = np.array(mult, dtype=float)

        # Setup visualization if requested
        if self.visualize:
            self._setup_visualization(builder, scene_graph)

        # Setup Drake system contexts
        self._setup_drake_contexts(builder, scene_graph)

        # Setup optimization parameters
        self._setup_plant_params(q_a_init_idx)
        self.w_nominal_tracking_init = w_nominal_tracking_init
        self.nominal_tracking_tau = nominal_tracking_tau
        if track_nominal_indices is None:
            self.track_nominal_indices = np.arange(self.nq_a)
        else:
            tn = np.asarray(track_nominal_indices, dtype=int)
            if tn.ndim != 1:
                raise ValueError("track_nominal_indices must be a 1-D array of integers.")
            if np.any((tn < 0) | (tn >= self.nq_a)):
                raise ValueError(
                    "track_nominal_indices must be in [0, nq_a) where "
                    f"nq_a={self.nq_a}; got min={tn.min()}, max={tn.max()}."
                )
            self.track_nominal_indices = tn

    def _initialize_mappings(
        self, laplacian_match_links: dict[str, str], demo_joints: list[str]
    ):
        """Initialize foot links and joint mappings."""

        self.demo_joints = demo_joints
        self.laplacian_match_links = laplacian_match_links

        self.mano_mapped_joint_indices = [
            self.demo_joints.index(name) for name in self.laplacian_match_links.keys()
        ]

        # Setup weights and parameters
        self.laplacian_weights = 10
        self.smooth_weight = 0.5

    def _setup_visualization(self, builder, scene_graph):
        """Setup visualization components."""
        self.meshcat = StartMeshcat()
        self._meshcat_created_paths = set()
        self.vis = MeshcatVisualizer.AddToBuilder(builder, scene_graph, self.meshcat)

        proximity_vis_params = MeshcatVisualizerParams()
        proximity_vis_params.role = Role.kProximity
        proximity_vis_params.prefix = "proximity"
        proximity_vis_params.visible_by_default = False
        self.proximity_vis = MeshcatVisualizer.AddToBuilder(
            builder, scene_graph, self.meshcat, proximity_vis_params
        )

    def _setup_drake_contexts(self, builder, scene_graph):
        """Setup Drake system contexts."""
        self.diagram = builder.Build()
        self.context = self.diagram.CreateDefaultContext()
        self.plant_context = self.plant.GetMyMutableContextFromRoot(self.context)

        sg_context = scene_graph.GetMyMutableContextFromRoot(self.context)
        self.query_object = scene_graph.get_query_output_port().Eval(sg_context)
        self.inspector = self.query_object.inspector()

        # Setup autodiff contexts
        diagram_ad = self.diagram.ToAutoDiffXd()
        context_ad = diagram_ad.CreateDefaultContext()
        self.plant_ad = diagram_ad.GetSubsystemByName(self.plant.get_name())
        self.plant_context_ad = self.plant_ad.GetMyMutableContextFromRoot(context_ad)
        self.sg_ad = diagram_ad.GetSubsystemByName(scene_graph.get_name())
        self.sg_context_ad = self.sg_ad.GetMyMutableContextFromRoot(context_ad)
        self.query_object_ad = self.sg_ad.get_query_output_port().Eval(
            self.sg_context_ad
        )
        self.inspector_ad = self.query_object_ad.inspector()

        if hasattr(self, "vis"):
            self.vis_context = self.vis.GetMyMutableContextFromRoot(self.context)
            self.proximity_vis_context = self.proximity_vis.GetMyMutableContextFromRoot(
                self.context
            )

    def _setup_plant_params(self, q_a_init_idx: int = -7, q_a_pad: float = 0.9):
        """Setup optimization parameters and indices."""
        self.q_a_init_idx = q_a_init_idx
        self._mimic_position_map = []

        if self.independent_joint_names:
            if q_a_init_idx != -7:
                raise ValueError(
                    "robot-specific independent joints currently require "
                    "q_a_init_idx=-7 so wrist pose and hand joints are optimized together"
                )
            independent_indices = [
                self.plant.GetJointByName(name).position_start()
                for name in self.independent_joint_names
            ]
            if len(set(independent_indices)) != self.robot_dof:
                raise ValueError(
                    "independent_joint_names must resolve to exactly "
                    f"{self.robot_dof} unique scalar positions"
                )
            self.actuated_position_indices = np.asarray(independent_indices, dtype=int)
            self.q_a_indices = np.concatenate(
                (np.arange(7, dtype=int), self.actuated_position_indices)
            )

            for follower_name, (leader_name, multiplier, offset) in self.mimic_joints.items():
                follower = self.plant.GetJointByName(follower_name)
                leader = self.plant.GetJointByName(leader_name)
                if follower.num_positions() != 1 or leader.num_positions() != 1:
                    raise ValueError(
                        f"mimic joints must be scalar: {follower_name}, {leader_name}"
                    )
                self._mimic_position_map.append(
                    (
                        follower.position_start(),
                        leader.position_start(),
                        float(multiplier),
                        float(offset),
                    )
                )
        else:
            self.q_a_indices = np.arange(7 + self.q_a_init_idx, 7 + self.robot_dof)
            self.actuated_position_indices = np.arange(7, 7 + self.robot_dof)

        self.nq_a = len(self.q_a_indices)
        self.q_a_lb = self.plant.GetPositionLowerLimits()[self.q_a_indices]
        self.q_a_ub = self.plant.GetPositionUpperLimits()[self.q_a_indices]

        self.q_a_lb[:4] = -1
        self.q_a_ub[:4] = 1

    def _apply_mimic_positions(self, q):
        """Synchronize dependent positions without introducing optimizer variables."""
        for follower, leader, multiplier, offset in self._mimic_position_map:
            q[follower] = offset + multiplier * q[leader]
        return q

    def _project_jacobian_to_independent(self, jacobian: np.ndarray) -> np.ndarray:
        """Apply mimic chain rule, then select wrist + six independent columns."""
        projected = np.array(jacobian, copy=True)
        for follower, leader, multiplier, _ in self._mimic_position_map:
            projected[..., leader] += multiplier * projected[..., follower]
        return projected[..., self.q_a_indices]

    def get_actuated_joint_positions(self, q: np.ndarray) -> np.ndarray:
        """Extract only independently actuated joints from one or many plant states."""
        values = np.asarray(q)
        return values[..., self.actuated_position_indices]

    def _verify_initial_q_configuration(
        self,
        q: np.ndarray,
        q_a_init: np.ndarray | None,
        articulated_object: bool,
    ) -> None:
        """Check nq layout, hand base in q vs q_a_init, and SetPositions round-trip."""
        nq_hand = self.robot_position_count
        expected_obj_dofs = 8 if articulated_object else 7
        expected_nq = nq_hand + expected_obj_dofs
        tail = self.nq - nq_hand

        print("\n[verify initial q] plant ↔ assembled q_locked_list[0]")
        print(f"  plant.num_positions() (nq) = {self.nq}")
        print(f"  len(q) = {len(q)}")
        print(
            f"  assumed layout: hand floating+joints = {nq_hand}, "
            f"object DOFs expected = {expected_obj_dofs}, total nq expected = {expected_nq}"
        )
        print(f"  actual tail size (nq - nq_hand) = {tail}")
        if self.nq != expected_nq:
            print(
                "  WARNING: nq != nq_hand + object DOFs — trailing slices (-7:/-8:) may "
                "overlap hand indices 0:{}.".format(nq_hand - 1)
            )
        if tail != expected_obj_dofs:
            print(
                "  WARNING: tail size does not match articulated vs free object assumption."
            )

        print(f"  q[0:7] (hand base quat_wxyz + xyz):\n    {q[:7]}")
        if q_a_init is not None:
            ref = np.asarray(q_a_init, dtype=float).ravel()
            print(f"  q_a_init[0:7]:\n    {ref[:7]}")
            if len(ref) >= 7 and not np.allclose(q[:7], ref[:7], rtol=0.0, atol=1e-9):
                print(
                    "  WARNING: q[0:7] != q_a_init[0:7] — hand base was overwritten or "
                    "indices do not match q_a_init layout."
                )

        self.plant.SetPositions(self.plant_context, q)
        q_read = self.plant.GetPositions(self.plant_context)

        def _normalize_hand_quat(v: np.ndarray) -> np.ndarray:
            out = np.array(v, dtype=float, copy=True)
            n = np.linalg.norm(out[:4])
            if n > 0:
                out[:4] /= n
            return out

        q_cmp = _normalize_hand_quat(q)
        q_read_cmp = _normalize_hand_quat(q_read)
        round_trip_hand = np.allclose(q_read_cmp[:7], q_cmp[:7], rtol=0.0, atol=1e-7)
        if not round_trip_hand:
            print(
                "  WARNING: SetPositions/GetPositions round-trip mismatch on hand base "
                f"(max abs diff {np.max(np.abs(q_read_cmp[:7] - q_cmp[:7]))})."
            )
        else:
            print("  SetPositions / GetPositions round-trip OK for q[0:7].")

        if self.nq > nq_hand:
            idx0 = nq_hand
            if self.nq - idx0 >= 4:
                qo = q_read_cmp[idx0 : idx0 + 4]
                qro = q_cmp[idx0 : idx0 + 4]
                obj_quat_ok = np.allclose(qro, qo, rtol=0.0, atol=1e-7)
                if not obj_quat_ok:
                    print(
                        "  WARNING: object quaternion block round-trip mismatch at "
                        f"q[{idx0}:{idx0 + 4}]."
                    )
                else:
                    print(f"  Object quaternion block q[{idx0}:{idx0 + 4}] round-trip OK.")

    def retarget_motion(
        self,
        human_joint_motions,
        object_poses,
        object_poses_augmented,
        object_points_local_demo,
        object_points_local,
        object_points_local_demo_2=None,  # for articulated object
        object_points_local_2=None,  # for articulated object
        object_joints=None,  # for articulated object
        q_a_init=None,
        q_nominal_list=None,
        original=True,
    ):
        """
        The main function to retarget an entire motion sequence frame by frame.

        Args:
            human_joint_motions (np.ndarray): (num_frames, num_joints, 3) array.
            object_poses (np.ndarray): (num_frames, 7) array of demo object poses (quat, trans).
            object_poses_augmented (np.ndarray): (num_frames, 7) array of augmented object poses (quat, trans).
            object_points_local_demo (np.ndarray): Demo object points in local frame.
            object_points_local (np.ndarray): Current object points in local frame.
            q_a_init (np.ndarray, optional): Initial robot configuration.
            q_nominal_list (np.ndarray, optional): Per-frame nominal ``q`` (e.g. stage-1 result).

        Returns:
            tuple: (retargeted_motions, obj_pts_demo_list, obj_pts_list, tetrahedra,
                obj_pts_demo_2_list, obj_pts_list_2)
            The ``obj_pts_*_2`` lists are world-frame positions for an optional second
            object keypoint set (same vertex order as ``all_points``' third block);
            empty when that block is not used. Populated only when ``self.debug`` is True.
        """
        num_frames = human_joint_motions.shape[0]
        if q_nominal_list is not None:
            q_locked_list = q_nominal_list
        else:
            q_locked_list = np.zeros((num_frames, self.nq))
            q_locked_list[0, self.q_a_indices] = q_a_init

        if object_joints is None:
            q_locked_list[:, -7:] = object_poses_augmented
        else:
            q_locked_list[:, -8:-1] = object_poses_augmented
            q_locked_list[:, -1] = object_joints[:, 0]
        q = np.copy(q_locked_list[0])
        self._apply_mimic_positions(q)
        if self.debug:
            self._verify_initial_q_configuration(
                q,
                None if q_nominal_list is not None else q_a_init,
                object_joints is not None,
            )
        # ``q`` is always the last *successful* solution.  Failed frames are
        # represented by NaNs in the returned trajectory and never become a
        # warm start for the following frame.
        last_success_q = np.array(q, copy=True)
        last_success_frame = None
        retargeted_motions = []
        self.frame_diagnostics = []
        self.failure_indices = []
        self.solver_failure_count = 0

        tetrahedra = []
        obj_pts_demo_list = []
        obj_pts_list = []
        obj_pts_demo_2_list = []
        obj_pts_list_2 = []

        print(f"\nStarting motion retargeting for {num_frames} frames...")

        with tqdm(range(num_frames)) as pbar:
            for i in pbar:
                if self.visualize:
                    self.draw_q(q)
                # Get object poses and transform points
                object_quat_demo = object_poses[i, :4]
                object_trans_demo = object_poses[i, 4:]
                object_quat = object_poses_augmented[i, :4]
                object_trans = object_poses_augmented[i, 4:]
                object_joint = object_joints[i] if object_joints is not None else None  # articulation

                # Get human joint positions and create interaction mesh in object frame
                human_mapped_joints = human_joint_motions[
                    i, self.mano_mapped_joint_indices
                ]
                if self.object_name == "ground" or self.object_name == "multi_boxes":
                    human_mapped_joints_in_object = human_mapped_joints
                else:
                    human_mapped_joints_in_object = transform_points_world_to_local(
                        object_quat_demo, object_trans_demo, human_mapped_joints
                    )

                all_points = [human_mapped_joints_in_object, object_points_local_demo]
                if object_joints is not None:
                    object_points_local_demo_2_rotated = rotate_points_along_axis(
                        object_points_local_demo_2, [0, 0, -1], object_joint
                    )  # TODO: visualize the rotated object points to check if the rotation is correct
                    all_points.append(object_points_local_demo_2_rotated)
                    object_points_local_2_rotated = rotate_points_along_axis(
                        object_points_local_2, [0, 0, -1], object_joint
                    )
                else:
                    object_points_local_demo_2_rotated = None
                    object_points_local_2_rotated = None

                source_vertices, source_tetrahedra = create_interaction_mesh(
                    np.vstack(all_points)
                )
                tetrahedra.append(source_tetrahedra)

                if self.debug:
                    # NOTE: visualization is wrong for the flagship data since object_points_local
                    # are on the terrain but object_pose in the chair's.
                    obj_pts_demo = transform_points_local_to_world(
                        object_quat_demo, object_trans_demo, object_points_local_demo
                    )
                    obj_pts = transform_points_local_to_world(
                        object_quat, object_trans, object_points_local
                    )

                    obj_pts_demo_list.append(obj_pts_demo)
                    obj_pts_list.append(obj_pts)
                    self.draw_keypoints(human_mapped_joints, name="human")
                    self.draw_keypoints(
                        obj_pts_demo, name="object_demo", rgba=(1, 0, 0, 1), size=0.005
                    )
                    self.draw_keypoints(obj_pts, name="object", rgba=(0, 1, 1, 1), size=0.005)

                    if object_points_local_demo_2_rotated is not None:
                        obj_pts_demo_2 = transform_points_local_to_world(
                            object_quat_demo, object_trans_demo, object_points_local_demo_2_rotated
                        )
                        self.draw_keypoints(obj_pts_demo_2, name="object_demo_2", rgba=(1, 0, 0, 1), size=0.005)
                        obj_pts_demo_2_list.append(obj_pts_demo_2)
                    if object_points_local_2_rotated is not None:
                        obj_pts_2 = transform_points_local_to_world(
                            object_quat, object_trans, object_points_local_2_rotated
                        )
                        self.draw_keypoints(obj_pts_2, name="object_2", rgba=(0, 1, 1, 1), size=0.005)
                        obj_pts_list_2.append(obj_pts_2)

                # Create adjacency list and calculate target Laplacian coordinates
                adj_list = get_adjacency_list(source_tetrahedra, len(source_vertices))
                if self._hand_keypoint_laplacian_mult is None:
                    target_laplacian = calculate_laplacian_coordinates(
                        source_vertices, adj_list
                    )
                else:
                    R = len(self.laplacian_match_links)
                    n_tail = len(source_vertices) - R
                    neighbor_vertex_weights = np.concatenate(
                        [
                            self._hand_keypoint_laplacian_mult,
                            np.ones(n_tail, dtype=float),
                        ]
                    )
                    target_laplacian = calculate_laplacian_coordinates(
                        source_vertices,
                        adj_list,
                        neighbor_vertex_weights=neighbor_vertex_weights,
                    )

                # Run optimization
                if original:
                    w_nominal_tracking = self.w_nominal_tracking_init
                else:
                    w_nominal_tracking = self.w_nominal_tracking_init * np.exp(
                        -i / self.nominal_tracking_tau
                    )

                q_t_last = last_success_q
                q_a_nominal = (
                    q_nominal_list[i, self.q_a_indices]
                    if q_nominal_list is not None
                    else None
                )

                self.last_solver_success = None
                self.last_objective = np.nan
                warm_start_frame = last_success_frame
                try:
                    q_candidate, cost = self.iterate(
                        q_locked=q_locked_list[i],
                        q_n=np.array(last_success_q, copy=True),
                        q_t_last=q_t_last,
                        target_laplacian=target_laplacian,
                        adj_list=adj_list,
                        obj_pts_local=object_points_local,
                        obj_pts_local_2=object_points_local_2,  # for articulated object
                        object_joint=object_joint,  # for articulated object
                        w_nominal_tracking=w_nominal_tracking,
                        q_a_nominal=q_a_nominal,
                        init_t=(last_success_frame is None),
                        n_iter=50 if last_success_frame is None else 10,
                    )
                except RuntimeError as exc:
                    self.last_solver_success = False
                    self.last_objective = np.nan
                    self.failure_indices.append(i)
                    failed_q = np.full(self.nq, np.nan, dtype=float)
                    retargeted_motions.append(failed_q)
                    diagnostic = {
                        "frame_index": i,
                        "solver_success": False,
                        "objective_value": np.nan,
                        "joint_limit_violation": False,
                        "max_joint_limit_violation": np.nan,
                        "keypoint_shape": (len(self.laplacian_match_links), 3),
                        "keypoints_finite": False,
                        "warm_start_frame": warm_start_frame,
                        "error": str(exc),
                    }
                    self.frame_diagnostics.append(diagnostic)
                    print(
                        f"\n[frame {i} FAILURE] {exc} "
                        f"(next frame warm-starts from {warm_start_frame})"
                    )
                    pbar.set_postfix(status="FAILED")
                    continue

                q = self._apply_mimic_positions(q_candidate)
                last_success_q = np.array(q, copy=True)
                last_success_frame = i
                diagnostic = self.get_retargeting_diagnostics(q, cost)
                diagnostic.update(
                    {
                        "frame_index": i,
                        "warm_start_frame": warm_start_frame,
                        "error": "",
                    }
                )
                self.frame_diagnostics.append(diagnostic)
                if self.debug:
                    robot_link_positions = self._get_robot_link_positions(
                        q, self.laplacian_match_links
                    )
                    self.draw_keypoints(
                        robot_link_positions, name="robot", rgba=(0, 1, 0, 1)
                    )
                retargeted_motions.append(q)
                if self.visualize:
                    self.draw_q(q)
                pbar.set_postfix(cost=cost)

        return (
            np.array(retargeted_motions),
            obj_pts_demo_list,
            obj_pts_list,
            tetrahedra,
            obj_pts_demo_2_list,
            obj_pts_list_2,
        )

    def solve_single_iteration(
        self,
        q_locked: np.ndarray,
        q_a_n_last: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        obj_pts_local_2: np.ndarray | None = None,  # for articulated object
        object_joint: np.ndarray | None = None,  # for articulated object
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        verbose=False,
        init_t=False,
    ):
        """The main function to solve a single iteration of the DiffIK problem.
        Args:
            q_locked: the locked robot and object configuration.
            q_a_n_last: the last optimized robot configuration at current time step.
            q_t_last: the robot and object configuration at the last time step.
            init_t: the current time step is the first time step.
        """
        assert len(q_a_n_last) == self.nq_a

        prog = MathematicalProgram()
        dqa = prog.NewContinuousVariables(len(self.q_a_indices), "dqa")

        # Lock the object pose and unoptimized robot configuration as indicated by self.q_a_indices
        q = np.copy(q_locked)
        q[self.q_a_indices] = q_a_n_last
        self._apply_mimic_positions(q)

        # Minimize laplacian deformation in the object frame
        J_OC_dict, p_OC_dict, _ = self._calc_manipulator_jacobians(
            q,
            links=self.laplacian_match_links,
            obj_frame=self.object_name != "ground"
            and self.object_name != "multi_boxes",
        )
        robot_link_keys = list(self.laplacian_match_links.keys())
        num_robot_links = len(robot_link_keys)
        num_obj_pts = len(obj_pts_local)
        if obj_pts_local_2 is not None:
            num_obj_pts += len(obj_pts_local_2)
        num_vertices = num_robot_links + num_obj_pts
        J_V = np.zeros((3 * num_vertices, self.nq_a))
        for i, key in enumerate(robot_link_keys):
            J_V[3 * i : 3 * (i + 1), :] = J_OC_dict[key]

        robot_pts_local = np.array([p_OC_dict[key] for key in robot_link_keys])
        all_pts = [robot_pts_local, obj_pts_local]
        if obj_pts_local_2 is not None:
            obj_pts_local_2_rotated = rotate_points_along_axis(
                obj_pts_local_2, [0, 0, -1], object_joint
            )
            all_pts.append(obj_pts_local_2_rotated)
        vertices = np.vstack(all_pts)
        if self._hand_keypoint_laplacian_mult is None:
            neighbor_vertex_weights = None
        else:
            neighbor_vertex_weights = np.concatenate(
                [
                    self._hand_keypoint_laplacian_mult,
                    np.ones(num_obj_pts, dtype=float),
                ]
            )
        laplacian_matrix = calculate_laplacian_matrix(
            vertices, adj_list, neighbor_vertex_weights=neighbor_vertex_weights
        )

        laplacian_nominal = laplacian_matrix @ vertices
        J_L = np.kron(laplacian_matrix, np.eye(3)) @ J_V
        laplacian_nominal_vec = laplacian_nominal.flatten()
        target_laplacian_vec = target_laplacian.flatten()
        if self._hand_keypoint_laplacian_mult is None:
            hand_mult = np.ones(num_robot_links, dtype=float)
        else:
            hand_mult = self._hand_keypoint_laplacian_mult
        w_per_vertex = np.concatenate(
            [
                self.laplacian_weights * hand_mult,
                self.laplacian_weights * np.ones(num_obj_pts, dtype=float),
            ]
        )
        W = np.kron(np.diag(w_per_vertex), np.eye(3))
        laplacian_var = prog.NewContinuousVariables(3 * num_vertices, "laplacian")
        prog.AddQuadraticErrorCost(W, target_laplacian_vec, laplacian_var)
        prog.AddLinearEqualityConstraint(
            np.hstack([J_L, -np.eye(3 * num_vertices)]),
            -laplacian_nominal_vec,
            np.hstack([dqa, laplacian_var]),
        )

        # Add non-penetration constraints
        Js, phis = self._update_jacobians_and_phis_from_q(q)
        non_penetration_constraints = []

        for key, phi in phis.items():
            # Ignore the penetration between the object and the ground
            if (
                self.object_name in self.inspector.GetName(key[0])
                and "ground" in self.inspector.GetName(key[1])
            ) or (
                "ground" in self.inspector.GetName(key[0])
                and self.object_name in self.inspector.GetName(key[1])
            ):
                continue

            Ja_n = self._project_jacobian_to_independent(Js[key])
            non_penetration_constraints.append(
                prog.AddLinearConstraint(
                    Ja_n, -phi - self.penetration_tolerance, np.inf, dqa
                )
            )

        # Add joint limit constraints
        if self.activate_joint_limits:
            joint_limits_constraints = prog.AddBoundingBoxConstraint(
                self.q_a_lb - q_a_n_last, self.q_a_ub - q_a_n_last, dqa
            )

        # Add step size constraints
        step_size_constraint = prog.AddLorentzConeConstraint(
            np.concatenate((np.array([self.step_size]), dqa))
        )

        if w_nominal_tracking > 0 and q_a_nominal is not None:
            prog.AddQuadraticErrorCost(
                w_nominal_tracking * np.eye(len(self.track_nominal_indices)),
                q_a_nominal[self.track_nominal_indices]
                - q_a_n_last[self.track_nominal_indices],
                dqa[self.track_nominal_indices],
            )

        dqa_smooth = q_t_last[self.q_a_indices] - q_a_n_last
        dqa_smooth_cost = prog.AddQuadraticErrorCost(self.smooth_weight, dqa_smooth, dqa)

        if verbose:
            options = SolverOptions()
            options.SetOption(CommonSolverOption.kPrintToConsole, True)
            prog.SetSolverOptions(options)

        # Solve the program
        result = self.solver.Solve(prog)
        if init_t:
            # If we are in the first time step, remove the step size constraint
            # since the optimal configuration can be far from the initial guess.
            prog.RemoveConstraint(step_size_constraint)
            result = self.solver.Solve(prog)

        self.last_solver_success = bool(result.is_success())
        if not self.last_solver_success:
            self.solver_failure_count += 1

        if not result.is_success():
            tol = 1e-8

            def dump(binding, label):
                x_val = result.GetSolution(binding.variables())
                ok = binding.evaluator().CheckSatisfied(x_val, tol)
                if not ok:
                    print(f"[VIOLATION] {label}: {type(binding.evaluator()).__name__}")
                    # Try to compute a violation magnitude if bounds exist
                    try:
                        y = binding.evaluator().Eval(x_val)
                        lb = binding.evaluator().lower_bound()
                        ub = binding.evaluator().upper_bound()
                        viol = np.maximum(lb - y, 0) + np.maximum(y - ub, 0)
                        print(f"  max_violation = {float(np.max(viol))}")
                    except Exception:
                        # Non box/bound-type constraints (e.g., cones) won’t have lb/ub.
                        # For Lorentz cone, a quick residual is ||y[1:]|| - y[0] via Eval().
                        try:
                            y = binding.evaluator().Eval(x_val)
                            cone_residual = np.linalg.norm(y[1:]) - y[0]
                            print(f"  cone_residual = {float(cone_residual)}")
                        except Exception:
                            pass

            # Your stored bindings:
            for k, b in enumerate(non_penetration_constraints):
                dump(b, f"non_penetration[{k}]")

            if self.activate_joint_limits:
                dump(joint_limits_constraints, "joint_limits")

            dump(step_size_constraint, "step_size")

        if not result.is_success():
            raise RuntimeError("Retargeting solver failed; see constraint diagnostics above.")

        dqa_star = result.GetSolution(dqa)
        cost = result.get_optimal_cost()
        self.last_objective = float(cost)

        # Backtracking line search to reduce/eliminate penetration after applying dqa
        base_penetration, _ = self._compute_max_penetration(q)

        alpha = 1.0
        beta = 0.5
        min_alpha = 1e-3
        max_iters = 0  # 0 to disable line search

        best_alpha = 0.0
        best_penetration = np.inf
        q_star = np.copy(q)

        q_star[self.q_a_indices] = dqa_star + q_a_n_last
        q_star[:4] /= np.linalg.norm(q_star[:4])
        self._apply_mimic_positions(q_star)

        for _ in range(max_iters):
            q_candidate = np.copy(q)
            q_candidate[self.q_a_indices] = q_a_n_last + alpha * dqa_star
            # Normalize quaternion
            q_candidate[:4] /= np.linalg.norm(q_candidate[:4])
            self._apply_mimic_positions(q_candidate)

            penetration, _ = self._compute_max_penetration(q_candidate)
            # Accept immediately if within tolerance
            if penetration <= 0:
                q_star = q_candidate
                best_alpha = alpha
                best_penetration = penetration
                break

            # Track the best (least penetration) candidate
            if penetration < best_penetration:
                best_penetration = penetration
                best_alpha = alpha
                q_star = q_candidate

            alpha *= beta
            if alpha < min_alpha:
                break

        return q_star, cost

    def iterate(
        self,
        q_locked: np.ndarray,
        q_n: np.ndarray,
        q_t_last: np.ndarray,
        target_laplacian: np.ndarray,
        adj_list: list[list[int]],
        obj_pts_local: np.ndarray,
        obj_pts_local_2: np.ndarray | None = None,
        object_joint: np.ndarray | None = None,
        w_nominal_tracking: float = 0.0,
        q_a_nominal: np.ndarray | None = None,
        init_t: bool = False,
        n_iter: int = 10,
    ):
        """Iterate the solver for multiple iterations."""
        last_cost = np.inf
        for i in range(n_iter):
            q_a_n_last = q_n[self.q_a_indices]
            q_n, cost = self.solve_single_iteration(
                q_locked=q_locked,
                q_a_n_last=q_a_n_last,
                q_t_last=q_t_last,
                target_laplacian=target_laplacian,
                adj_list=adj_list,
                obj_pts_local=obj_pts_local,
                obj_pts_local_2=obj_pts_local_2,
                object_joint=object_joint,
                q_a_nominal=q_a_nominal,
                w_nominal_tracking=w_nominal_tracking,
                init_t=init_t,
            )
            if np.isclose(cost, last_cost, atol=1e-10):
                break
            last_cost = cost
        return q_n, cost

    def draw_q(self, q: np.ndarray):
        """Draw a single robot configuration."""
        q = self._apply_mimic_positions(np.array(q, copy=True))
        self.plant.SetPositions(self.plant_context, q)
        self.vis.ForcedPublish(self.vis_context)
        self.proximity_vis.ForcedPublish(self.proximity_vis_context)

    def draw_q_knots(self, q_knots: np.ndarray, dt: float):
        """Draw a robot trajectory over time."""
        self.vis.DeleteRecording()
        self.vis.StartRecording()
        t_knots = np.arange(len(q_knots)) * dt
        for t, q in zip(t_knots, q_knots):
            if not np.isfinite(q).all():
                continue
            self.context.SetTime(t)
            self.draw_q(q)
        self.vis.StopRecording()
        self.vis.PublishRecording()

    def draw_keypoints(self, p, name="keypoint", rgba=(0, 0, 1, 1), size=0.015):
        """Draw keypoints in visualization."""
        if not hasattr(self, "meshcat"):
            return

        sphere = Sphere(size)
        if len(p.shape) == 1:
            if name not in self._meshcat_created_paths:
                self.meshcat.SetObject(path=name, shape=sphere, rgba=Rgba(*rgba))
                self._meshcat_created_paths.add(name)
            self.meshcat.SetTransform(name, RigidTransform(p))
        elif len(p.shape) == 2:
            for i in range(p.shape[0]):
                path = f"{name}/{i}"
                if path not in self._meshcat_created_paths:
                    self.meshcat.SetObject(path=path, shape=sphere, rgba=Rgba(*rgba))
                    self._meshcat_created_paths.add(path)
                self.meshcat.SetTransform(path, RigidTransform(p[i]))

    def remove_all_keypoints(self):
        """Remove all keypoints previously added via draw_keypoints."""
        if not hasattr(self, "meshcat"):
            return
        # Keypoints are created using these common prefixes throughout this class
        keypoint_prefixes = ("keypoint", "human", "object_demo", "object", "robot", "object_demo_2", "object_2")
        existing_paths = getattr(self, "_meshcat_created_paths", set())
        to_remove = [
            path
            for path in existing_paths
            if any(path == prefix or path.startswith(prefix + "/") for prefix in keypoint_prefixes)
        ]
        for path in to_remove:
            self.meshcat.Delete(path)
        if not hasattr(self, "_meshcat_created_paths"):
            self._meshcat_created_paths = set()
        self._meshcat_created_paths -= set(to_remove)

    def _update_jacobians_and_phis_from_q(self, q: np.ndarray):
        """Update Jacobians and signed distances from current configuration."""
        self._validate_positions(q)
        q = self._apply_mimic_positions(np.array(q, copy=True))
        self.plant.SetPositions(self.plant_context, q)
        sdps = self.query_object.ComputeSignedDistancePairwiseClosestPoints(
            self.collision_detection_threshold
        )

        Js = {}
        phis = {}
        for sdp in sdps:
            bodyA_idx = self.plant.GetBodyFromFrameId(
                self.inspector.GetFrameId(sdp.id_A)
            ).index()
            bodyB_idx = self.plant.GetBodyFromFrameId(
                self.inspector.GetFrameId(sdp.id_B)
            ).index()
            J_bodyA = self._calc_contact_jacobia_from_point(bodyA_idx, sdp.p_ACa)
            J_bodyB = self._calc_contact_jacobia_from_point(bodyB_idx, sdp.p_BCb)
            Jc = J_bodyA - J_bodyB
            Js[sdp.id_A, sdp.id_B] = sdp.nhat_BA_W @ Jc
            phis[sdp.id_A, sdp.id_B] = sdp.distance

        return Js, phis

    def _calc_contact_jacobia_from_point(self, body_idx: int, pC_Body: np.ndarray):
        """Calculate contact Jacobian from point."""
        frameB = self.plant.get_body(body_idx).body_frame()
        J_body = self.plant.CalcJacobianTranslationalVelocity(
            self.plant_context,
            JacobianWrtVariable.kQDot,  # for floating base
            frameB,
            pC_Body,
            self.plant.world_frame(),
            self.plant.world_frame(),
        )
        return J_body

    def _compute_max_penetration(self, q: np.ndarray):
        """Compute maximum penetration beyond tolerance at configuration q.

        Returns:
            tuple[float, tuple | None]: (max_penetration, worst_pair_ids)
                max_penetration >= 0 (meters). 0 means within tolerance.
                worst_pair_ids is a tuple of (id_A, id_B) or None if no penetration.
        """
        _, phis = self._update_jacobians_and_phis_from_q(q)
        max_penetration = 0.0
        worst_pair = None
        for key, phi in phis.items():
            # Skip object–ground pairs, consistent with constraint setup
            if (
                self.object_name in self.inspector.GetName(key[0])
                and "ground" in self.inspector.GetName(key[1])
            ) or (
                "ground" in self.inspector.GetName(key[0])
                and self.object_name in self.inspector.GetName(key[1])
            ):
                continue
            violation = -(phi + self.penetration_tolerance)
            if violation > max_penetration:
                max_penetration = violation
                worst_pair = key
        return max_penetration, worst_pair

    def _calc_manipulator_jacobians(
        self, q: np.ndarray, links: dict[str, str], obj_frame: bool = False
    ):
        """Compute semantic-point Jacobians in object or world coordinates.

        Legacy robots use body origins. A semantic keypoint spec changes the
        evaluated point to ``parent_link @ local_xyz`` without changing the
        interaction-mesh optimizer.
        """
        J_XC_dict = {}
        p_XC_dict = {}

        if obj_frame and self.object_body_name is None:
            obj_frame_id = self.plant_ad.GetBodyFrameIdOrThrow(
                self.plant_ad.GetBodyByName("bottom").index()
            )
            obj_geometry_id = self.inspector_ad.GetGeometryIdByName(
                obj_frame_id,
                Role.kProximity,
                self.object_name + "::" + "Mesh_1",
            )
            P_WO = self.query_object_ad.GetPoseInWorld(obj_geometry_id)
        else:
            P_WO = None

        for name, link_name in links.items():
            q_ad = InitializeAutoDiff(q)
            self._apply_mimic_positions(q_ad)
            self.plant_ad.SetPositions(self.plant_context_ad, q_ad)
            if obj_frame and self.object_body_name is not None:
                P_WO = (
                    self.plant_ad.GetBodyByName(self.object_body_name)
                    .body_frame()
                    .CalcPoseInWorld(self.plant_context_ad)
                )
            semantic = self.semantic_keypoints.get(name)
            if semantic is not None:
                link_name = semantic["parent_link"]
                p_BC = np.asarray(semantic["xyz"], dtype=float)
            else:
                p_BC = np.zeros(3)
            P_WB = (
                self.plant_ad.GetBodyByName(link_name)
                .body_frame()
                .CalcPoseInWorld(self.plant_context_ad)
            )
            p_WC = P_WB.multiply(p_BC)

            if obj_frame:
                p_XC = P_WO.inverse().multiply(p_WC)
            else:
                p_XC = p_WC

            J_XC_dict[name] = self._project_jacobian_to_independent(
                ExtractGradient(p_XC)
            )
            p_XC_dict[name] = ExtractValue(p_XC).squeeze()

        return J_XC_dict, p_XC_dict, P_WO

    def _get_p_OG(
        self, smpl_joints_original: np.ndarray, obj_original: np.ndarray, link_name: str
    ):
        """Get SMPL coordinates in object frame."""
        joint_idx = self.demo_joints.index(link_name)
        P_WO = RigidTransform(Quaternion(obj_original[:4]), obj_original[-3:])
        P_WG = RigidTransform(p=smpl_joints_original[joint_idx])
        P_OG = P_WO.inverse().multiply(P_WG)
        return P_OG.translation()

    def _get_robot_link_positions(self, q, links):
        """Get robot semantic-point positions for a mapping or body origins for names."""
        q = self._apply_mimic_positions(np.array(q, copy=True))
        self.plant.SetPositions(self.plant_context, q)

        if hasattr(links, "items"):
            entries = list(links.items())
        else:
            entries = [(None, link_name) for link_name in links]

        robot_link_positions = []
        for key, link_name in entries:
            semantic = self.semantic_keypoints.get(key)
            if semantic is not None:
                link_name = semantic["parent_link"]
                local_xyz = np.asarray(semantic["xyz"], dtype=float)
            else:
                local_xyz = np.zeros(3)
            pose = (
                self.plant.GetBodyByName(link_name)
                .body_frame()
                .CalcPoseInWorld(self.plant_context)
            )
            robot_link_positions.append(pose.multiply(local_xyz))
        return np.array(robot_link_positions)

    def get_retargeting_diagnostics(self, q, objective_value=None) -> dict:
        """Return solver, bounds, and semantic-keypoint validation metrics."""
        keypoints = self._get_robot_link_positions(q, self.laplacian_match_links)
        actuated = self.get_actuated_joint_positions(q)
        lower = self.plant.GetPositionLowerLimits()[self.actuated_position_indices]
        upper = self.plant.GetPositionUpperLimits()[self.actuated_position_indices]
        violation = np.maximum(lower - actuated, 0.0) + np.maximum(
            actuated - upper, 0.0
        )
        return {
            "solver_success": bool(self.last_solver_success),
            "objective_value": float(
                self.last_objective if objective_value is None else objective_value
            ),
            "joint_limit_violation": bool(np.any(violation > 1e-9)),
            "max_joint_limit_violation": float(np.max(violation, initial=0.0)),
            "keypoint_shape": keypoints.shape,
            "keypoints_finite": bool(np.isfinite(keypoints).all()),
        }

    def _validate_positions(self, q: np.ndarray):
        if q.shape[0] != self.nq:
            raise ValueError(f"Expected q of length {self.nq}, got {q.shape[0]}")
        if not np.all(np.isfinite(q)):
            bad = np.where(~np.isfinite(q))[0]
            raise ValueError(f"Non-finite q at indices {bad}: {q[bad]}")
