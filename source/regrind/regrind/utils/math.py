import torch
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul


@torch.jit.script
def quat_from_euler_xyz_intrinsic(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert intrinsic XYZ (Tait–Bryan) Euler angles in radians to unit quaternions ``(w, x, y, z)``.

    Matches :func:`scipy.spatial.transform.Rotation.from_euler` with sequence ``"XYZ"`` (uppercase):
    SciPy uses **uppercase** letters for intrinsic rotations and **lowercase** for extrinsic [1]_.

    Body-fixed order: rotate about X, then Y', then Z'' (same rotation as SciPy ``from_euler("XYZ", ...)``).

    Args:
        roll: Rotation around x-axis (in radians). Shape is (N,).
        pitch: Rotation around y-axis (in radians). Shape is (N,).
        yaw: Rotation around z-axis (in radians). Shape is (N,).

    Returns:
        The quaternion in (w, x, y, z). Shape is (N, 4).

    References:
        .. [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.from_euler.html
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    # Closed form for q_x * q_y * q_z (Hamilton, wxyz); matches scipy Rotation.from_euler("XYZ", ...).
    qw = cr * cp * cy - sr * sp * sy
    qx = sr * cp * cy + cr * sp * sy
    qy = cr * sp * cy - sr * cp * sy
    qz = cr * cp * sy + sr * sp * cy

    return torch.stack([qw, qx, qy, qz], dim=-1)


def euler_xyz_intrinsic_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`quat_from_euler_xyz_intrinsic` (intrinsic XYZ Tait–Bryan angles).

    Matches :meth:`scipy.spatial.transform.Rotation.as_euler` with ``seq="XYZ"`` (intrinsic).

    Args:
        q: Quaternion ``(w, x, y, z)``, shape ``(..., 4)``.

    Returns:
        ``(roll, pitch, yaw)`` in radians, shape ``(..., 3)``, matching the argument order of
        :func:`quat_from_euler_xyz_intrinsic`.
    """
    # SciPy quaternion layout is (x, y, z, w); Isaac uses (w, x, y, z). Port of scipy's
    # `as_euler` for intrinsic "XYZ" (see scipy..._rotation_xp.as_euler).
    x, y, z, w = q[..., 1], q[..., 2], q[..., 3], q[..., 0]
    sign = -1.0
    a = w - y
    b = z + sign * x
    c = y + w
    d = sign * x - z

    half_sum = torch.atan2(b, a)
    half_diff = torch.atan2(d, c)
    angles = torch.zeros(q.shape[:-1] + (3,), device=q.device, dtype=q.dtype)
    angles[..., 1] = 2.0 * torch.atan2(torch.hypot(c, d), torch.hypot(a, b))

    angle_first = 2
    angle_third = 0
    eps = 1e-7
    case1 = angles[..., 1].abs() <= eps
    case2 = (angles[..., 1] - torch.pi).abs() <= eps
    case0 = ~(case1 | case2)

    a0 = torch.where(case1, 2.0 * half_sum, 2.0 * half_diff)
    angles[..., 0] = a0

    a1 = torch.where(case0, half_sum - half_diff, angles[..., angle_first])
    angles[..., angle_first] = a1

    a3 = torch.where(case0, half_sum + half_diff, angles[..., angle_third])
    a3 = a3 * sign
    angles[..., 1] = angles[..., 1] - (torch.pi / 2.0)
    angles[..., angle_third] = a3

    angles = (angles + torch.pi) % (2.0 * torch.pi) - torch.pi
    return angles


@torch.jit.script
def rotation_distance(object_rot, target_rot):
    # Orientation alignment
    quat_diff = quat_mul(object_rot, quat_conjugate(target_rot))
    return 2.0 * torch.asin(torch.clamp(torch.norm(quat_diff[:, 1:4], p=2, dim=-1), max=1.0))  # changed quat convention


@torch.jit.script
def angvel_from_quat_diff(q1: torch.Tensor, q2: torch.Tensor, dt: float) -> torch.Tensor:
    quat_diff = quat_mul(q2, quat_conjugate(q1))
    return axis_angle_from_quat(quat_diff) / dt


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to matrix using Gram-Schmidt."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = torch.nn.functional.normalize(a1, dim=-1, eps=1e-8)
    a2_proj = (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = torch.nn.functional.normalize(a2 - a2_proj, dim=-1, eps=1e-8)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-1)


def matrix_to_rot6d(mat: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrix to flattened first-two-columns rot6d."""
    return mat[..., :2].reshape(mat.shape[0], -1)


def sample_geometric(
    p: float,
    size: int | tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """
    Sample from a geometric distribution.
    """
    dist = torch.distributions.Geometric(
        probs=torch.tensor(p, device=device)
    )
    return dist.sample(size)


def sample_truncated_geometric(
    p: float | torch.Tensor,
    L: int,
    size: int | tuple[int, ...],
    device: str,
) -> torch.Tensor:
    """
    Sample from a geometric distribution truncated to [1, L].
    """
    if not isinstance(p, torch.Tensor):
        p = torch.tensor(p, dtype=torch.float, device=device)

    u = torch.rand(size, device=device)

    # Inverse CDF of truncated geometric
    # F(k) = (1 - (1-p)^k) / (1 - (1-p)^L)
    # => k = ceil( log(1 - u * (1 - (1-p)^L)) / log(1-p) )
    q = 1 - p
    k = torch.ceil(
        torch.log1p(-u * (1 - q**L)) / torch.log(q)
    )

    return k.long()
