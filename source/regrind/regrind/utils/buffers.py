from isaaclab.utils.buffers import DelayBuffer
import torch


class DelayBufferWithFrameStack(DelayBuffer):
    """Delay buffer that returns a stack of delayed frames per batch element.

    If the current timestep is t, the delay (time lag) is k, and the number of stacked
    frames is h+1 (history length h), the buffer returns observations from t-k-h to t-k
    inclusive, ordered oldest-to-newest along a new temporal dimension.
    """

    def __init__(self, batch_size: int, device: str, num_stacked_frames: int, max_time_lag: int = 0):
        """Initialize the stacked delay buffer.

        Args:
            batch_size: Batch size of the incoming data (first dimension).
            device: Torch device string.
            num_stacked_frames: Number of frames to return per call (h+1). Must be >= 1.
            max_time_lag: Maximum supported time lag k (>= 0). The internal buffer length is
                set to num_stacked_frames + max_time_lag to guarantee feasibility.
        """
        if num_stacked_frames < 1:
            raise ValueError(f"'num_stacked_frames' must be >= 1. Received: {num_stacked_frames}")
        if max_time_lag < 0:
            raise ValueError(f"'max_time_lag' must be >= 0. Received: {max_time_lag}")
        # total buffer length needed to support stack and lag
        buffer_length = int(num_stacked_frames) + int(max_time_lag)
        # underlying DelayBuffer expects history_length == buffer_length - 1
        super().__init__(history_length=buffer_length - 1, batch_size=batch_size, device=device)
        self._num_stacked_frames = int(num_stacked_frames)
        self._buffer_length = int(buffer_length)
        self._max_supported_time_lag = int(max_time_lag)

    @property
    def num_stacked_frames(self) -> int:
        """Number of frames returned per compute call (temporal length)."""
        return self._num_stacked_frames

    @property
    def buffer_length(self) -> int:
        """Total internal buffer length (frames) including the current frame."""
        return self._buffer_length

    def set_time_lag(self, time_lag: int | torch.Tensor, batch_ids=None):
        """Sets the time lag and validates feasibility with frame stacking."""
        super().set_time_lag(time_lag=time_lag, batch_ids=batch_ids)
        # After setting, validate that the maximum required lag fits within the history
        max_required_lag = self.max_time_lag + (self._num_stacked_frames - 1)
        # Underlying DelayBuffer history_length == buffer_length - 1
        if max_required_lag > (self._buffer_length - 1):
            raise ValueError(
                f"Requested time_lag (max {self.max_time_lag}) with num_stacked_frames="
                f"{self._num_stacked_frames} requires total lag {max_required_lag}, "
                f"which exceeds the available buffer capacity ({self._buffer_length - 1})."
            )

    def compute(self, data: torch.Tensor, flatten_history_dim: bool = False) -> torch.Tensor:
        """Append input and return a stack of delayed frames based on per-batch time lags.

        Args:
            data: Input tensor of shape (batch_size, ...).
            flatten_history_dim: If True, flattens the history and feature dimensions into a single
                last dimension.

        Returns:
            If flatten_history_dim is False: tensor of shape (batch_size, num_stacked_frames, ...),
            ordered oldest-to-newest along the history dimension (dim=1).
            If True: tensor of shape (batch_size, num_stacked_frames * prod(...)).
        """
        # Append latest data to underlying circular buffer
        self._circular_buffer.append(data)
        # Build per-batch lag indices for the stacked frames:
        # From oldest (time_lag + h) down to newest (time_lag)
        h = self._num_stacked_frames - 1
        # Shape: (h+1,)
        column_offsets = torch.arange(h, -1, -1, device=self.device, dtype=torch.long)
        # Shape: (batch_size, h+1)
        lag_matrix = self._time_lags[:, None].to(dtype=torch.long) + column_offsets[None, :]
        # Gather frames for each offset column-wise and stack along temporal dimension
        frames = []
        for j in range(lag_matrix.shape[1]):
            frames.append(self._circular_buffer[lag_matrix[:, j]])
        stacked = torch.stack(frames, dim=1)
        if flatten_history_dim:
            return stacked.view(stacked.shape[0], -1)
        else:
            return stacked


class ContinuousDelayBuffer(DelayBuffer):

    def __init__(self, history_length: int, batch_size: int, device: str, use_quaternion_interpolation: bool = False):
        """Delay buffer that supports continuous (fractional) time lags via interpolation.

        Args:
            history_length: Number of past discrete steps to keep (max admissible lag).
            batch_size: Batch dimension of incoming data (first dimension).
            device: Torch device string.
            use_quaternion_interpolation: If True, use SLERP for interpolation assuming the last
                dimension of the data represents quaternion(s) (size 4). Falls back to linear
                interpolation when False.
        """
        super().__init__(history_length=history_length, batch_size=batch_size, device=device)
        # store continuous (float) time lags per batch index
        self._cont_time_lags = torch.zeros(batch_size, dtype=torch.float32, device=device)
        # interpolation mode
        self._use_quaternion_interpolation = bool(use_quaternion_interpolation)

    @property
    def continuous_time_lags(self) -> torch.Tensor:
        """Returns the per-batch continuous time lags as a float tensor of shape (batch_size,)."""
        return self._cont_time_lags

    def set_time_lag(self, time_lag: float | torch.Tensor, batch_ids=None):
        """Sets the continuous time lag (can be fractional) per batch element.

        Args:
            time_lag: Float or float tensor with shape (len(batch_ids),) giving the desired lag.
            batch_ids: Batch indices to set. If None, sets for all batch elements.

        Raises:
            TypeError: If provided tensor is not floating point.
            ValueError: If any lag is negative or exceeds history_length.
        """
        if batch_ids is None:
            batch_ids = slice(None)
        if isinstance(time_lag, torch.Tensor):
            if not torch.is_floating_point(time_lag):
                raise TypeError(
                    f"Invalid dtype for time_lag: {time_lag.dtype}. Expected a floating point tensor."
                )
            values = time_lag.to(device=self.device, dtype=torch.float32)
        else:
            # scalar python float or int (allow ints to ease caller use)
            values = torch.tensor(time_lag, device=self.device, dtype=torch.float32).expand(
                self._cont_time_lags[batch_ids].shape
            )

        # Validate bounds against configured history capacity
        min_val = float(torch.min(values).item())
        max_val = float(torch.max(values).item())
        if min_val < 0.0:
            raise ValueError(f"The minimum continuous time lag cannot be negative. Received: {min_val}")
        if max_val > float(self.history_length):
            raise ValueError(
                f"The maximum continuous time lag cannot be larger than the history length "
                f"({self.history_length}). Received: {max_val}"
            )

        self._cont_time_lags[batch_ids] = values

    def compute(self, data: torch.Tensor) -> torch.Tensor:
        """Append input and return data delayed by a continuous lag using interpolation.

        Given a continuous lag d in [0, history_length], the output is:
            - Linear (default): (1 - frac(d)) * x[t - floor(d)] + frac(d) * x[t - ceil(d)]
            - Quaternion (if enabled): SLERP between x[t - floor(d)] and x[t - ceil(d)] along last dim (size 4).

        If older frames than currently available are requested (e.g., right after reset),
        indices are clamped to the oldest available frame by the underlying circular buffer.

        Args:
            data: Input tensor of shape (batch_size, ...). Must be floating point.

        Returns:
            Tensor of shape (batch_size, ...) containing the delayed/interpolated data.
        """
        # Append latest data
        self._circular_buffer.append(data)

        # Prepare floor/ceil integer lags and interpolation weights
        floor_lags = torch.floor(self._cont_time_lags).to(dtype=torch.long)
        ceil_lags = torch.ceil(self._cont_time_lags).to(dtype=torch.long)
        # Safety clamp within configured history length (additional clamping to availability happens in __getitem__)
        floor_lags = torch.clamp(floor_lags, min=0, max=self.history_length)
        ceil_lags = torch.clamp(ceil_lags, min=0, max=self.history_length)

        frac = (self._cont_time_lags - floor_lags.to(dtype=torch.float32)).to(dtype=torch.float32)

        # Gather the two neighboring frames
        x0 = self._circular_buffer[floor_lags]
        x1 = self._circular_buffer[ceil_lags]

        # Ensure we interpolate in a floating type
        if not torch.is_floating_point(x0):
            raise TypeError(
                "ContinuousDelayBuffer requires floating point data for interpolation."
            )
        # Broadcast weight across feature dimensions
        # Shape: (batch_size, 1, 1, ..., 1) to match data dimensions
        expand_shape = [data.shape[0]] + [1] * (data.dim() - 1)
        w = frac.view(*expand_shape).to(dtype=x0.dtype)

        if self._use_quaternion_interpolation:
            # Require quaternion(s) in the last dimension
            if x0.shape[-1] != 4:
                raise ValueError(
                    "Quaternion interpolation requires last tensor dimension to be 4 (w,x,y,z or x,y,z,w). "
                    f"Got shape {tuple(x0.shape)}"
                )
            out = self._slerp_quaternion_batch(x0, x1, w)
        else:
            out = (1.0 - w) * x0 + w * x1
        return out.clone()

    @staticmethod
    def _normalize_quaternion(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        norms = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(eps)
        return q / norms

    @classmethod
    def _slerp_quaternion_batch(cls, q0: torch.Tensor, q1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation for batched quaternions.

        Args:
            q0: Tensor (..., 4)
            q1: Tensor (..., 4)
            t:  Tensor broadcastable to (..., 1), with values in [0, 1]

        Returns:
            Tensor (..., 4) of interpolated unit quaternions.
        """
        # Normalize inputs to be safe
        q0n = cls._normalize_quaternion(q0)
        q1n = cls._normalize_quaternion(q1)
        # Dot product along quaternion dimension
        dot = torch.sum(q0n * q1n, dim=-1, keepdim=True)
        # If dot < 0, negate target to take shortest path
        q1n = torch.where(dot < 0.0, -q1n, q1n)
        dot = torch.abs(dot)

        # If quaternions are very close, fall back to normalized lerp
        close = dot > 0.9995
        # Linear path for close pairs
        lerp = cls._normalize_quaternion((1.0 - t) * q0n + t * q1n)

        # SLERP for general case
        theta0 = torch.acos(torch.clamp(dot, -1.0, 1.0))
        sin_theta0 = torch.sin(theta0)
        theta = theta0 * t
        sin_theta = torch.sin(theta)
        s0 = torch.sin(theta0 - theta) / (sin_theta0 + 1e-8)
        s1 = sin_theta / (sin_theta0 + 1e-8)
        slerp = s0 * q0n + s1 * q1n
        slerp = cls._normalize_quaternion(slerp)

        # Select between lerp and slerp per element
        return torch.where(close, lerp, slerp)
