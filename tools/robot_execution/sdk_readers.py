"""Official SDK read-only connections, NOT a commissioned motion backend.

Imports and constructors do not connect. No robot control channel, mode setter,
homing, auto-detection scan, current/torque or motion API is used. See
docs/ROBOT_EXECUTION.md for pinned source/provenance and unsupported quantities.
"""
import asyncio
from dataclasses import dataclass
import importlib.metadata
import ipaddress
from pathlib import Path
import stat
import time

import numpy as np

from .contracts import vector


SDK_VERSIONS = {"rbpodo": "0.16.14", "bc-stark-sdk": "2.0.3"}
RB3_CHANNELS = ("base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3")
REVO2_CHANNELS = ("Thumb", "ThumbAux", "Index", "Middle", "Ring", "Pinky")
# Supported explicit SDK enum spellings, not guessed baudrate enum integers.
BAUDRATES = {19200: "Baud19200", 57600: "Baud57600", 115200: "Baud115200",
             460800: "Baud460800", 1000000: "Baud1Mbps", 2000000: "Baud2Mbps",
             3000000: "Baud3Mbps", 4000000: "Baud4Mbps", 5000000: "Baud5Mbps", 6000000: "Baud6Mbps"}


def _timeout(value):
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value) or not 0 < value <= 10:
        raise ValueError("timeout_s must be finite in (0, 10]")
    return float(value)


def check_version(distribution):
    actual = importlib.metadata.version(distribution)
    if actual != SDK_VERSIONS[distribution]:
        raise RuntimeError(f"{distribution}: expected {SDK_VERSIONS[distribution]}, got {actual}; revalidate before upgrading")
    return actual


@dataclass(frozen=True)
class RB3Connection:
    address: str
    data_port: int = 5001
    timeout_s: float = 1.

    def __post_init__(self):
        if not isinstance(self.address, str):
            raise ValueError("RB3 address is missing; specify the controller IPv4, not an example IP")
        address = ipaddress.IPv4Address(self.address)
        if address.is_unspecified or address.is_multicast or str(address) == "255.255.255.255":
            raise ValueError("RB3 requires a specific unicast IPv4")
        if self.data_port != 5001:
            raise ValueError("Read-only RB3 adapter only permits the official data port 5001, never control port 5000")
        _timeout(self.timeout_s)


@dataclass(frozen=True)
class Revo2Connection:
    port: str
    baudrate: int
    slave_id: int
    transport: str = "rs485"
    expected_hand: str = "right"
    timeout_s: float = 1.

    def __post_init__(self):
        if not isinstance(self.port, str) or not Path(self.port).is_absolute() or not self.port.startswith('/dev/'):
            raise ValueError("Revo2 port missing/invalid; specify an explicit /dev/serial/by-id/... or /dev/ttyUSB...")
        if self.transport != 'rs485':
            raise ValueError('Only user-selected RS-485/Modbus RTU is implemented')
        if self.baudrate not in BAUDRATES:
            raise ValueError('Provide the actual configured Revo2 baudrate; no probing or fallback baudrate')
        if type(self.slave_id) is not int or not 1 <= self.slave_id <= 247:
            raise ValueError('Provide a unicast Modbus slave_id in 1..247; broadcast/automatic selection forbidden')
        if self.expected_hand != 'right':
            raise ValueError('This project uses the right Revo2 hand; left-hand connection is not accepted')
        _timeout(self.timeout_s)


@dataclass(frozen=True)
class Telemetry:
    """Read result retains clock uncertainty; cannot masquerade as RobotState.

    host_received is NOT a synchronized sensor acquisition timestamp. In
    particular, Revo2 get_motor_status provides no device timestamp.
    """
    device: str
    sequence: int
    host_request_started: float
    host_received: float
    device_time_s: float | None
    channels: tuple[str, ...]
    values: dict
    provenance: dict


class RB3DataReader:
    def __init__(self, config: RB3Connection, *, sdk=None, clock=time.monotonic):
        self.config, self.sdk, self.clock = config, sdk, clock
        self.client = None
        self.sequence = 0
        self.last_device_time = None

    def connect(self):
        if self.client is not None:
            raise RuntimeError('RB3 reader already connected')
        if self.sdk is None:
            check_version('rbpodo')
            import rbpodo
            self.sdk = rbpodo
        # CobotData uses reqdata on 5001 only. Cobot (5000) is never constructed.
        # SDK connect() has no exposed timeout: CLI isolates it in a bounded child.
        self.client = self.sdk.CobotData(self.config.address, self.config.data_port)

    def read(self):
        if self.client is None:
            raise RuntimeError('RB3 reader not connected')
        started = self.clock()
        state = self.client.request_data(self.config.timeout_s)
        received = self.clock()
        if state is None:
            raise TimeoutError('RB3 reqdata returned no state before timeout')
        data = state.sdata
        device_time = float(data.time)
        if not np.isfinite(device_time) or device_time < 0:
            raise ValueError('Invalid RB3 device clock')
        if self.last_device_time is not None and device_time <= self.last_device_time:
            raise ValueError('RB3 repeated/backward device timestamp; no freshness restamp')
        measured = vector(data.jnt_ang, 6, 'RB3 measured degrees')
        reference = vector(data.jnt_ref, 6, 'RB3 reference degrees')
        current = vector(data.jnt_cur, 6, 'RB3 measured current A')
        values = dict(encoder_position_deg=measured.tolist(),
            encoder_position_rad_sdk_order=np.deg2rad(measured).tolist(),
            reference_position_deg=reference.tolist(), current_A=current.tolist(),
            actual_velocity_rad_s=None,
            status={name: int(getattr(data, name)) for name in (
                'robot_state', 'real_vs_simulation_mode', 'init_state_info', 'init_error',
                'op_stat_collision_occur', 'op_stat_ems_flag', 'op_stat_self_collision',
                'op_stat_soft_estop_occur', 'op_stat_sos_flag')})
        row = Telemetry('rb3', self.sequence, started, received, device_time, RB3_CHANNELS, values,
            dict(position='SystemState.sdata.jnt_ang: encoder degrees (not jnt_ref)',
                 velocity='UNAVAILABLE on this data channel; not fabricated as zero',
                 current='jnt_cur in A, NOT torque or a saturation measurement',
                 wrist='NOT emitted: SDK tcp_pos warns of reference overwrite; model/base calibration required',
                 time='device timer + host request interval; clocks not synchronized',
                 motion_enabled=False))
        self.last_device_time = device_time
        self.sequence += 1
        return row

    def close(self):
        # The Python binding exposes no close(); dropping the owned C++ object
        # invokes Socket's destructor. CLI process exit also releases the socket.
        self.client = None


class Revo2RS485Reader:
    def __init__(self, config: Revo2Connection, *, sdk=None, clock=time.monotonic):
        self.config, self.sdk, self.clock = config, sdk, clock
        self.client = None
        self.info = None
        self.sequence = 0

    async def _wait(self, awaitable):
        return await asyncio.wait_for(awaitable, timeout=self.config.timeout_s)

    async def connect(self):
        if self.client is not None:
            raise RuntimeError('Revo2 reader already connected')
        if self.sdk is None:
            check_version('bc-stark-sdk')
            from bc_stark_sdk import main_mod
            self.sdk = main_mod
        path = Path(self.config.port)
        if not path.exists() or not stat.S_ISCHR(path.stat().st_mode):
            raise FileNotFoundError(f'Revo2 serial character device not found: {path}')
        baudrate = getattr(self.sdk.Baudrate, BAUDRATES[self.config.baudrate])
        self.client = await self._wait(self.sdk.modbus_open(str(path), baudrate))
        try:
            info = await self._wait(self.client.get_device_info(self.config.slave_id))
            if info.hand_type != self.sdk.HandType.Right:
                raise ValueError('Connected hand is not RIGHT; refusing further reads')
            if not str(info.hardware_type).startswith('StarkHardwareType.Revo2'):
                raise ValueError(f'Expected Revo2 hardware, got {info.hardware_type}')
            if not info.serial_number or not info.firmware_version:
                raise ValueError('Missing device serial/firmware; do not assume default hardware detection is valid')
            self.info = dict(serial_number=info.serial_number, firmware_version=info.firmware_version,
                             hardware_type=str(info.hardware_type), hand_type=str(info.hand_type))
        except BaseException:
            await self.close()
            raise

    async def read(self):
        if self.client is None or self.info is None:
            raise RuntimeError('Revo2 reader not connected/identified')
        sid = self.config.slave_id
        mode_before = await self._wait(self.client.get_finger_unit_mode(sid))
        started = self.clock()
        status = await self._wait(self.client.get_motor_status(sid))
        received = self.clock()
        mode_after = await self._wait(self.client.get_finger_unit_mode(sid))
        if mode_before != mode_after:
            raise ValueError('Revo2 unit mode changed during read; snapshot discarded')
        if mode_before not in (self.sdk.FingerUnitMode.Normalized, self.sdk.FingerUnitMode.Physical):
            raise ValueError('Unknown Revo2 unit mode')
        values = {name+'_sdk_raw': vector(getattr(status, name), 6, 'Revo2 '+name).tolist()
                  for name in ('positions', 'speeds', 'currents')}
        # Installed 2.0.3 returns MotorState enums. Preserve enum codes
        # explicitly, not float(enum).
        values['states_sdk_raw'] = vector([int(s) for s in status.states], 6, 'Revo2 states').astype(int).tolist()
        values.update(unit_mode=str(mode_before), device_info=self.info.copy(), model_position_rad=None)
        row = Telemetry('revo2', self.sequence, started, received, None, REVO2_CHANNELS, values,
            dict(position='SDK 2.0.3 declares unified 0..1000; preserved as raw, no angle calibration assumed',
                 velocity='SDK raw speeds, NOT assumed rad/s',
                 current='SDK raw currents, NOT assumed mA/Nm',
                 model_mapping='UNCALIBRATED: Thumb/ThumbAux must not be blindly mapped to model order',
                 time='host request/response bounds only; sensor acquisition age UNKNOWN',
                 motion_enabled=False))
        self.sequence += 1
        return row

    async def close(self):
        client, self.client, self.info = self.client, None, None
        if client is not None:
            await self._wait(self.sdk.modbus_close(client))
