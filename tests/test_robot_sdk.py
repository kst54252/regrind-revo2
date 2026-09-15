"""Read-only SDK boundary tests; no physical devices or unicast network probes."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace as NS
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from tools.robot_execution.sdk_readers import (
    RB3Connection, Revo2Connection, RB3DataReader, Revo2RS485Reader,
)
from tools.robot_execution.sdk_probe import load_config, probe_device


class TestSDKReaders(unittest.TestCase):
    def arm_data(self):
        return NS(time=1., jnt_ang=[0,10,20,30,40,50], jnt_ref=[90]*6, jnt_cur=[1]*6,
            robot_state=1, real_vs_simulation_mode=0, init_state_info=6, init_error=0,
            op_stat_collision_occur=0, op_stat_ems_flag=0, op_stat_self_collision=0,
            op_stat_soft_estop_occur=0, op_stat_sos_flag=0)

    def test_arm_uses_data_channel_and_encoder_not_reference_or_tcp(self):
        calls, data = [], self.arm_data()
        sdk = NS(CobotData=lambda address, port: calls.append((address,port)) or NS(
            request_data=lambda timeout: NS(sdata=data)))
        reader = RB3DataReader(RB3Connection('127.0.0.1'), sdk=sdk)
        self.assertFalse(calls)
        reader.connect()
        sample = reader.read()
        self.assertEqual(calls, [('127.0.0.1',5001)])
        self.assertEqual(sample.values['encoder_position_deg'], [0,10,20,30,40,50])
        np.testing.assert_allclose(sample.values['encoder_position_rad_sdk_order'], np.deg2rad(data.jnt_ang))
        self.assertIsNone(sample.values['actual_velocity_rad_s'])
        self.assertNotIn('tcp_pos', sample.values)
        self.assertFalse(sample.provenance['motion_enabled'])
        reader.close()
        self.assertIsNone(reader.client)

    def test_arm_missing_timeout_duplicate_clock_and_nan_rejected(self):
        data = self.arm_data()
        reader = RB3DataReader(RB3Connection('127.0.0.1'), sdk=NS())
        reader.client = NS(request_data=lambda timeout: NS(sdata=data))
        reader.read()
        with self.assertRaisesRegex(ValueError, 'timestamp'):
            reader.read()
        data.time = 2.
        data.jnt_ang[0] = float('nan')
        with self.assertRaisesRegex(ValueError, 'finite'):
            reader.read()
        reader.client = NS(request_data=lambda timeout: None)
        with self.assertRaises(TimeoutError):
            reader.read()

    def test_explicit_unicast_settings_required(self):
        for address in (None, '0.0.0.0', '224.0.0.1', '255.255.255.255', 'controller.local'):
            with self.subTest(address=address), self.assertRaises(ValueError):
                RB3Connection(address)
        with self.assertRaises(ValueError):
            RB3Connection('127.0.0.1', data_port=5000)
        for change in ({'slave_id':0}, {'slave_id':248}, {'baudrate':None}, {'baudrate':123},
                       {'port':None}, {'transport':'canfd'}, {'expected_hand':'left'}, {'timeout_s':float('nan')}):
            values = dict(port='/dev/ttyUSB0', baudrate=460800, slave_id=127)
            values.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                Revo2Connection(**values)

    def hand_sdk(self, side='Right', hw='StarkHardwareType.Revo2Basic', modes=None):
        calls = []
        mode_values = iter(modes or ['Normalized']*10)
        async def info(sid):
            calls.append(('info',sid))
            return NS(hand_type=side, hardware_type=hw, serial_number='SYNTHETIC_TEST_ONLY', firmware_version='test')
        async def mode(sid):
            calls.append(('unit_read',sid))
            return next(mode_values)
        async def status(sid):
            calls.append(('status_read',sid))
            return NS(positions=[100,200,300,400,500,600], speeds=[-10,20,30,40,50,60],
                      currents=[1,2,3,4,5,6], states=[0]*6)
        ctx = NS(get_device_info=info, get_finger_unit_mode=mode, get_motor_status=status)
        async def open_(port, baud):
            calls.append(('open',port,baud))
            return ctx
        async def close_(ctx):
            calls.append(('close',))
        sdk = NS(modbus_open=open_, modbus_close=close_, Baudrate=NS(Baud460800='460800'),
                 HandType=NS(Right='Right'), FingerUnitMode=NS(Normalized='Normalized', Physical='Physical'))
        return sdk, calls

    def hand_reader(self, sdk):
        # /dev/null is a character device; injected fake SDK never opens it.
        return Revo2RS485Reader(Revo2Connection('/dev/null',460800,127), sdk=sdk)

    def test_hand_connect_reads_only_and_does_not_assume_radians(self):
        sdk, calls = self.hand_sdk()
        reader = self.hand_reader(sdk)
        self.assertFalse(calls)
        async def run():
            await reader.connect()
            sample = await reader.read()
            await reader.close()
            return sample
        row = asyncio.run(run())
        self.assertEqual([c[0] for c in calls], ['open','info','unit_read','status_read','unit_read','close'])
        self.assertEqual(row.channels[:2], ('Thumb','ThumbAux'))
        self.assertIsNone(row.values['model_position_rad'])
        self.assertIsNone(row.device_time_s)
        self.assertEqual(row.values['positions_sdk_raw'][0], 100)

    def test_wrong_hand_or_model_closes_connection(self):
        for side, hw in [('Left','StarkHardwareType.Revo2Basic'),('Right','StarkHardwareType.Revo1Basic')]:
            sdk, calls = self.hand_sdk(side, hw)
            reader = self.hand_reader(sdk)
            with self.assertRaises(ValueError):
                asyncio.run(reader.connect())
            self.assertEqual(calls[-1], ('close',))
            self.assertIsNone(reader.client)

    def test_mode_change_rejects_snapshot_without_changing_device(self):
        sdk, calls = self.hand_sdk(modes=['Normalized','Physical'])
        reader = self.hand_reader(sdk)
        async def run():
            await reader.connect()
            try:
                await reader.read()
            finally:
                await reader.close()
        with self.assertRaisesRegex(ValueError, 'mode changed'):
            asyncio.run(run())
        self.assertFalse(any(c[0].startswith('set') for c in calls))

    def test_serial_path_missing_before_open(self):
        sdk, calls = self.hand_sdk()
        reader = Revo2RS485Reader(Revo2Connection('/dev/regrind_nonexistent_test',460800,127), sdk=sdk)
        with self.assertRaises(FileNotFoundError):
            asyncio.run(reader.connect())
        self.assertFalse(calls)

    def test_worker_config_validation_no_fallback_address(self):
        path = Path(__file__).resolve().parents[1]/'config/robot_execution/hardware.example.json'
        for device in ('rb3','revo2'):
            with self.assertRaises(ValueError):
                probe_device(device,path,1,runner=lambda *a,**kw: self.fail('worker spawned with missing address'))

    def test_native_connect_bounded_by_worker_deadline(self):
        # Mock subprocess only; no child or real endpoint here.
        config = dict(mode='read_only', rb3=dict(address='127.0.0.1',data_port=5001,timeout_s=.1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(config))
            def timeout(*args, **kwargs):
                self.assertLess(kwargs['timeout'], 5)
                raise subprocess.TimeoutExpired(args[0],kwargs['timeout'])
            with self.assertRaisesRegex(TimeoutError, 'process terminated'):
                probe_device('rb3',path,1,runner=timeout)

    def test_configuration_cannot_enable_motion(self):
        config = dict(mode='motion',rb3=dict(address='127.0.0.1'))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'config.json'
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError,'read_only'):
                load_config(path,'rb3')


if __name__ == '__main__':
    unittest.main()
