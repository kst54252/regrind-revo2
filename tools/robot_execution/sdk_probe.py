"""Check installed official SDKs or explicitly probe a named device, read-only."""
import argparse
import asyncio
from dataclasses import asdict
import glob
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys

from .sdk_readers import SDK_VERSIONS, RB3Connection, Revo2Connection, RB3DataReader, Revo2RS485Reader


def doctor():
    packages = {}
    for distribution, module in [('rbpodo', 'rbpodo'), ('bc-stark-sdk', 'bc_stark_sdk.main_mod')]:
        try:
            version = importlib.metadata.version(distribution)
            sdk = importlib.import_module(module)
            names = ('CobotData',) if distribution == 'rbpodo' else ('modbus_open', 'modbus_close', 'DeviceContext')
            missing = [name for name in names if not hasattr(sdk, name)]
            packages[distribution] = dict(installed=version, expected=SDK_VERSIONS[distribution],
                                          ok=version == SDK_VERSIONS[distribution] and not missing,
                                          missing_api=missing)
        except Exception as exc:
            packages[distribution] = dict(ok=False, error=str(exc))
    return dict(sdk=packages, serial_devices=sorted(set(
        glob.glob('/dev/serial/by-id/*')+glob.glob('/dev/ttyUSB*')+glob.glob('/dev/ttyACM*'))),
        hardware_connected=False, motion_enabled=False,
        message='Local enumeration/import only; no port scan, serial open or robot command')


def load_config(path, device):
    config = json.loads(Path(path).read_text())
    if config.get('mode') != 'read_only':
        raise ValueError('Only mode=read_only is supported; no hardware motion switch')
    cls = RB3Connection if device == 'rb3' else Revo2Connection
    return cls(**config[device])


def run_worker(device, config, samples):
    rows = []
    if device == 'rb3':
        reader = RB3DataReader(config)
        try:
            reader.connect()
            for _ in range(samples):
                rows.append(asdict(reader.read()))
                if len(rows) < samples:
                    import time
                    time.sleep(.1)
        finally:
            reader.close()
    else:
        async def read_hand():
            reader = Revo2RS485Reader(config)
            try:
                await reader.connect()
                for _ in range(samples):
                    rows.append(asdict(await reader.read()))
                    if len(rows) < samples:
                        await asyncio.sleep(.1)
            finally:
                await reader.close()
        asyncio.run(read_hand())
    return dict(device=device, connected=True, motion_enabled=False, samples=rows)


def probe_device(device, config_path, samples, *, runner=subprocess.run):
    # Validate before creating a child or importing/opening device SDKs.
    config = load_config(config_path, device)
    # Whole-process wall timeout also covers native blocking connect/destructors.
    # This is read-only process cleanup, NOT a physical emergency stop.
    budget = (4*samples + 4)*config.timeout_s + samples*.1 + 3.
    command = [sys.executable, '-m', 'tools.robot_execution.sdk_probe',
               '--worker', device, '--config', str(Path(config_path).resolve()), '--samples', str(samples)]
    try:
        result = runner(command, capture_output=True, text=True, timeout=budget)
    except subprocess.TimeoutExpired as exc:
        raise TimeoutError(f'{device}: read-only SDK worker exceeded {budget:.1f}s; connection process terminated') from exc
    marker = 'REGRIND_SDK_RESULT='
    encoded = next((line[len(marker):] for line in reversed(result.stdout.splitlines()) if line.startswith(marker)), None)
    if encoded is None:
        raise RuntimeError(f'{device} worker failed (exit {result.returncode}): {result.stderr[-2000:]}')
    payload = json.loads(encoded)
    if result.returncode or not payload.get('connected'):
        raise RuntimeError(payload.get('error', f'{device} worker failed'))
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', nargs='?', choices=('doctor', 'probe'), default='doctor')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--device', choices=('all', 'rb3', 'revo2'), default='all')
    parser.add_argument('--samples', type=int, default=2)
    parser.add_argument('--output', type=Path, help='New report path, never overwritten')
    parser.add_argument('--worker', choices=('rb3', 'revo2'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 1 <= args.samples <= 10:
        parser.error('--samples must be 1..10 (bounded read-only check, not a servo loop)')
    if args.worker:
        try:
            report = run_worker(args.worker, load_config(args.config, args.worker), args.samples)
            code = 0
        except Exception as exc:
            report = dict(device=args.worker, connected=False, motion_enabled=False, error=f'{type(exc).__name__}: {exc}')
            code = 2
        print('REGRIND_SDK_RESULT='+json.dumps(report, allow_nan=False))
        return code
    if args.mode == 'probe' and args.config is None:
        parser.error('probe requires --config with actual device connection settings')
    output = None
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        output = args.output.open('x', encoding='utf-8')  # BEFORE device I/O
    try:
        if args.mode == 'doctor':
            report = doctor()
            code = 0 if all(p['ok'] for p in report['sdk'].values()) else 2
        else:
            reports = []
            for device in (('rb3', 'revo2') if args.device == 'all' else (args.device,)):
                try:
                    reports.append(probe_device(device, args.config, args.samples))
                except Exception as exc:
                    reports.append(dict(device=device, connected=False, motion_enabled=False,
                                        error=f'{type(exc).__name__}: {exc}'))
            report = dict(mode='read_only', devices=reports, motion_enabled=False)
            code = 0 if all(r['connected'] for r in reports) else 2
        text = json.dumps(report, indent=2, allow_nan=False)
        print(text)
        if output:
            output.write(text+'\n')
        return code
    finally:
        if output:
            output.close()


if __name__ == '__main__':
    sys.exit(main())
