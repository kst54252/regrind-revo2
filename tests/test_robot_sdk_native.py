"""Opt-in REAL SDK parser/transport tests using localhost / a pseudo-terminal.

Run in tools/robot_execution/.venv. No physical robot, control channel or motor
command. These tests cannot establish firmware compatibility or physical units.
"""
import asyncio
import importlib.util
import os
import select
import socket
import struct
import threading
import unittest
from types import SimpleNamespace as NS

from tools.robot_execution.sdk_readers import RB3Connection, RB3DataReader, Revo2Connection, Revo2RS485Reader


def crc16(data):
    crc = 0xffff
    for byte in data:
        crc ^= byte
        for _ in range(8):
            crc = (crc >> 1) ^ (0xa001 if crc & 1 else 0)
    return struct.pack('<H', crc)


class TestNativeSDK(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec('rbpodo'), 'isolated rbpodo SDK not installed in this interpreter')
    def test_rbpodo_real_sdk_reqdata_only_and_actual_encoder_parser(self):
        import rbpodo
        received = []
        # Packet follows pinned data_type.hpp: header, timer, reference6, encoder6.
        packet = bytearray(512)
        # Official header's size is the whole 512-byte structure, not size-4.
        struct.pack_into('<cHB',packet,0,b'$',len(packet),3)
        struct.pack_into('<f',packet,4,1.25)
        struct.pack_into('<6f',packet,8,*([99]*6))
        struct.pack_into('<6f',packet,32,0,10,20,30,40,50)
        server = socket.socket()
        server.bind(('127.0.0.1',0))
        server.listen(1)
        server.settimeout(3)
        def serve():
            connection, _ = server.accept()
            with connection:
                received.append(connection.recv(128))
                connection.sendall(packet)
        worker = threading.Thread(target=serve,daemon=True)
        worker.start()
        # Test-only redirect to an ephemeral LOOPBACK port; production remains 5001.
        sdk = NS(CobotData=lambda address, port: rbpodo.CobotData(address,server.getsockname()[1]))
        reader = RB3DataReader(RB3Connection('127.0.0.1',timeout_s=1),sdk=sdk)
        try:
            reader.connect()
            row = reader.read()
        finally:
            reader.close()
            worker.join(3)
            server.close()
        self.assertFalse(worker.is_alive())
        self.assertEqual(received, [b'reqdata\n'])
        self.assertEqual(row.values['encoder_position_deg'], [0,10,20,30,40,50])
        self.assertEqual(row.values['reference_position_deg'], [99]*6)

    @unittest.skipUnless(importlib.util.find_spec('bc_stark_sdk'), 'isolated BrainCo SDK not installed in this interpreter')
    def test_stark_real_sdk_modbus_reads_only_on_pty(self):
        from bc_stark_sdk import main_mod as sdk
        master, slave = os.openpty()
        path = os.ttyname(slave)
        stop = threading.Event()
        requests, errors = [], []
        # Synthetic read-only register responder. Unknown read registers return 0.
        # Does NOT emulate device identity/firmware; only unit/status transport.
        registers = {2000+i:100*(i+1) for i in range(6)}
        def serve():
            buffer = b''
            while not stop.is_set():
                if not select.select([master],[],[],.1)[0]:
                    continue
                buffer += os.read(master,4096)
                while len(buffer) >= 8:
                    request, buffer = buffer[:8], buffer[8:]
                    sid, function, address, count = struct.unpack('>BBHH',request[:6])
                    requests.append((sid,function,address,count))
                    if crc16(request[:-2]) != request[-2:] or function not in (3,4) or not 1 <= count <= 125:
                        errors.append(request.hex())
                        stop.set()
                        break
                    values = [registers.get(i,0) for i in range(address,address+count)]
                    body = bytes([sid,function,count*2])+struct.pack('>'+'H'*count,*values)
                    os.write(master,body+crc16(body))
        worker = threading.Thread(target=serve,daemon=True)
        worker.start()
        async def run():
            config = Revo2Connection(path,460800,127,timeout_s=1)
            reader = Revo2RS485Reader(config,sdk=sdk)
            # Explicit synthetic identity: this test does NOT validate connect's
            # device-identity reads (covered by injected SDK tests separately).
            reader.client = await sdk.modbus_open(path,sdk.Baudrate.Baud460800)
            reader.info = dict(serial_number='TEST_PTY',firmware_version='SYNTHETIC')
            try:
                return await reader.read()
            finally:
                await reader.close()
        try:
            row = asyncio.run(asyncio.wait_for(run(),5))
        finally:
            stop.set()
            worker.join(2)
            os.close(master)
            os.close(slave)
        self.assertFalse(errors,errors)
        self.assertTrue(requests)
        self.assertTrue(all(r[1] in (3,4) for r in requests))
        self.assertIsNone(row.values['model_position_rad'])
        self.assertEqual(len(row.values['positions_sdk_raw']),6)


if __name__ == '__main__':
    unittest.main()
