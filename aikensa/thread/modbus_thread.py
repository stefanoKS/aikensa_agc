# modbus_qthread.py

from PyQt5.QtCore import QThread, pyqtSignal
from pymodbus.server.sync import StartTcpServer
from pymodbus.datastore import ModbusSlaveContext, ModbusServerContext
from pymodbus.datastore import ModbusSequentialDataBlock
from pymodbus.device import ModbusDeviceIdentification

import threading
import time
import logging


class ModbusServerThread(QThread):
    registers_updated = pyqtSignal(dict)  # Emits {address: value}

    def __init__(self, config):
        super().__init__()
        self.config = config
        self._running = True

        self.store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0]*1000),
            co=ModbusSequentialDataBlock(0, [0]*1000),
            hr=ModbusSequentialDataBlock(0, [0]*1000),
            ir=ModbusSequentialDataBlock(0, [0]*1000)
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)

        self.identity = ModbusDeviceIdentification()
        self.identity.VendorName = 'HIROSHIMAKASEI'
        self.identity.ProductCode = 'AGC_Jidou_Line'
        self.identity.ProductName = 'AGC_Jidou_Line Modbus TCP Server'
        self.identity.ModelName = 'Modbus TCP Server'
        self.identity.MajorMinorRevision = '1.0'

        self.monitor_thread = threading.Thread(target=self.monitor_registers, daemon=True)

    def run(self):
        self.monitor_thread.start()
        StartTcpServer(self.context, identity=self.identity,
                       address=(self.config.ip_address, self.config.port))

    def stop(self):
        self._running = False

    def monitor_registers(self):
        while self._running:
            values = self.context[0x00].getValues(3, self.config.register_start, self.config.register_count)
            data = {addr: val for addr, val in enumerate(values, start=self.config.register_start)}
            logging.info(f"Registers: {data}")
            self.registers_updated.emit(data)
            time.sleep(self.config.update_interval)

    def write_register(self, address, value):
        self.context[0x00].setValues(3, address, [value])
