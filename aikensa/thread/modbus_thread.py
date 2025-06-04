# modbus_server_thread.py

import asyncio
import logging
from PyQt5.QtCore import QThread
from pymodbus.server import StartAsyncTcpServer
from pymodbus.device import ModbusDeviceIdentification
from pymodbus.datastore import (
    ModbusSequentialDataBlock,
    ModbusSlaveContext,
    ModbusServerContext,
)

# Optional: tune logging level if you want server‐side debug/info prints
_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)


class ModbusServerThread(QThread):
    """
    Run a TCP‐only asyncio Modbus server inside a QThread.
    - host: the interface to bind (e.g. "0.0.0.0" or "192.168.1.100")
    - port: the TCP port (default 502)
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 502, parent=None):
        super().__init__(parent)
        self.host = host
        self.port = port
        self.loop = None
        self.context = None

    def setup_context(self):
        """
        Prepare a single‐slave ModbusServerContext with:
            - 100 discrete inputs (DI)
            - 100 coils (CO)
            - 100 holding registers (HR)
            - 100 input registers (IR)
        You can tweak the size and initial values as needed.
        """
        store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * 100),
            co=ModbusSequentialDataBlock(0, [0] * 100),
            hr=ModbusSequentialDataBlock(0, [0] * 100),
            ir=ModbusSequentialDataBlock(0, [0] * 100),
        )
        # single=True means all requests map to this one slave context
        self.context = ModbusServerContext(slaves=store, single=True)

    def setup_identity(self):
        """
        Return a ModbusDeviceIdentification object.
        You can adjust any fields you like.
        """
        identity = ModbusDeviceIdentification()
        identity.VendorName = "PyQtApp"
        identity.ProductCode = "PM"
        identity.VendorUrl = "https://example.com"
        identity.ProductName = "Modbus Server"
        identity.ModelName = "ModbusTCP"
        identity.MajorMinorRevision = "1.0"
        return identity

    def run(self):
        """
        This is called when .start() is invoked on the thread.
        We create a fresh asyncio loop, set up context/identity, start the TCP server,
        and then call loop.run_forever().
        """
        # 1) Create a brand‐new event loop for this thread
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # 2) Build datastore context
        self.setup_context()

        # 3) Build device identity
        identity = self.setup_identity()

        # 4) Start the async TCP server; once that completes, loop.run_forever() will keep it alive.
        coro = StartAsyncTcpServer(
            context=self.context,
            identity=identity,
            address=(self.host, self.port),
        )
        # run_until_complete ensures StartAsyncTcpServer binds/listens before we go into run_forever()
        self.loop.run_until_complete(coro)

        # 5) Hand control over to the loop so the server can serve requests
        self.loop.run_forever()

    def stop(self):
        """
        Call this from the GUI thread to stop the server cleanly.
        It schedules loop.stop() on the event loop, which tears down the TCP listener.
        """
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
