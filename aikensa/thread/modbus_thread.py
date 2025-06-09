# # modbus_server_thread.py

# import asyncio
# import logging
# from PyQt5.QtCore import QThread, pyqtSignal
# from pymodbus.server import StartAsyncTcpServer
# from pymodbus.device import ModbusDeviceIdentification
# from pymodbus.datastore import (
#     ModbusSequentialDataBlock,
#     ModbusSlaveContext,
#     ModbusServerContext,
# )

# # Optional: tune logging level if you want server‐side debug/info prints
# _logger = logging.getLogger(__name__)
# _logger.setLevel(logging.INFO)


# class ModbusServerThread(QThread):
#     """
#     Run a TCP‐only asyncio Modbus server inside a QThread.
#     - host: the interface to bind (e.g. "0.0.0.0" or "192.168.1.100")
#     - port: the TCP port (default 502)
#     """
#     holdingUpdated = pyqtSignal(dict)

#     def __init__(self, host: str = "0.0.0.0", port: int = 502, parent=None):
#         super().__init__(parent)
#         self.host = host
#         self.port = port
#         self.loop = None
#         self.context = None
        
#         # You can change these to control which registers get polled:
#         self._poll_start = 0    # starting address of HR to poll
#         self._poll_count = 200   # how many registers to read each time
#         self._poll_interval = 0.5  # seconds between polls

#     def setup_context(self):
#         """
#         Prepare a single‐slave ModbusServerContext with:
#             - 1000 discrete inputs (DI)
#             - 1000 coils (CO)
#             - 1000 holding registers (HR)
#             - 1000 input registers (IR)
#         You can tweak the size and initial values as needed.
#         """
#         size = 1000
#         store = ModbusSlaveContext(
#             di=ModbusSequentialDataBlock(0, [0] * size),
#             co=ModbusSequentialDataBlock(0, [0] * size),
#             hr=ModbusSequentialDataBlock(0, [0] * size),
#             ir=ModbusSequentialDataBlock(0, [0] * size),
#         )

#         # single=True means all requests map to this one slave context
#         self.context = ModbusServerContext(slaves=store, single=True)

#     def setup_identity(self):
#         """
#         Return a ModbusDeviceIdentification object.
#         You can adjust any fields you like.
#         """
#         identity = ModbusDeviceIdentification()
#         identity.VendorName = "PyQtApp"
#         identity.ProductCode = "PM"
#         identity.VendorUrl = "https://example.com"
#         identity.ProductName = "Modbus Server"
#         identity.ModelName = "ModbusTCP"
#         identity.MajorMinorRevision = "1.0"
#         return identity

#     async def _poll_holding_registers(self):
#         """
#         An asyncio task that runs in the same event loop as the server.
#         It sleeps `self._poll_interval` seconds, then reads HR[start : start+count].
#         If those values differ from the last read, it emits `holdingUpdated`.
#         """
#         prev_values = None
#         while True:
#             await asyncio.sleep(self._poll_interval)

#             # Make sure context is ready
#             if self.context is None:
#                 continue

#             # Read holding registers via function code 3
#             try:
#                 raw = self.context[0].getValues(
#                     3,                     # 3 = holding register
#                     self._poll_start,      # starting address
#                     self._poll_count       # number of registers
#                 )
#             except Exception as e:
#                 _logger.error(f"Error reading holding registers: {e}")
#                 continue

#             # Build a simple dict: {address: value}
#             new_values = {
#                 self._poll_start + i: raw[i]
#                 for i in range(len(raw))
#             }

#             # Emit only if changed
#             if new_values != prev_values:
#                 prev_values = new_values
#                 # Emit the dict to any connected slots
#                 self.holdingUpdated.emit(new_values)
#                 print("New values are emitted")

#             # Print all values for debugging
#             # _logger.debug(f"Polled HR[{self._poll_start}..{self._poll_start + self._poll_count - 1}] = {raw}")

#     def write_holding_registers(self, start_addr: int, values: list[int]):
#         """
#         Thread‐safe way to write into holding registers from outside this thread.
#         Schedules the setValues call on the server’s asyncio loop.

#         Example: write_holding_registers(112, [v0, v1, ..., v63]) to set HR[112..175].
#         """
#         def _do_write():
#             try:
#                 # 3 = function code for holding registers
#                 self.context[0].setValues(3, start_addr, values)
#                 _logger.info(f"Wrote HR[{start_addr}..{start_addr + len(values)-1}] = {values}")
#             except Exception as err:
#                 _logger.error(f"Failed to write HR[{start_addr}..]: {err}")

#         if self.loop and self.loop.is_running():
#             # Schedule on the event loop thread
#             self.loop.call_soon_threadsafe(_do_write)
#         else:
#             _logger.warning("Cannot write: Modbus server loop is not running")

#     def run(self):
#         """
#         Called when .start() is invoked. Creates an asyncio loop,
#         schedules the Modbus TCP server and the poller as tasks, then run_forever.
#         """
#         # 1) Create a fresh asyncio event loop for this QThread.
#         self.loop = asyncio.new_event_loop()
#         asyncio.set_event_loop(self.loop)

#         # 2) Build datastore context
#         self.setup_context()

#         # 3) Build device identity
#         identity = self.setup_identity()

#         # 4) Schedule the StartAsyncTcpServer coroutine as a task
#         server_coro = StartAsyncTcpServer(
#             context=self.context,
#             identity=identity,
#             address=(self.host, self.port),
#         )
#         self.loop.create_task(server_coro)

#         # 5) Schedule the polling coroutine
#         self.loop.create_task(self._poll_holding_registers())

#         # 6) Run the event loop until stop() is called
#         self.loop.run_forever()

#     def stop(self):
#         """
#         Call this from the GUI thread to stop the server cleanly.
#         """
#         if self.loop and self.loop.is_running():
#             self.loop.call_soon_threadsafe(self.loop.stop)


# modbus_server_thread.py

import asyncio
import logging
from typing import List, Optional
from PyQt5.QtCore import QThread, pyqtSignal
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
    holdingUpdated = pyqtSignal(dict)

    def __init__(self, host: str = "0.0.0.0", port: int = 502, parent=None):
        super().__init__(parent)
        self.host: str = host
        self.port: int = port
        self.loop: Optional[asyncio.AbstractEventLoop] = None
        self.context: Optional[ModbusServerContext] = None
        
        # You can change these to control which registers get polled:
        self._poll_start: int = 0      # starting address of HR to poll
        self._poll_count: int = 200    # how many registers to read each time
        self._poll_interval: float = 0.5  # seconds between polls

    def setup_context(self) -> None:
        """
        Prepare a single‐slave ModbusServerContext with:
            - 1000 discrete inputs (DI)
            - 1000 coils (CO)
            - 1000 holding registers (HR)
            - 1000 input registers (IR)
        You can tweak the size and initial values as needed.
        """
        size = 1000
        store = ModbusSlaveContext(
            di=ModbusSequentialDataBlock(0, [0] * size),
            co=ModbusSequentialDataBlock(0, [0] * size),
            hr=ModbusSequentialDataBlock(0, [0] * size),
            ir=ModbusSequentialDataBlock(0, [0] * size),
        )
        self.context = ModbusServerContext(slaves=store, single=True)

    def setup_identity(self) -> ModbusDeviceIdentification:
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

    async def _poll_holding_registers(self) -> None:
        """
        An asyncio task that runs in the same event loop as the server.
        It sleeps `self._poll_interval` seconds, then reads HR[start : start+count].
        If those values differ from the last read, it emits `holdingUpdated`.
        """
        prev_values = None
        while True:
            await asyncio.sleep(self._poll_interval)
            if self.context is None:
                continue

            try:
                raw = self.context[0].getValues(
                    3,                     # 3 = holding register
                    self._poll_start,      # starting address
                    self._poll_count       # number of registers
                )
            except Exception as e:
                _logger.error(f"Error reading holding registers: {e}")
                continue

            new_values = {
                self._poll_start + i: raw[i]
                for i in range(len(raw))
            }

            if new_values != prev_values:
                prev_values = new_values
                self.holdingUpdated.emit(new_values)
                print("New values are emitted")

    def write_holding_registers(self, start_addr: int, values: List[int]) -> None:
        """
        Thread‐safe way to write into holding registers from outside this thread.
        Schedules the setValues call on the server’s asyncio loop.
        """
        def _do_write() -> None:
            try:
                self.context[0].setValues(3, start_addr, values)
                _logger.info(f"Wrote HR[{start_addr}..{start_addr + len(values)-1}] = {values}")
            except Exception as err:
                _logger.error(f"Failed to write HR[{start_addr}..]: {err}")

        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(_do_write)
        else:
            _logger.warning("Cannot write: Modbus server loop is not running")

    def run(self) -> None:
        """
        Called when .start() is invoked. Creates an asyncio loop,
        schedules the Modbus TCP server and the poller as tasks, then run_forever.
        """
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        self.setup_context()
        identity = self.setup_identity()

        server_coro = StartAsyncTcpServer(
            context=self.context,
            identity=identity,
            address=(self.host, self.port),
        )
        self.loop.create_task(server_coro)
        self.loop.create_task(self._poll_holding_registers())
        self.loop.run_forever()

    def stop(self) -> None:
        """
        Call this from the GUI thread to stop the server cleanly.
        """
        if self.loop and self.loop.is_running():
            self.loop.call_soon_threadsafe(self.loop.stop)
