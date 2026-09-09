"""SO-DexARM adapter around LeRobot's Feetech motor bus.

AmazingHand and SO-101 use different Feetech protocols and may share one
serial device.  LeRobot owns its port handler by default, whereas SO-DexARM
needs two protocol handlers to reuse an externally owned port.  Keeping that
small variation here avoids patching ``lerobot.motors``.
"""

from __future__ import annotations

from contextlib import suppress

import serial
from lerobot.motors import Motor, MotorCalibration
from lerobot.motors.feetech import FeetechMotorsBus as LeRobotFeetechMotorsBus


class FeetechMotorsBus(LeRobotFeetechMotorsBus):
    """LeRobot Feetech bus with optional external ``PortHandler`` ownership."""

    def __init__(
        self,
        port: str,
        motors: dict[str, Motor],
        calibration: dict[str, MotorCalibration] | None = None,
        protocol_version: int = 0,
        shared_port_handler=None,
    ) -> None:
        super().__init__(
            port=port,
            motors=motors,
            calibration=calibration,
            protocol_version=protocol_version,
        )
        self._owns_port = shared_port_handler is None
        if shared_port_handler is not None:
            import scservo_sdk as scs

            self.port_handler = shared_port_handler
            self.sync_reader = scs.GroupSyncRead(self.port_handler, self.packet_handler, 0, 0)
            self.sync_writer = scs.GroupSyncWrite(self.port_handler, self.packet_handler, 0, 0)

    def _connect(self, handshake: bool = True) -> None:
        """Initialize this protocol handler without reopening a shared port."""
        try:
            if self._owns_port and not self.port_handler.openPort():
                raise OSError(f"Failed to open port '{self.port}'.")
            if handshake:
                self._handshake()
        except (FileNotFoundError, OSError, serial.SerialException) as exc:
            raise ConnectionError(
                f"\nCould not connect on port '{self.port}'. Make sure you are using the correct port."
                "\nTry running `lerobot-find-port`\n"
            ) from exc

    def write_calibration(
        self, calibration_dict: dict[str, MotorCalibration], cache: bool = True
    ) -> None:
        """Write every motor independently so one flaky finger does not hide the rest."""
        for motor, calibration in calibration_dict.items():
            with suppress(Exception):
                if self.protocol_version == 0:
                    self.write("Homing_Offset", motor, calibration.homing_offset)
                self.write("Min_Position_Limit", motor, calibration.range_min)
                self.write("Max_Position_Limit", motor, calibration.range_max)
        if cache:
            self.calibration = calibration_dict

    def disable_torque(self, motors: str | list[str] | None = None, num_retry: int = 0) -> None:
        """Best-effort torque release for all requested motors."""
        for motor in self._get_motors_list(motors):
            with suppress(Exception):
                self.write("Torque_Enable", motor, 0, num_retry=num_retry)
                self.write("Lock", motor, 0, num_retry=num_retry)
