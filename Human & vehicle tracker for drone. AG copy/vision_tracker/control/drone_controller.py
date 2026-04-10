from .controller_interface import ControlInterface


class DroneController(ControlInterface):
    def __init__(self, connection=None):
        self.connection = connection

    def send_target_position(self, x, y):
        if self.connection is None:
            raise RuntimeError("DroneController requires a connection to send commands.")
        # Todo: implement real drone command translation.
        self.connection.send({"target_x": x, "target_y": y})
