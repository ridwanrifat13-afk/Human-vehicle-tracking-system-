from .controller_interface import ControlInterface


class NullController(ControlInterface):
    def send_target_position(self, x, y):
        # This controller is a placeholder for future drone commands.
        pass
