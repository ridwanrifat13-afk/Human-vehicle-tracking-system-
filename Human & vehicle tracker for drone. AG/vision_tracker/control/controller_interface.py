from abc import ABC, abstractmethod


class ControlInterface(ABC):
    @abstractmethod
    def send_target_position(self, x, y):
        raise NotImplementedError
