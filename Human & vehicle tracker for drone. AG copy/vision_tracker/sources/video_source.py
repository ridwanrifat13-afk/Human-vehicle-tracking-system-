from abc import ABC, abstractmethod


class VideoSource(ABC):
    @abstractmethod
    def start(self):
        raise NotImplementedError

    @abstractmethod
    def read_frame(self):
        raise NotImplementedError

    @abstractmethod
    def stop(self):
        raise NotImplementedError
