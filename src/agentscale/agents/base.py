from abc import ABC, abstractmethod


class BaseAgent(ABC):
    @property
    @abstractmethod
    def capabilities(self) -> list:
        pass

    # @abstractmethod
    # def process(self, query: str) -> str:
    #     pass
