from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    BaseAgent is the abstract base class for all agents in AgentArk.
    """

    def __init__(self, name="BaseAgent"):
        self.name = name

    def reset(self, obs):
        """
        Reset the agent state at the beginning of an episode.
        Args:
            obs: The initial observation from the environment.
        Returns:
            The initial action or None.
        """
        pass

    @abstractmethod
    def forward(self, obs):
        pass

    def close(self):
        """
        Clean up resources if necessary.
        """
        pass
