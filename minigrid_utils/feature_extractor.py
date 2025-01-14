# From https://minigrid.farama.org/content/training/
import gymnasium as gym
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = torch.nn.Sequential(
            torch.nn.Conv2d(n_input_channels, 16, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(16, 32, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, (2, 2)),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = torch.nn.Sequential(torch.nn.Linear(n_flatten, features_dim), torch.nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))