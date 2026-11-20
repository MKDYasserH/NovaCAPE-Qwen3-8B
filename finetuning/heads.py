"""The state head, the action head, and the learned temperature they share."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from schemas.training import HeadDims, LossSettings


class ProjectionHead(nn.Module):
    """input_dim -> width -> width (LayerNorm) -> projection_dim, with GELU in between."""

    def __init__(self, dims: HeadDims):
        super().__init__()
        # Layer names (and the one-item ModuleLists) match the original train_heads.py checkpoints.
        self.inp = nn.Linear(dims.input_dim, dims.width)
        self.hidden = nn.ModuleList([nn.Linear(dims.width, dims.width)])
        self.norms = nn.ModuleList([nn.LayerNorm(dims.width)])
        self.out = nn.Linear(dims.width, dims.projection_dim)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.inp(x))
        x = self.activation(self.norms[0](self.hidden[0](x)))
        return self.out(x)


class ContrastiveHeads(nn.Module):
    def __init__(self, dims: HeadDims, loss: LossSettings):
        super().__init__()
        self.state_head = ProjectionHead(dims)
        self.action_head = ProjectionHead(dims)
        # Stored as a log so exp() keeps the scale positive. scale = 1 / temperature.
        self.log_logit_scale = nn.Parameter(torch.tensor(math.log(1 / loss.initial_temperature)))
        self.max_logit_scale = loss.max_logit_scale

    def logit_scale(self) -> torch.Tensor:
        return self.log_logit_scale.exp().clamp(max=self.max_logit_scale)

    def embed_states(self, states: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.state_head(states), dim=-1)

    def embed_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.action_head(actions), dim=-1)

    def checkpoint_weights(self) -> dict:
        # "logit_scale" holds the log, as in CLIP and the original checkpoints.
        return {
            "state_head": self.state_head.state_dict(),
            "action_head": self.action_head.state_dict(),
            "logit_scale": self.log_logit_scale.detach().cpu(),
        }

    def load_checkpoint_weights(self, checkpoint: dict) -> None:
        self.state_head.load_state_dict(checkpoint["state_head"])
        self.action_head.load_state_dict(checkpoint["action_head"])
        self.log_logit_scale.data.copy_(torch.as_tensor(checkpoint["logit_scale"]).float())
