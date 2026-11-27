"""Graph and dense actor-critic policies for fleet repositioning."""

from collections.abc import Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical
from torch_geometric.nn import SAGEConv

from .city import City, GridCity, Move
from .env import ZONE_FEATURES

PolicyObservation = Mapping[str, Tensor | np.ndarray]
_CLOCK_FEATURES = 2
_ZONE_INPUT_SIZE = len(ZONE_FEATURES) + _CLOCK_FEATURES


class FleetPolicy(nn.Module):
    """Share action scoring, masking, and value prediction across encoders."""

    def __init__(self, city: City, hidden_size: int) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("Hidden size must be positive")

        move_zones = []
        for zone in range(city.zone_count):
            destinations = [city.destination(zone, move) for move in Move]
            move_zones.append([
                zone if destination is None else destination for destination in destinations
            ])
        self.register_buffer("_move_zones", torch.tensor(move_zones, dtype=torch.long), persistent=False)
        self.register_buffer("_is_stay", torch.arange(len(Move)) == Move.STAY, persistent=False)

        self.actor = nn.Sequential(
            nn.Linear(2 * hidden_size + 1, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )
        self.critic = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, 1)
        )

    def forward(self, observation: PolicyObservation) -> tuple[Tensor, Tensor]:
        """Return masked vehicle logits and one value per observation.

        Accept a single environment observation or a batch from the same city
        with the same fleet size. The last logits dimension follows ``Move``.
        """
        device = self._move_zones.device
        dtype = self.actor[0].weight.dtype
        features = torch.as_tensor(observation["zone_features"], device=device, dtype=dtype).log1p()
        clock = torch.as_tensor(observation["clock"], device=device, dtype=dtype).log1p()
        vehicle_zones = torch.as_tensor(observation["vehicle_zones"], device=device, dtype=torch.long)
        mask = torch.as_tensor(observation["action_mask"], device=device, dtype=torch.bool)
        single = features.ndim == 2
        if single:
            features = features.unsqueeze(0)
            clock = clock.unsqueeze(0)
            vehicle_zones = vehicle_zones.unsqueeze(0)
            mask = mask.unsqueeze(0)

        embeddings = self._encode_zones(features, clock)
        value = self.critic(embeddings.mean(dim=1)).squeeze(-1)

        # Busy vehicles use zone 0 as a safe gather index. The mask forces stay.
        origins = vehicle_zones.clamp_min(0)
        destinations = self._move_zones[origins]
        batch_rows = torch.arange(embeddings.shape[0], device=device)[:, None, None]
        destination_embeddings = embeddings[batch_rows, destinations]
        origin_embeddings = embeddings[batch_rows, origins.unsqueeze(-1)]
        origin_embeddings = origin_embeddings.expand_as(destination_embeddings)
        is_stay = self._is_stay.view(1, 1, -1, 1).expand(*destinations.shape, 1).to(dtype=dtype)
        action_features = torch.cat((origin_embeddings, destination_embeddings, is_stay), dim=-1)
        logits = self.actor(action_features).squeeze(-1).masked_fill(~mask, -torch.inf)
        if single:
            return logits.squeeze(0), value.squeeze(0)
        return logits, value

    @torch.no_grad()
    def act(self, observation: PolicyObservation, *, deterministic: bool = False) -> Tensor:
        """Sample legal moves or select the highest-scoring move for each vehicle."""
        logits, _ = self(observation)
        if deterministic or logits.numel() == 0:
            return logits.argmax(dim=-1)
        return Categorical(logits=logits).sample()

    def _encode_zones(self, features: Tensor, clock: Tensor) -> Tensor:
        raise NotImplementedError


class GraphFleetPolicy(FleetPolicy):
    """Encode zone features with two layers of GraphSAGE message passing."""

    def __init__(self, city: City, hidden_size: int = 64) -> None:
        super().__init__(city, hidden_size)
        edge_index = torch.tensor(city.directed_edges(), dtype=torch.long).reshape(-1, 2).T.contiguous()
        self.register_buffer("_edge_index", edge_index, persistent=False)
        self.first_conv = SAGEConv(_ZONE_INPUT_SIZE, hidden_size)
        self.second_conv = SAGEConv(hidden_size, hidden_size)

    def _encode_zones(self, features: Tensor, clock: Tensor) -> Tensor:
        zone_clock = clock.unsqueeze(1).expand(-1, features.shape[1], -1)
        zones = torch.cat((features, zone_clock), dim=-1)
        zones = self.first_conv(zones, self._edge_index).relu()
        return self.second_conv(zones, self._edge_index).relu()


class MLPFleetPolicy(FleetPolicy):
    """Encode the fixed grid with dense layers and shared per-zone processing."""

    def __init__(self, city: GridCity, hidden_size: int = 64) -> None:
        super().__init__(city, hidden_size)
        context_size = max(1, hidden_size // 2)
        self.context_encoder = nn.Linear(
            city.zone_count * len(ZONE_FEATURES) + _CLOCK_FEATURES, context_size
        )
        self.zone_encoder = nn.Linear(_ZONE_INPUT_SIZE + context_size, hidden_size)

    def _encode_zones(self, features: Tensor, clock: Tensor) -> Tensor:
        context = self.context_encoder(torch.cat((features.flatten(1), clock), dim=-1)).relu()
        context = context.unsqueeze(1).expand(-1, features.shape[1], -1)
        zone_clock = clock.unsqueeze(1).expand(-1, features.shape[1], -1)
        return self.zone_encoder(torch.cat((features, zone_clock, context), dim=-1)).relu()
