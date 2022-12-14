from cgitb import text
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from gym import Space
from habitat import Config
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo.policy import Net
from torch import Tensor

from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.models.encoders import resnet_encoders
from vlnce_baselines.models.encoders.instruction_encoder import (
    InstructionEncoder,
)
from vlnce_baselines.models.policy import ILPolicy


@baseline_registry.register_policy
class CMAPolicy(ILPolicy):
    def __init__(
        self,
        observation_space: Space,
        action_space: Space,
        model_config: Config,
    ) -> None:
        super().__init__(
            CMANet(
                observation_space=observation_space,
                model_config=model_config,
                num_actions=action_space.n,
            ),
            action_space.n,
        )

    @classmethod
    def from_config(
        cls, config: Config, observation_space: Space, action_space: Space
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            model_config=config.MODEL,
        )

class AutoFusion(nn.Module):
    def __init__(self, input_features,latent_dim):
        super().__init__()
        self.input_features = input_features
        self.latent_dim=latent_dim
        self.fuse_in = nn.Sequential(
            nn.Linear(input_features, input_features//2),
            nn.Tanh(),
            nn.Linear(input_features//2, latent_dim),
            nn.ReLU()
            )
        self.fuse_out = nn.Sequential(
            nn.Linear(latent_dim, input_features//2),
            nn.ReLU(),
            nn.Linear(input_features//2, input_features)
            )
        self.criterion = nn.MSELoss()

    def forward(self, z):
        compressed_z = self.fuse_in(z)
        loss = self.criterion(self.fuse_out(compressed_z), z)
        output = {
            'z': compressed_z,
            'loss': loss
        }
        return output

class CMANet(Net):
    """An implementation of the cross-modal attention (CMA) network in
    https://arxiv.org/abs/2004.02857
    """

    def __init__(
        self, observation_space: Space, model_config: Config, num_actions: int
    ) -> None:
        super().__init__()
        self.model_config = model_config
        model_config.defrost()
        model_config.INSTRUCTION_ENCODER.final_state_only = False
        model_config.freeze()

        # Init the instruction encoder
        self.instruction_encoder = InstructionEncoder(
            model_config.INSTRUCTION_ENCODER
        )

        # Init the depth encoder
        assert model_config.DEPTH_ENCODER.cnn_type in ["VlnResnetDepthEncoder"]
        self.depth_encoder = getattr(
            resnet_encoders, model_config.DEPTH_ENCODER.cnn_type
        )(
            observation_space,
            output_size=model_config.DEPTH_ENCODER.output_size,
            checkpoint=model_config.DEPTH_ENCODER.ddppo_checkpoint,
            backbone=model_config.DEPTH_ENCODER.backbone,
            trainable=model_config.DEPTH_ENCODER.trainable,
            spatial_output=True,
        )

        # Init the RGB visual encoder
        assert model_config.RGB_ENCODER.cnn_type in [
            "TorchVisionResNet18",
            "TorchVisionResNet50",
        ]
        self.rgb_encoder = getattr(
            resnet_encoders, model_config.RGB_ENCODER.cnn_type
        )(
            model_config.RGB_ENCODER.output_size,
            normalize_visual_inputs=model_config.normalize_rgb,
            trainable=model_config.RGB_ENCODER.trainable,
            spatial_output=True,
        )

        self.prev_action_embedding = nn.Embedding(num_actions + 1, 32)

        hidden_size = model_config.STATE_ENCODER.hidden_size
        self._hidden_size = hidden_size

        self.rgb_linear = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(
                self.rgb_encoder.output_shape[0],
                model_config.RGB_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )
        self.depth_linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                np.prod(self.depth_encoder.output_shape),
                model_config.DEPTH_ENCODER.output_size,
            ),
            nn.ReLU(True),
        )

        # Init the RNN state decoder
        rnn_input_size = model_config.DEPTH_ENCODER.output_size
        rnn_input_size += model_config.RGB_ENCODER.output_size
        rnn_input_size += self.prev_action_embedding.embedding_dim

        self.state_encoder = build_rnn_state_encoder(
            input_size=rnn_input_size,
            hidden_size=model_config.STATE_ENCODER.hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )

        self._output_size = (
            model_config.STATE_ENCODER.hidden_size
            + model_config.RGB_ENCODER.output_size
            + model_config.DEPTH_ENCODER.output_size
            + self.instruction_encoder.output_size
        )

        self.rgb_kv = nn.Conv1d(
            self.rgb_encoder.output_shape[0],
            hidden_size // 2 + model_config.RGB_ENCODER.output_size,
            1,
        )

        self.depth_kv = nn.Conv1d(
            self.depth_encoder.output_shape[0],
            hidden_size // 2 + model_config.DEPTH_ENCODER.output_size,
            1,
        )

        

        self.state_q = nn.Linear(hidden_size, hidden_size // 2)
        self.text_k = nn.Conv1d(
            self.instruction_encoder.output_size, hidden_size // 2, 1
        )
        self.text_q = nn.Linear(
            self.instruction_encoder.output_size, hidden_size // 2
        )

        self.register_buffer(
            "_scale", torch.tensor(1.0 / ((hidden_size // 2) ** 0.5))
        )

        self.second_state_compress = nn.Sequential(
            nn.Linear(
                self._output_size + self.prev_action_embedding.embedding_dim,
                self._hidden_size,
            ),
            nn.ReLU(True),
        )

        self.auto_fusion=AutoFusion(1184,1184)

        self.second_state_encoder = build_rnn_state_encoder(
            input_size=self._hidden_size,
            hidden_size=self._hidden_size,
            rnn_type=model_config.STATE_ENCODER.rnn_type,
            num_layers=1,
        )
        self._output_size = model_config.STATE_ENCODER.hidden_size

        self.progress_monitor = nn.Linear(self.output_size, 1)

        self._init_layers()

        self.train()

    @property
    def output_size(self) -> int:
        return self._output_size

    @property
    def is_blind(self) -> bool:
        return self.rgb_encoder.is_blind or self.depth_encoder.is_blind

    @property
    def num_recurrent_layers(self) -> int:
        return self.state_encoder.num_recurrent_layers + (
            self.second_state_encoder.num_recurrent_layers
        )

    def _init_layers(self) -> None:
        if self.model_config.PROGRESS_MONITOR.use:
            nn.init.kaiming_normal_(
                self.progress_monitor.weight, nonlinearity="tanh"
            )
            nn.init.constant_(self.progress_monitor.bias, 0)

    def _attn(
        self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None
    ) -> Tensor:
        logits = torch.einsum("nc, nci -> ni", q, k)
        if mask is not None:
            logits = logits - mask.float() * 1e8

        attn = F.softmax(logits * self._scale, dim=1)

        return torch.einsum("ni, nci -> nc", attn, v), attn

    def forward(
        self,
        observations: Dict[str, Tensor],
        rnn_states: Tensor,
        prev_actions: Tensor,
        masks: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        
        # NOTEs:
        #   b = batch size
        # 1) State is comprised by 
        #   instruction   (b, 200)
        #   depth         (b, 224, 224, 3)
        #   rgb           (b, 256, 256, 1)
        #   prev_actions  (b, 1)
        # 2) State is embedded:
        #   instruction   (b, 256, 42)
        #   depth         (b, 192, 16)
        #   rgb           (b, 2112, 16)
        #   prev_actions  (b, 32)
        # 3) Embeddings are projected through a linear layer
        #   rgb           (b, 256)
        #   depth         (b, 128)
        # 4) These embeddings are concatenated with the previous actions 
        #    and passed through the RNN encoder along with the previous hidden 
        #    state, generating the new state (b, 512).   
        # 5) Attention embeddings are computed 
        #    text         (b, 256)
        #    rgb          (b, 256)
        #    depth        (b, 128)
        instruction_embedding = self.instruction_encoder(observations) #; print(observations)
        depth_embedding = self.depth_encoder(observations)
        depth_embedding = torch.flatten(depth_embedding, 2)
        
        rgb_embedding = self.rgb_encoder(observations)
        rgb_embedding = torch.flatten(rgb_embedding, 2)
        
        prev_actions = self.prev_action_embedding(
            ((prev_actions.float() + 1) * masks).long().view(-1)
        )

        if self.model_config.ablate_instruction:
            instruction_embedding = instruction_embedding * 0
        if self.model_config.ablate_depth:
            depth_embedding = depth_embedding * 0
        if self.model_config.ablate_rgb:
            rgb_embedding = rgb_embedding * 0

        rgb_in = self.rgb_linear(rgb_embedding)
        depth_in = self.depth_linear(depth_embedding)

        state_in = torch.cat([rgb_in, depth_in, prev_actions], dim=1)
        rnn_states_out = rnn_states.detach().clone()
        (
            state,
            rnn_states_out[:, 0 : self.state_encoder.num_recurrent_layers],
        ) = self.state_encoder(
            state_in,
            rnn_states[:, 0 : self.state_encoder.num_recurrent_layers],
            masks,
        )
        
        text_state_q = self.state_q(state)
        text_state_k = self.text_k(instruction_embedding)
        text_mask = (instruction_embedding == 0.0).all(dim=1)
        text_embedding, text_attn = self._attn(
            text_state_q, text_state_k, instruction_embedding, text_mask
        )
        
        rgb_k, rgb_v = torch.split(
            self.rgb_kv(rgb_embedding), self._hidden_size // 2, dim=1
        )
        depth_k, depth_v = torch.split(
            self.depth_kv(depth_embedding), self._hidden_size // 2, dim=1
        )

        text_q = self.text_q(text_embedding)
        rgb_embedding, rgb_attn = self._attn(text_q, rgb_k, rgb_v)
        depth_embedding, depth_attn = self._attn(text_q, depth_k, depth_v)

        x = torch.cat(
            [
                state,
                text_embedding,
                rgb_embedding,
                depth_embedding,
                prev_actions,
            ],
            dim=1,
        )
       
        #x is concatenated representation of instructions, depth features and rgb features
        #Pass x to autofusion and get better reconstructed embeddings

        #import ipdb
        #ipdb.set_trace()

        output=self.auto_fusion(x)

        x_fused=output["z"]
        autofusion_loss=output["loss"]

        if AuxLosses.is_active():
            AuxLosses.register_loss(
                "autofusion_loss",
                autofusion_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )

        x = self.second_state_compress(x_fused)
        (x,rnn_states_out[:, self.state_encoder.num_recurrent_layers :],) = self.second_state_encoder(x,rnn_states[:, self.state_encoder.num_recurrent_layers :],masks,)
    
        if self.model_config.PROGRESS_MONITOR.use and AuxLosses.is_active():
            progress_hat = torch.tanh(self.progress_monitor(x))
            progress_loss = F.mse_loss(
                progress_hat.squeeze(1),
                observations["progress"],
                reduction="none",
            )
            AuxLosses.register_loss(
                "progress_monitor",
                progress_loss,
                self.model_config.PROGRESS_MONITOR.alpha,
            )
        
        attention = {
            "text": text_attn.detach().cpu().numpy(), 
            "rgb": rgb_attn.detach().cpu().numpy(), 
            "depth": depth_attn.detach().cpu().numpy()
        }
        return x, rnn_states_out, attention


