from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.dmfgcrn_adapter import (
    _build_args_from_conf,
    _load_dmfgcrn_conf,
    _load_dmfgcrn_network,
)


class DMFGCRNMultiLabel(nn.Module):
    def __init__(
        self,
        *,
        project_root: Path,
        conf_path: Path,
        forecast_checkpoint_path: Path,
        num_nodes: int,
        lag: int,
        num_classes: int,
        device: torch.device,
    ) -> None:
        super().__init__()
        conf = _load_dmfgcrn_conf(conf_path)
        args = _build_args_from_conf(conf, num_nodes=num_nodes, lag=lag, device=str(device))
        network_cls = _load_dmfgcrn_network(project_root)
        self.backbone = network_cls(args)
        state_dict = torch.load(
            forecast_checkpoint_path,
            map_location=device,
            weights_only=False,
        )
        self.backbone.load_state_dict(state_dict)
        self.classifier_dropout = nn.Dropout(p=0.2)
        self.classifier = nn.Linear(2 * int(args.rnn_units), num_classes)
        self.args = args
        self.to(device)

    def _node_embeddings(self, source: torch.Tensor) -> list[torch.Tensor]:
        node_embedding = self.backbone.node_embeddings1
        if self.backbone.use_D:
            time_index = (
                source[..., self.backbone.input_dim] * 288
            ).round().long().clamp(0, 287)
            node_embedding = node_embedding * self.backbone.T_i_D_emb[time_index]
        if self.backbone.use_W:
            week_index = (
                source[..., self.backbone.input_dim + 1] * 7
            ).round().long().clamp(0, 6)
            node_embedding = node_embedding * self.backbone.D_i_W_emb[week_index]
        return [node_embedding, self.backbone.node_embeddings1]

    def encode(self, source: torch.Tensor) -> torch.Tensor:
        node_embeddings = self._node_embeddings(source)
        source_main = source[..., : self.backbone.input_dim]
        encoded = self.backbone.encoder1(source_main, node_embeddings)
        return encoded[:, -1, :, :]

    def forward(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        node_logits = self.classify(source)
        forecast = self.backbone(source)
        return node_logits, forecast

    def classify(self, source: torch.Tensor) -> torch.Tensor:
        embedding = self.encode(source)
        return self.classifier(self.classifier_dropout(embedding))

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable

    def enable_encoder_finetuning(self) -> None:
        self.set_backbone_trainable(False)
        for parameter in self.backbone.encoder1.parameters():
            parameter.requires_grad = True
        self.backbone.node_embeddings1.requires_grad = True
        self.backbone.T_i_D_emb.requires_grad = True
        self.backbone.D_i_W_emb.requires_grad = True
