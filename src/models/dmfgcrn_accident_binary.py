from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel


class DMFGCRNAccidentBinary(DMFGCRNMultiLabel):
    def __init__(
        self,
        *,
        project_root: Path,
        conf_path: Path,
        forecast_checkpoint_path: Path,
        num_nodes: int,
        lag: int,
        device: torch.device,
    ) -> None:
        super().__init__(
            project_root=project_root,
            conf_path=conf_path,
            forecast_checkpoint_path=forecast_checkpoint_path,
            num_nodes=num_nodes,
            lag=lag,
            num_classes=1,
            device=device,
        )
        hidden_size = 2 * int(self.args.rnn_units)
        self.classifier = nn.Linear(4 * hidden_size, 1)
        self.to(device)

    def load_finetuned_backbone(self, state_dict: dict[str, torch.Tensor]) -> None:
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state)

    def encode_sequence(self, source: torch.Tensor) -> torch.Tensor:
        node_embeddings = self._node_embeddings(source)
        source_main = source[..., : self.backbone.input_dim]
        return self.backbone.encoder1(source_main, node_embeddings)

    def classify(self, source: torch.Tensor) -> torch.Tensor:
        encoded = self.encode_sequence(source)
        recent = encoded[:, -min(3, encoded.shape[1]) :, :, :]
        last = encoded[:, -1, :, :]
        previous = encoded[:, -2, :, :] if encoded.shape[1] > 1 else last
        features = torch.cat(
            [
                last,
                recent.mean(dim=1),
                recent.max(dim=1).values,
                last - previous,
            ],
            dim=-1,
        )
        return self.classifier(self.classifier_dropout(features))
