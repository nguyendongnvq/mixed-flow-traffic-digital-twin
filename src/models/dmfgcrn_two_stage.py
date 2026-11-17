from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from src.models.dmfgcrn_multilabel import DMFGCRNMultiLabel


class DMFGCRNTwoStage(DMFGCRNMultiLabel):
    def __init__(
        self,
        *,
        project_root: Path,
        conf_path: Path,
        forecast_checkpoint_path: Path,
        num_nodes: int,
        lag: int,
        num_output_classes: int,
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
        temporal_size = 4 * 2 * int(self.args.rnn_units)
        window_size = 2 * temporal_size
        relative_size = int(self.backbone.input_dim) * 4 * 2
        head_size = window_size + relative_size
        self.detector = nn.Linear(head_size, 1)
        self.router = nn.Linear(head_size, num_output_classes)
        self.to(device)

    def load_finetuned_backbone(self, state_dict: dict[str, torch.Tensor]) -> None:
        backbone_state = {
            key.removeprefix("backbone."): value
            for key, value in state_dict.items()
            if key.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state)

    def window_features(self, source: torch.Tensor) -> torch.Tensor:
        node_embeddings = self._node_embeddings(source)
        source_main = source[..., : self.backbone.input_dim]
        encoded = self.backbone.encoder1(source_main, node_embeddings)
        recent = encoded[:, -min(3, encoded.shape[1]) :, :, :]
        last = encoded[:, -1, :, :]
        previous = encoded[:, -2, :, :] if encoded.shape[1] > 1 else last
        temporal = torch.cat(
            [
                last,
                recent.mean(dim=1),
                recent.max(dim=1).values,
                last - previous,
            ],
            dim=-1,
        )
        return torch.cat(
            [
                temporal.mean(dim=1),
                temporal.max(dim=1).values,
            ],
            dim=-1,
        )

    def relative_features(self, source: torch.Tensor) -> torch.Tensor:
        source_main = source[..., : self.backbone.input_dim]
        history_mean = source_main.mean(dim=1)
        last_deviation = source_main[:, -1] - history_mean
        history_std = source_main.std(dim=1, unbiased=False)
        if source_main.shape[1] > 1:
            time = torch.linspace(
                -1.0,
                1.0,
                source_main.shape[1],
                device=source.device,
                dtype=source.dtype,
            )
            denominator = torch.sum(time.square()).clamp_min(1e-6)
            slope = torch.sum(
                (source_main - history_mean[:, None]) * time[None, :, None, None],
                dim=1,
            ) / denominator
            max_change = torch.diff(source_main, dim=1).abs().max(dim=1).values
        else:
            slope = torch.zeros_like(last_deviation)
            max_change = torch.zeros_like(last_deviation)
        statistics = torch.stack(
            [last_deviation, slope, history_std, max_change],
            dim=-1,
        )
        return torch.cat(
            [
                statistics.mean(dim=1).flatten(start_dim=1),
                statistics.max(dim=1).values.flatten(start_dim=1),
            ],
            dim=-1,
        )

    def classify(self, source: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat(
            [self.window_features(source), self.relative_features(source)],
            dim=-1,
        )
        features = self.classifier_dropout(features)
        return self.detector(features).squeeze(-1), self.router(features)

    def forward(
        self,
        source: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        detector_logit, class_logits = self.classify(source)
        return detector_logit, class_logits, self.backbone(source)
