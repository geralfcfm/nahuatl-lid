from __future__ import annotations

import torch
import torch.nn as nn


def _conv_stack() -> nn.Sequential:
    """Shared 3-block Conv2d front-end (1->32->64->128, BN+ReLU+MaxPool each).

    Used by both CRNN_LID and CNN_LID so the two only differ in the head.
    """
    return nn.Sequential(
        nn.Conv2d(1, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 3, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
    )


class CRNN_LID(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = _conv_stack()
        self.lstm = nn.LSTM(1024, 128, batch_first=True, bidirectional=True)
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # logits; no sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        b, c, h, w = x.size()
        x = x.permute(0, 3, 1, 2).reshape(b, w, c * h)
        _, (hn, _) = self.lstm(x)
        x = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(x)


class CNN_LID(nn.Module):
    """LSTM-ablation of CRNN_LID: identical conv front-end, but the BiLSTM is
    replaced by global average pooling over (freq, time). Tests whether the
    recurrent layer contributes to discrimination on a given task.
    """

    def __init__(self) -> None:
        super().__init__()
        self.conv = _conv_stack()
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),  # logits; no sigmoid
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.mean(dim=(2, 3))  # global avg pool over (freq, time) -> (B, 128)
        return self.fc(x)
