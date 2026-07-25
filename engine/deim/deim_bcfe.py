"""
DEIM + BCFE wrapper

结构：
    backbone -> BCFE -> encoder -> decoder
"""

import torch.nn as nn
from ..core import register


@register()
class DEIMBCFE(nn.Module):
    __inject__ = ["backbone", "bcfe", "encoder", "decoder"]

    def __init__(
        self,
        backbone: nn.Module,
        bcfe: nn.Module,
        encoder: nn.Module,
        decoder: nn.Module,
    ):
        super().__init__()
        self.backbone = backbone
        self.bcfe = bcfe
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, targets=None):
        x = self.backbone(x)
        x = self.bcfe(x)
        x = self.encoder(x)
        x = self.decoder(x, targets)
        return x

    def deploy(self):
        self.eval()
        for m in self.modules():
            if hasattr(m, "convert_to_deploy"):
                m.convert_to_deploy()
        return self