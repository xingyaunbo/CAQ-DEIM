"""
Boundary-Context Feature Enhancement (BCFE)

用于猪只遮挡/粘连场景：
1. Boundary branch：增强局部边界、轮廓、接触区域；
2. Context branch：增强邻域上下文，辅助区分相邻个体；
3. Gate fusion：自适应选择边界信息或上下文信息；
4. Residual + small scale：避免破坏原始预训练特征。
"""

import torch
import torch.nn as nn
from ..core import register


def get_act(act: str = "silu"):
    act = act.lower()
    if act == "silu":
        return nn.SiLU(inplace=True)
    if act == "relu":
        return nn.ReLU(inplace=True)
    if act == "gelu":
        return nn.GELU()
    if act in ["none", "identity"]:
        return nn.Identity()
    raise ValueError(f"Unsupported activation: {act}")


class ConvBNAct(nn.Module):
    """
    Conv + BN + Activation

    注意：
    这里必须显式支持 dilation。
    否则 context branch 使用 p=dilation 但卷积本身 dilation=1，
    会导致输出尺寸变大，例如 60 -> 62。
    """

    def __init__(
        self,
        in_ch,
        out_ch,
        k=1,
        s=1,
        p=None,
        groups=1,
        dilation=1,
        act="silu",
    ):
        super().__init__()

        if p is None:
            p = dilation * (k - 1) // 2

        self.conv = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=k,
            stride=s,
            padding=p,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = get_act(act)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BCFEBlock(nn.Module):
    """
    单层特征增强模块。

    输入:
        x: [B, C, H, W]

    输出:
        x': [B, C, H, W]

    结构:
        boundary = local DWConv branch
        context  = dilated DWConv branch
        gate     = global gate
        out      = x + scale * (gate * boundary + (1 - gate) * context)
    """

    def __init__(
        self,
        channels: int,
        hidden_dim: int = 64,
        dilation: int = 2,
        gate_reduction: int = 16,
        init_scale: float = 0.1,
        act: str = "silu",
    ):
        super().__init__()

        hidden_dim = min(hidden_dim, channels)
        gate_hidden = max(channels // gate_reduction, 16)

        # 边界分支：局部 3x3 depthwise，强调轮廓、边缘、接触区域
        self.boundary_branch = nn.Sequential(
            ConvBNAct(
                channels,
                hidden_dim,
                k=1,
                act=act,
            ),
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                k=3,
                p=1,
                groups=hidden_dim,
                dilation=1,
                act=act,
            ),
            ConvBNAct(
                hidden_dim,
                channels,
                k=1,
                act="none",
            ),
        )

        # 上下文分支：空洞 depthwise，扩大局部邻域感受野
        # 关键修正：这里必须传 dilation=dilation
        self.context_branch = nn.Sequential(
            ConvBNAct(
                channels,
                hidden_dim,
                k=1,
                act=act,
            ),
            ConvBNAct(
                hidden_dim,
                hidden_dim,
                k=3,
                p=dilation,
                groups=hidden_dim,
                dilation=dilation,
                act=act,
            ),
            ConvBNAct(
                hidden_dim,
                channels,
                k=1,
                act="none",
            ),
        )

        # 门控分支：判断当前区域更依赖边界还是上下文
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, gate_hidden, kernel_size=1, bias=True),
            get_act(act),
            nn.Conv2d(gate_hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        # 小尺度残差，保证训练初期稳定
        self.scale = nn.Parameter(torch.ones(1) * init_scale)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

        # 让最后一个 gate conv 初始输出接近 0，
        # sigmoid(0)=0.5，避免一开始偏向 boundary 或 context
        last_gate_conv = self.gate[-2]
        if isinstance(last_gate_conv, nn.Conv2d):
            nn.init.zeros_(last_gate_conv.weight)
            nn.init.zeros_(last_gate_conv.bias)

    def forward(self, x):
        boundary = self.boundary_branch(x)
        context = self.context_branch(x)
        gate = self.gate(x)

        # 保险处理：
        # 正常情况下 boundary/context 都应该和 x 尺寸一致。
        # 如果后续改 dilation/kernel 造成 1 像素误差，这里可以避免训练直接崩。
        if boundary.shape[-2:] != x.shape[-2:]:
            boundary = torch.nn.functional.interpolate(
                boundary,
                size=x.shape[-2:],
                mode="nearest",
            )

        if context.shape[-2:] != x.shape[-2:]:
            context = torch.nn.functional.interpolate(
                context,
                size=x.shape[-2:],
                mode="nearest",
            )

        out = gate * boundary + (1.0 - gate) * context
        return x + self.scale * out


@register()
class BCFE(nn.Module):
    """
    多尺度 BCFE 包装器。

    in_channels:
        与 backbone 输出特征通道一致。
        你当前 HGNetv2-N 配置通常是 [512, 1024]。

    levels:
        要增强哪些特征层。
        levels=[0] 表示只增强 backbone 返回的第一个特征层。
        当前建议先用 levels=[0]，不要一开始全开。
    """

    def __init__(
        self,
        in_channels,
        levels=(0,),
        hidden_dim=64,
        dilation=2,
        gate_reduction=16,
        init_scale=0.1,
        act="silu",
    ):
        super().__init__()

        self.in_channels = list(in_channels)
        self.levels = set(levels)

        blocks = []
        for i, c in enumerate(self.in_channels):
            if i in self.levels:
                blocks.append(
                    BCFEBlock(
                        channels=c,
                        hidden_dim=hidden_dim,
                        dilation=dilation,
                        gate_reduction=gate_reduction,
                        init_scale=init_scale,
                        act=act,
                    )
                )
            else:
                blocks.append(nn.Identity())

        self.blocks = nn.ModuleList(blocks)

    def forward(self, feats):
        assert isinstance(feats, (list, tuple)), (
            f"BCFE expects list/tuple features, but got {type(feats)}"
        )

        assert len(feats) == len(self.blocks), (
            f"BCFE got {len(feats)} feature maps, "
            f"but configured with {len(self.blocks)} blocks. "
            f"Please check BCFE.in_channels and HGNetv2.return_idx."
        )

        outs = []
        for i, feat in enumerate(feats):
            outs.append(self.blocks[i](feat))

        return outs