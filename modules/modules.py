from collections.abc import Callable
from functools import partial
from typing import Optional, Sequence

# TODO: remove timm dependence
import torch
import torch.nn as nn
from timm.layers import DropPath
from timm.models.vision_transformer import Block

from skynet.ops.ms_deform_attn.src.ms_deform_attn import MSDeformAttn


class SpatialFeatPyramidDWConv(nn.Module):
    """DepthWise Convolution Module for the ViTAdapater Module"""

    def __init__(self, dim: int = 768):
        super().__init__()

        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    # XXX: the dimensions used in this layer seem kinda arbitrary
    def forward(
        self,
        c: torch.Tensor,
        c_spatial_shapes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """Forward of the DepthWise Convolution Module

        :param x: Input tensor of shape (B, N1+N2+N3, C)
        :returns: Processed input, where each feature map has been processed
            by a FFN Module and concatenated into a single tensor
        """
        B, _, C = c.shape  # (B, N2+N3+N4, C)

        # x comes to the function as vector of size (B, N1+N2+N3, C) in the function
        # so we have to split to process each feature map independently
        _, (H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4) = c_spatial_shapes

        i1 = H_c2 * W_c2  # index of slice for first feature map values
        i2 = i1 + (H_c3 * W_c3)  # second feature map

        c2 = c[:, 0:i1, :].transpose(1, 2).view(B, C, H_c2, W_c2).contiguous()
        c3 = c[:, i1:i2, :].transpose(1, 2).view(B, C, H_c3, W_c3).contiguous()
        c4 = c[:, i2:, :].transpose(1, 2).view(B, C, H_c4, W_c4).contiguous()

        c2 = self.dwconv(c2).flatten(2).transpose(1, 2)  # (B, N2, C)
        c3 = self.dwconv(c3).flatten(2).transpose(1, 2)  # (B, N3, C)
        c4 = self.dwconv(c4).flatten(2).transpose(1, 2)  # (B, N4, C)

        c = torch.cat([c2, c3, c4], dim=1)

        return c


class SpatialFeatPyramidConvFFN(nn.Module):
    """Convolutionnal Feed Forward Network for the ViTAdapater Module"""

    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable = nn.GELU,
        drop_rate: float = 0.0,
    ) -> None:
        super().__init__()

        hidden_features = hidden_features or in_features
        out_features = out_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = SpatialFeatPyramidDWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop_rate)

    def forward(
        self,
        c: torch.Tensor,
        c_spatial_shapes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        c = self.fc1(c)
        c = self.dwconv(c, c_spatial_shapes)
        c = self.act(c)
        c = self.drop(c)

        c = self.fc2(c)
        c = self.drop(c)

        return c


class Injector(nn.Module):
    """Injector Module for adding spatial information in the transformer"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        num_points: int = 4,
        num_levels: int = 1,
        deform_ratio: float = 1.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
        init_values: float = 0.0,
    ) -> None:
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.attn = MSDeformAttn(
            hidden_dim=embed_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            proj_ratio=deform_ratio,
        )
        self.gamma = nn.Parameter(
            init_values * torch.ones((embed_dim)), requires_grad=True
        )

    def forward(
        self,
        q: torch.Tensor,
        reference_points: torch.Tensor,
        kv: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param q: Vision transformer block output used as query for
            cross-attention of size (B, N, C)
        :param kv: Spatial feature used as key/value for cross-attention of size
            (B, N2+N3+N4, C)
        :return: New flattened spatial feature pyramid of size (B, N, C)
        """

        attn = self.attn(
            self.norm1(q),
            reference_points,
            self.norm2(kv),
            spatial_shapes,
            level_start_index,
            None,
        )  # (B, N, C)

        return q + self.gamma * attn


class Extractor(nn.Module):
    """Extractor Module for extracting information from the backbone"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        num_points: int = 4,
        num_levels: int = 1,
        deform_ratio: float = 1.0,
        cffn_ratio: Optional[float] = 0.25,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.attn = MSDeformAttn(
            hidden_dim=embed_dim,
            num_levels=num_levels,
            num_heads=num_heads,
            num_points=num_points,
            proj_ratio=deform_ratio,
        )

        if cffn_ratio is not None:
            self.ffn = SpatialFeatPyramidConvFFN(
                in_features=embed_dim,
                hidden_features=int(embed_dim * cffn_ratio),
                drop_rate=drop_rate,
            )
            self.has_ffn = True
            self.norm = norm_layer(embed_dim)
            self.drop_path = (
                DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()
            )

    def forward(
        self,
        q: torch.Tensor,  # (B, HW/8**2+HW/16**2+HW/32**2, C)
        reference_points: torch.Tensor,
        kv: torch.Tensor,  # (B, N, C)
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        c_spatial_shapes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        """
        :param q: Spatial feature used as query for cross-attention of size
            (B, N2+N3+N4, C)
        :param kv: Vision transformer block output used as key/value for
            cross-attention of size (B, N, C)
        :param c_spatial_shapes: Expected spatial shapes of the spatial feature
            pyramid
        :returns: New spatial feature of size (B, N, C)
        """

        attn = self.attn(
            self.norm1(q),
            reference_points,
            self.norm2(kv),
            spatial_shapes,
            level_start_index,
            None,
        )
        q = q + attn

        if self.has_ffn:
            q = q + self.drop_path(self.ffn(self.norm(q), c_spatial_shapes))

        return q


class InteractionBlock(nn.Module):
    """Block of interaction with the backbone"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 6,
        num_points: int = 4,
        drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        cffn_ratio: Optional[float] = 0.25,
        init_values: float = 0.0,
        deform_ratio: float = 1.0,
        extra_extractors: bool = False,
        norm_layer: Callable = partial(nn.LayerNorm, eps=1e-6),
    ) -> None:
        super().__init__()

        self.injector = Injector(
            embed_dim,
            num_levels=3,
            num_heads=num_heads,
            init_values=init_values,
            num_points=num_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
        )

        self.extractor = Extractor(
            embed_dim,
            num_levels=1,
            num_heads=num_heads,
            num_points=num_points,
            norm_layer=norm_layer,
            deform_ratio=deform_ratio,
            cffn_ratio=cffn_ratio,
            drop_rate=drop_rate,
            drop_path_rate=drop_path_rate,
        )

        self.has_extra_extracors = extra_extractors

        if extra_extractors:
            self.extras = nn.Sequential(
                *[
                    Extractor(
                        embed_dim,
                        num_heads=num_heads,
                        num_points=num_points,
                        norm_layer=norm_layer,
                        cffn_ratio=cffn_ratio,
                        deform_ratio=deform_ratio,
                        drop_rate=drop_rate,
                        drop_path_rate=drop_path_rate,
                    )
                    for _ in range(2)
                ]
            )
        else:
            self.extras = None

    def forward_injector(
        self, x: torch.Tensor, c: torch.Tensor, deform_inputs: Sequence[torch.Tensor]
    ) -> torch.Tensor:
        x = self.injector(
            q=x,  # feat_vit
            reference_points=deform_inputs[0],
            kv=c,  # feat_sp
            spatial_shapes=deform_inputs[1],
            level_start_index=deform_inputs[2],
        )

        return x

    def forward_extractor(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        deform_inputs: Sequence[torch.Tensor],
        c_spatial_shapes: Sequence[tuple[int, int]],
    ) -> torch.Tensor:
        c = self.extractor(
            q=c,  # feat_sp
            reference_points=deform_inputs[0],
            kv=x,  # feat_vit
            spatial_shapes=deform_inputs[1],
            level_start_index=deform_inputs[2],
            c_spatial_shapes=c_spatial_shapes,
        )

        if self.extras is not None:
            for e in self.extras:
                c = e(
                    q=c,
                    reference_points=deform_inputs[0],
                    kv=x,
                    spatial_shapes=deform_inputs[1],
                    level_start_index=deform_inputs[2],
                    c_spatial_shapes=c_spatial_shapes,
                )

        return c

    def forward_blocks(self, x: torch.Tensor, blocks: Sequence[Block]) -> torch.Tensor:
        for b in blocks:
            x = b(x)

        return x

    def forward_blocks_with_cls(
        self, x: torch.Tensor, cls: torch.Tensor, blocks: Sequence[Block]
    ) -> tuple[torch.Tensor, ...]:
        x = torch.cat((cls, x), dim=1)

        x = self.forward_blocks(x, blocks)

        cls = x[:, :1]
        x = x[:, 1:]

        return x, cls

    def forward(
        self,
        x: torch.Tensor,
        c: torch.Tensor,
        blocks: Sequence[Block],
        deform_inputs_injector: Sequence[torch.Tensor],
        deform_inputs_extractor: Sequence[torch.Tensor],
        c_spatial_shapes: Sequence[tuple[int, int]],
        cls: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, ...]:
        """Forward of the InteractionBlock

        :param x: Transformer feature
        :param c: Spatial feature map
        :param blocks: List of blocks to be forwarded in the IntaractionBlock
        :param c_spatial_shapes: Expected shapes of the feature maps from the spatial
            feature pyramid
        :returns:
            - Processed feature with spatial information injected

            - Processed spatial feature map

            - [CLS] Token
        """
        x = self.forward_injector(x, c, deform_inputs_injector)

        if cls is None:
            x = self.forward_blocks(x, blocks)
        else:
            x, cls = self.forward_blocks_with_cls(x, cls, blocks)

        c = self.forward_extractor(x, c, deform_inputs_extractor, c_spatial_shapes)

        return x, c, cls


class SpatialPriorModule(nn.Module):
    """Spatial Prior Module for Extracting Local Semantics of Input"""

    def __init__(
        self,
        in_chans: int = 3,
        inplanes: int = 64,
        embed_dim: int = 384,
    ) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            *[
                nn.Conv2d(
                    in_chans, inplanes, kernel_size=3, stride=2, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.Conv2d(
                    inplanes, inplanes, kernel_size=3, stride=1, padding=1, bias=False
                ),
                nn.SyncBatchNorm(inplanes),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ]
        )

        self.conv2 = nn.Sequential(
            *[
                nn.Conv2d(
                    inplanes,
                    2 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(2 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv3 = nn.Sequential(
            *[
                nn.Conv2d(
                    2 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )
        self.conv4 = nn.Sequential(
            *[
                nn.Conv2d(
                    4 * inplanes,
                    4 * inplanes,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.SyncBatchNorm(4 * inplanes),
                nn.ReLU(inplace=True),
            ]
        )

        self.fc1 = nn.Conv2d(
            inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc2 = nn.Conv2d(
            2 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc3 = nn.Conv2d(
            4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )
        self.fc4 = nn.Conv2d(
            4 * inplanes, embed_dim, kernel_size=1, stride=1, padding=0, bias=True
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        c1 = self.stem(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)

        c1 = self.fc1(c1)
        c2 = self.fc2(c2)
        c3 = self.fc3(c3)
        c4 = self.fc4(c4)

        B, C, _, _ = c1.shape  # (B, C, /4, /4)
        # c1 = c1.view(bs, dim, -1).transpose(1, 2)  # /4
        c2 = c2.view(B, C, -1).transpose(1, 2)  # /8
        c3 = c3.view(B, C, -1).transpose(1, 2)  # /16
        c4 = c4.view(B, C, -1).transpose(1, 2)  # /32

        return c1, c2, c3, c4


def get_reference_points(spatial_shapes: torch.Tensor, device: str) -> torch.Tensor:
    reference_points_list = []

    for lvl, (H_ci, W_ci) in enumerate(spatial_shapes):
        ref_y, ref_x = torch.meshgrid(
            torch.linspace(0.5, H_ci - 0.5, H_ci, dtype=torch.float32, device=device),
            torch.linspace(0.5, W_ci - 0.5, W_ci, dtype=torch.float32, device=device),
        )

        ref_y = ref_y.reshape(-1)[None] / H_ci
        ref_x = ref_x.reshape(-1)[None] / W_ci
        ref = torch.stack((ref_x, ref_y), -1)

        reference_points_list.append(ref)

    reference_points = torch.cat(reference_points_list, 1)
    reference_points = reference_points[:, :, None]

    return reference_points


def deform_inputs(
    x: torch.Tensor,
    patch_size: tuple[int, int],
    c_spatial_shapes: Sequence[tuple[int, int]],
) -> tuple[Sequence[torch.Tensor], ...]:
    _, _, H, W = x.shape
    _, (H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4) = c_spatial_shapes

    spatial_shapes = torch.as_tensor(
        [(H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4)],
        dtype=torch.long,
        device=x.device,
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(H // patch_size[0], W // patch_size[1])], x.device
    )
    deform_inputs1 = [reference_points, spatial_shapes, level_start_index]

    spatial_shapes = torch.as_tensor(
        [(H // patch_size[0], W // patch_size[1])], dtype=torch.long, device=x.device
    )
    level_start_index = torch.cat(
        (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
    )
    reference_points = get_reference_points(
        [(H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4)], x.device
    )
    deform_inputs2 = [reference_points, spatial_shapes, level_start_index]

    return deform_inputs1, deform_inputs2
