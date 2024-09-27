__all__ = ["ViTAdapter"]

import math
from collections.abc import Callable
from functools import partial
from typing import Annotated, Any, Literal, Optional, Sequence, Union

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.layers import LayerType, Mlp, PatchEmbed, SwiGLUPacked
from timm.models._builder import build_model_with_cfg

# from timm.models._features import feature_take_indices
from timm.models._registry import generate_default_cfgs, register_model
from timm.models.features import feature_take_indices
from timm.models.vision_transformer import Block

from ..modules.vit_adapter import (
    InteractionBlock,
    SpatialPriorModule,
    deform_inputs,
)
from ms_deform_attn import MSDeformAttn


class ViTAdapter(timm.models.VisionTransformer):
    """Vision Transformer Adapter TIMM Model Implementation"""

    def __init__(
        self,
        img_size: Union[int, tuple[int, int]] = 224,
        patch_size: Union[int, tuple[int, int]] = 16,
        in_chans: int = 3,
        num_classes: int = 1000,
        global_pool: Literal["", "avg", "avgmax", "max", "token", "map"] = "token",
        embed_dim: int = 768,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = True,
        qk_norm: bool = False,
        init_values: Optional[float] = None,
        class_token: bool = True,
        pos_embed: str = "learn",
        no_embed_class: bool = False,
        reg_tokens: int = 0,
        pre_norm: bool = False,
        fc_norm: Optional[bool] = None,
        dynamic_img_size: bool = False,
        dynamic_img_pad: bool = False,
        drop_rate: float = 0.0,
        pos_drop_rate: float = 0.0,
        patch_drop_rate: float = 0.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        weight_init: Literal["skip", "jax", "jax_nlhb", "moco", ""] = "",
        fix_init: bool = False,
        embed_layer: Callable = PatchEmbed,
        norm_layer: Optional[LayerType] = None,
        act_layer: Optional[LayerType] = None,
        block_fn: type[nn.Module] = Block,
        mlp_layer: str = "mlp",
        # vit adapter params
        spm_in_chans: int = 3,  # to train with augmented rgb data and frozen rgb bb
        interact_indices: Optional[Sequence[Annotated[Sequence[int], 2]]] = None,
        spm_inplanes: int = 64,
        deform_num_points: int = 4,
        deform_num_heads: int = 6,
        cffn_ratio: Optional[float] = 0.25,
        deform_ratio: float = 1.0,
        add_vit_feature: bool = True,
        extra_extractors: bool = True,
        freeze_vit: bool = True,
    ) -> None:
        """
        :param interact_indices:
            A list of tuple indicating start and end of each interaction block.
            Indices are relative to the number of blocks in the TIMM VisionTransformer
            implementation
        :param spm_inplane:
            The number used as a basis for determining SPM convolutions output channels
        :param deform_num_points:
            The number of samplling points for deformable attention
        :param deform_num_heads:
            The number of heads used in the deformable attention, must divide the number
            of tokens
        :param cffn_ratio:
            The of hidden features relative to the embed dimension in the extractor
        :param deform_ratio:
            Internal projection ratio relative to embed dimension for the deformable
            attention
        :param add_vit_features:
            Flag to indicate if the features processed trough the vit backbone should
            be added to the final output
        :param extra_extractors:
            Flag to indicate if the last extractor module should have extra extractors
        :param freeze_vit:
            Flag to indicate if the VisionTransformer backbone should be frozen

        """
        # in TIMM mlp_layer is passed as class type, so we do this to maintain
        # compatibility but have it easily instantiable with hydra
        if mlp_layer == "mlp":
            mlp_layer = Mlp
        elif mlp_layer == "swiglu":
            mlp_layer = SwiGLUPacked

        super().__init__(
            img_size,
            patch_size,
            in_chans,
            num_classes,
            global_pool,
            embed_dim,
            depth,
            num_heads,
            mlp_ratio,
            qkv_bias,
            qk_norm,
            init_values,
            class_token,
            pos_embed,
            no_embed_class,
            reg_tokens,
            pre_norm,
            fc_norm,
            dynamic_img_size,
            dynamic_img_pad,
            drop_rate,
            pos_drop_rate,
            patch_drop_rate,
            proj_drop_rate,
            attn_drop_rate,
            drop_path_rate,
            weight_init,
            fix_init,
            embed_layer,
            norm_layer,
            act_layer,
            block_fn,
            mlp_layer,
        )

        # freeze the VisionTransformer backbone
        if freeze_vit:
            for p in self.parameters():
                p.requires_grad = False

        self.vit_in_chans = in_chans

        init_values = 0.0
        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        if isinstance(patch_size, int):
            self.patch_size = (patch_size, patch_size)
        elif isinstance(patch_size, tuple):
            self.patch_size = patch_size

        # if no interact_indexes create interactions by dividing backbone blocks in
        # 4 (almost) equal parts
        # TODO: improve math
        if interact_indices is None:
            n = len(self.blocks)
            interact_indices = [
                [int(i * n / 4), int((i * n + n) / 4)] for i in range(4)
            ]
        assert len(interact_indices) == 4
        self.interact_indexes = interact_indices

        self.add_vit_feature = add_vit_feature

        self.lvl_embed = nn.Parameter(torch.zeros(3, embed_dim))

        self.spm = SpatialPriorModule(
            in_chans=spm_in_chans, inplanes=spm_inplanes, embed_dim=embed_dim
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    embed_dim=embed_dim,
                    num_heads=deform_num_heads,
                    num_points=deform_num_points,
                    init_values=init_values,
                    drop_path_rate=drop_path_rate,
                    norm_layer=norm_layer,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractors=(
                        (True if i == len(interact_indices) - 1 else False)
                        and extra_extractors
                    ),
                )
                for i in range(len(interact_indices))
            ]
        )

        # replace vit feature_info with the interactions feature_info
        reduction = (
            self.patch_embed.feat_ratio()
            if hasattr(self.patch_embed, "feat_ratio")
            else patch_size
        )
        self.feature_info = [
            dict(module=f"interactions.{i}", num_chs=embed_dim, reduction=reduction)
            for i in range(len(interact_indices))
        ]

        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)

        nn.init.normal_(self.lvl_embed)

    def _init_weights(self, m: nn.Module) -> None:
        super()._init_weights(m)  # XXX: check if this is ok

        match m:
            case nn.Linear():
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            case nn.LayerNorm() | nn.BatchNorm2d():
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            case nn.Conv2d() | nn.ConvTranspose2d():
                fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                fan_out //= m.groups
                m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
                if m.bias is not None:
                    m.bias.data.zero_()
            case _:  # XXX: is this necessary?
                pass

    # XXX: maybe merge with _init_weigts() ?
    def _init_deform_weights(self, m: nn.Module) -> None:
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(
        self, c2: torch.Tensor, c3: torch.Tensor, c4: torch.Tensor
    ) -> tuple[torch.Tensor, ...]:
        c2 = c2 + self.lvl_embed[0]
        c3 = c3 + self.lvl_embed[1]
        c4 = c4 + self.lvl_embed[2]

        return c2, c3, c4

    def _calculate_c_spatial_shapes(self, x: torch.Tensor) -> Sequence[tuple[int, int]]:
        """Helper function to calculate spatial shapes of the feature pyramid

        :param x: the input tensor of size (B, Cin, H, W)
        :returns:
            this returns a list of shapes, 1 for each feature map of the pyramid, 4 in
            total
        """
        _, _, H, W = x.shape  # (B, C, H, W)

        H_c1 = math.floor(math.floor(H / 2 + 0.5) / 2 + 0.5)  # /4
        H_c2 = math.floor(H_c1 / 2 + 0.5)  # /8
        H_c3 = math.floor(H_c2 / 2 + 0.5)  # /16
        H_c4 = math.floor(H_c3 / 2 + 0.5)  # /32

        W_c1 = math.floor(math.floor(W / 2 + 0.5) / 2 + 0.5)  # /4
        W_c2 = math.floor(W_c1 / 2 + 0.5)  # /8
        W_c3 = math.floor(W_c2 / 2 + 0.5)  # /16
        W_c4 = math.floor(W_c3 / 2 + 0.5)  # /32

        s = [(H_c1, W_c1), (H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4)]

        return s

    def prune_intermediate_layers(
        self,
        indices: Union[int, list[int], tuple[int]] = 1,
        prune_norm: bool = False,
        prune_head: bool = True,
    ) -> list[int]:
        """Prune layers not required for specified intermediates."""

        take_indices, max_index = feature_take_indices(len(self.interactions), indices)
        # XXX: since we don't care about intermediates feature pyramids we alway need to
        #   go to last layer
        # self.interactions = self.interactions[: max_index + 1]  # truncate blocks
        if prune_norm:
            self.norm = nn.Identity()
        if prune_head:
            self.fc_norm = nn.Identity()
            self.reset_classifier(0, "")
        return take_indices

    # TODO: improve the way pretrained backbone is fed augmented data
    def forward_spatial_features(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Process the input trough the ViTAdapter"""
        # frozen backbone trained on dinov2 ds only works on rgb so select only 3 first
        x_vit = x[:, 0 : self.vit_in_chans, ...]
        x_spm = x

        B, _, H, W = x_vit.shape  # (B, C, H, W)
        c_spatial_shapes = self._calculate_c_spatial_shapes(x_vit)

        # Compute Spatial Priors
        deform_inputs_injector, deform_inputs_extractor = deform_inputs(
            x_vit, self.patch_size, c_spatial_shapes
        )
        # XXX: c_spatial_shapes could also be returned by spm, because this is just the
        #   size of the reshaped feature maps so, sqrt(Ni) for feature map of shape
        #   (B, C, Ni), wouldn't work in case of non square input/patch_size
        c1, c2, c3, c4 = self.spm(x_spm)  # (B, Ni, C)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)  # (B, N2+N3+N4, C)

        # Embeddings
        x = self.patch_embed(x_vit)  # (B, N, C)
        x = self._pos_embed(x)
        x = self.norm_pre(x)

        # Forward Blocks
        # TODO: handle case with register token
        if self.num_prefix_tokens:
            cls = x[:, :1]
            # reg = x[:, 1 : self.num_prefix_tokens]
            x = x[:, self.num_prefix_tokens :]
        else:
            cls = None

        # math is from : https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
        # H_grid and W_grid represent the height and width of the projected tokens grid
        # aka the height and width of the output of the Conv2Dconv2d
        H_grid = math.floor((H - (self.patch_size[0] - 1) - 1) / self.patch_size[0] + 1)
        W_grid = math.floor((W - (self.patch_size[1] - 1) - 1) / self.patch_size[1] + 1)

        _, _, C = x.shape

        # TODO: manual blocks forwarding and blocks interactions in this loop for better
        #   readability
        intermediates = list()
        for i, layer in enumerate(self.interactions):
            indexes = self.interact_indexes[i]

            x, c, cls = layer(
                x=x,
                c=c,
                cls=cls,
                blocks=self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs_injector=deform_inputs_injector,
                deform_inputs_extractor=deform_inputs_extractor,
                c_spatial_shapes=c_spatial_shapes,
            )

            intermediates.append(
                x.transpose(1, 2).view(B, C, H_grid, W_grid).contiguous()
            )

        # Split & Reshape Feature Pyramid
        (H_c1, W_c1), (H_c2, W_c2), (H_c3, W_c3), (H_c4, W_c4) = c_spatial_shapes

        i1 = H_c2 * W_c2
        i2 = i1 + (H_c3 * W_c3)

        c2 = c[:, 0:i1, :]
        c3 = c[:, i1:i2, :]
        c4 = c[:, i2:, :]

        c2 = c2.transpose(1, 2).view(B, C, H_c2, W_c2).contiguous()
        c3 = c3.transpose(1, 2).view(B, C, H_c3, W_c3).contiguous()
        c4 = c4.transpose(1, 2).view(B, C, H_c4, W_c4).contiguous()
        c1 = self.up(c2) + c1

        # Add the final output of the vit backbone to the adapter output
        if self.add_vit_feature:
            x1, x2, x3, x4 = intermediates

            # need to interpolate because features of transformer layers to match
            # spatial feature pyramid sizes
            x1 = F.interpolate(
                x1, size=(H_c1, W_c1), mode="bilinear", align_corners=False
            )
            x2 = F.interpolate(
                x2, size=(H_c2, W_c2), mode="bilinear", align_corners=False
            )
            x3 = F.interpolate(
                x3, size=(H_c3, W_c3), mode="bilinear", align_corners=False
            )
            x4 = F.interpolate(
                x4, size=(H_c4, W_c4), mode="bilinear", align_corners=False
            )

            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        # TODO: check the effect of normalization of output on Mask2Former
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)

        return f1, f2, f3, f4

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.forward_spatial_features(x)

    # XXX: since every intermediate outputs a feature pyramid, we need to always
    #   forward the whole network in order to get it, so
    #   forward_intermediates == forward
    def forward_intermediates(
        self,
        x: torch.Tensor,
        indices: Optional[Union[int, list[int], list[int]]] = None,
        return_prefix_tokens: bool = False,
        norm: bool = False,
        output_fmt: str = "NCHW",
        intermediates_only: bool = False,
    ) -> tuple[torch.Tensor, ...]:
        # XXX: for now only returns the spatial feature pyramid
        #   in original implem the final features can be returned too
        f1, f2, f3, f4 = self.forward(x)

        return f1, f2, f3, f4


def _cfg(url: str = "", **kwargs) -> dict[str, Any]:  # type: ignore
    return {
        "url": url,
        "num_classes": 1000,
        "input_size": (3, 518, 518),
        "pool_size": None,
        "crop_pct": 1.0,
        "interpolation": "bicubic",
        "fixed_input_size": True,
        "mean": IMAGENET_DEFAULT_MEAN,
        "std": IMAGENET_DEFAULT_STD,
        "first_conv": "patch_embed.proj",
        "classifier": "head",
        **kwargs,
    }


# default_cfgs can also point to a local file by replacing url="url/to/resource.com"
# with file="path/to/file.pth"
# XXX: in our case the lvd142m tag is inexact, because only the vit backbone is
#   pretrained on that dataset, the adapter part is randomly initialized
default_cfgs = {
    "vit_adapter_small_patch14_518.lvd142m": _cfg(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/dinov2_vits14_pretrain.pth",  # noqa: E501
        license="apache-2.0",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
        input_size=(3, 518, 518),
        crop_pct=1.0,
    ),
    "vit_adapter_base_patch14_518.lvd142m": _cfg(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth",  # noqa: E501
        license="apache-2.0",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
        input_size=(3, 518, 518),
        crop_pct=1.0,
    ),
    "vit_adapter_large_patch14_518.lvd142m": _cfg(
        url="https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth",  # noqa: E501
        license="apache-2.0",
        mean=IMAGENET_DEFAULT_MEAN,
        std=IMAGENET_DEFAULT_STD,
        num_classes=0,
        input_size=(3, 518, 518),
        crop_pct=1.0,
    ),
}

# this variable is parsed by TIMM to access default cfgs when registering models
default_cfgs = generate_default_cfgs(default_cfgs)


def _create_vision_transformer_adapter(  # type: ignore
    variant: str, pretrained: bool = False, **kwargs
) -> ViTAdapter:
    out_indices = kwargs.pop("out_indices", 3)

    strict = False  # False because checkpoints miss
    return build_model_with_cfg(
        ViTAdapter,
        variant,
        pretrained,
        pretrained_strict=strict,
        feature_cfg=dict(out_indices=out_indices, feature_cls="getter"),
        **kwargs,
    )


# Register model variants in TIMM
@register_model
def vit_adapter_small_patch14_518(  # type: ignore
    pretrained: bool = False, **kwargs
) -> ViTAdapter:
    """ViTAdaper-Base for DINOv2 configuration"""
    model_args = dict(
        img_size=518,
        patch_size=14,
        embed_dim=384,
        depth=12,
        num_heads=6,
        init_values=1e-5,
        interact_indices=[(0, 2), (3, 5), (6, 8), (9, 11)],
        deform_num_heads=6,
    )

    model = _create_vision_transformer_adapter(
        "vit_adapter_small_patch14_518",
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )

    return model


@register_model
def vit_adapter_base_patch14_518(  # type: ignore
    pretrained: bool = False, **kwargs
) -> ViTAdapter:
    """ViTAdaper-Base for DINOv2 configuration"""
    model_args = dict(
        img_size=518,
        patch_size=14,
        embed_dim=768,
        depth=12,
        num_heads=12,
        init_values=1e-5,
        interact_indices=[(0, 2), (3, 5), (6, 8), (9, 11)],
        deform_num_heads=12,
    )

    model = _create_vision_transformer_adapter(
        "vit_adapter_base_patch14_518",
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )

    return model


@register_model
def vit_adapter_large_patch14_518(  # type: ignore
    pretrained: bool = False, **kwargs
) -> ViTAdapter:
    """ViTAdaper-Base for DINOv2 configuration"""
    model_args = dict(
        img_size=518,
        patch_size=14,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        init_values=1e-5,
        interact_indices=[(0, 5), (6, 11), (12, 17), (18, 23)],
        deform_num_heads=16,
    )

    model = _create_vision_transformer_adapter(
        "vit_adapter_large_patch14_518",
        pretrained=pretrained,
        **dict(model_args, **kwargs),
    )

    return model
