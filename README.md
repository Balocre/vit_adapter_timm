# ViTAdapter TIMM

This project is an implementation of the Vision Transformer Adapter decribed in this [paper](https://arxiv.org/pdf/2205.08534).

I tried following the same standard as the TIMM implementation of the [Vision Transformer](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py).
Therefore the model integrates with the TIMM library and it can be used as any TIMM model, it can also be used as a TIMM backbone in the HuggingFace transformers library.

You will need to install my implementation of the CUDA deformable attention in order to have the model work, you can find it [here](https://github.com/Balocre/ms_deform_attn)

The vision transformer adapter can be used as a backbone for any model requiring a feature pyramid from the HuggingFace library, it was tested it with Mask2Former.

There is still some work that needs to be done, particularily on the intermediate feature extraction. I will add more customization options, with regards to the number of interaction block and the type of attention used.
