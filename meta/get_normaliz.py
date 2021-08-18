# import paddle
import paddle.nn as nn
# from paddle.nn import functional as F
def get_norm(norm, out_channels):
    """
    Args:
        norm (str or callable): either one of BN, SyncBN, FrozenBN, GN;
            or a callable that takes a channel number and returns
            the normalization layer as a nn.Module.

    Returns:
        nn.Module or None: the normalization layer
    """
    if norm is None:
        return None
    if isinstance(norm, str):
        if len(norm) == 0:
            return None
        norm = {
            # "BN": BatchNorm2d,
            # # Fixed in https://github.com/pytorch/pytorch/pull/36382
            # "SyncBN": NaiveSyncBatchNorm if env.TORCH_VERSION <= (1, 5) else nn.SyncBatchNorm,
            # "FrozenBN": FrozenBatchNorm2d,
            "GN": lambda channels: nn.GroupNorm(32, channels),
            # # for debugging:
            # "nnSyncBN": nn.SyncBatchNorm,
            # "naiveSyncBN": NaiveSyncBatchNorm,
        }[norm]
    return norm(out_channels)

# class FrozenBatchNorm2d(nn.Layer):
#     """
#     BatchNorm2d where the batch statistics and the affine parameters are fixed.

#     It contains non-trainable buffers called
#     "weight" and "bias", "running_mean", "running_var",
#     initialized to perform identity transformation.

#     The pre-trained backbone models from Caffe2 only contain "weight" and "bias",
#     which are computed from the original four parameters of BN.
#     The affine transform `x * weight + bias` will perform the equivalent
#     computation of `(x - running_mean) / sqrt(running_var) * weight + bias`.
#     When loading a backbone model from Caffe2, "running_mean" and "running_var"
#     will be left unchanged as identity transformation.

#     Other pre-trained backbone models may contain all 4 parameters.

#     The forward is implemented by `F.batch_norm(..., training=False)`.
#     """

#     _version = 3

#     def __init__(self, num_features, eps=1e-5):
#         super().__init__()
#         self.num_features = num_features
#         self.eps = eps
#         self.register_buffer("weight", paddle.ones(num_features))
#         self.register_buffer("bias", paddle.zeros(num_features))
#         self.register_buffer("running_mean", paddle.zeros(num_features))
#         self.register_buffer("running_var", paddle.ones(num_features) - eps)

#     def forward(self, x):
#         if x.requires_grad:
#             # When gradients are needed, F.batch_norm will use extra memory
#             # because its backward op computes gradients for weight/bias as well.
#             scale = self.weight * (self.running_var + self.eps).rsqrt()
#             bias = self.bias - self.running_mean * scale
#             scale = scale.reshape(1, -1, 1, 1)
#             bias = bias.reshape(1, -1, 1, 1)
#             out_dtype = x.dtype  # may be half
#             return x * scale.to(out_dtype) + bias.to(out_dtype)
#         else:
#             # When gradients are not needed, F.batch_norm is a single fused op
#             # and provide more optimization opportunities.
#             return F.batch_norm(
#                 x,
#                 self.running_mean,
#                 self.running_var,
#                 self.weight,
#                 self.bias,
#                 training=False,
#                 eps=self.eps,
#             )

#     def _load_from_state_dict(
#         self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#     ):
#         version = local_metadata.get("version", None)

#         if version is None or version < 2:
#             # No running_mean/var in early versions
#             # This will silent the warnings
#             if prefix + "running_mean" not in state_dict:
#                 state_dict[prefix + "running_mean"] = paddle.zeros_like(self.running_mean)
#             if prefix + "running_var" not in state_dict:
#                 state_dict[prefix + "running_var"] = paddle.ones_like(self.running_var)

#         super()._load_from_state_dict(
#             state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
#         )

#     def __repr__(self):
#         return "FrozenBatchNorm2d(num_features={}, eps={})".format(self.num_features, self.eps)

#     @classmethod
#     def convert_frozen_batchnorm(cls, module):
#         """
#         Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

#         Args:
#             module (torch.nn.Module):

#         Returns:
#             If module is BatchNorm/SyncBatchNorm, returns a new module.
#             Otherwise, in-place convert module and return it.

#         Similar to convert_sync_batchnorm in
#         https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
#         """
#         bn_module = nn.modules.batchnorm
#         bn_module = (bn_module.BatchNorm2d, bn_module.SyncBatchNorm)
#         res = module
#         if isinstance(module, bn_module):
#             res = cls(module.num_features)
#             if module.affine:
#                 res.weight.data = module.weight.data.clone().detach()
#                 res.bias.data = module.bias.data.clone().detach()
#             res.running_mean.data = module.running_mean.data
#             res.running_var.data = module.running_var.data
#             res.eps = module.eps
#         else:
#             for name, child in module.named_children():
#                 new_child = cls.convert_frozen_batchnorm(child)
#                 if new_child is not child:
#                     res.add_module(name, new_child)
#         return res
