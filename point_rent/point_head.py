import paddle
from paddle import nn
from paddle.nn import functional as F

from utils.shape_spec import ShapeSpec
from utils.register import Registry

_CURRENT_STORAGE_STACK = []
def get_event_storage():
    """
    Returns:
        The :class:`EventStorage` object that's currently being used.
        Throws an error if no :class:`EventStorage` is currently enabled.
    """
    assert len(
        _CURRENT_STORAGE_STACK
    ), "get_event_storage() has to be called inside a 'with EventStorage(...)' context!"
    return _CURRENT_STORAGE_STACK[-1]

POINT_HEAD_REGISTRY = Registry("POINT_HEAD")
POINT_HEAD_REGISTRY.__doc__ = """
Registry for point heads, which makes prediction for a given set of per-point features.

The registered object will be called with `obj(cfg, input_shape)`.
"""


def roi_mask_point_loss(mask_logits, instances, point_labels):
    """
    Compute the point-based loss for instance segmentation mask predictions
    given point-wise mask prediction and its corresponding point-wise labels.
    Args:
        mask_logits (Tensor): A tensor of shape (R, C, P) or (R, 1, P) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images, C is the
            number of foreground classes, and P is the number of points sampled for each mask.
            The values are logits.
        instances (list[Instances]): A list of N Instances, where N is the number of images
            in the batch. These instances are in 1:1 correspondence with the `mask_logits`. So, i_th
            elememt of the list contains R_i objects and R_1 + ... + R_N is equal to R.
            The ground-truth labels (class, box, mask, ...) associated with each instance are stored
            in fields.
        point_labels (Tensor): A tensor of shape (R, P), where R is the total number of
            predicted masks and P is the number of points for each mask.
            Labels with value of -1 will be ignored.
    Returns:
        point_loss (Tensor): A scalar tensor containing the loss.
    """
    with paddle.no_grad():
        cls_agnostic_mask = mask_logits.shape[1] == 1
        total_num_masks = mask_logits.shape[0]

        gt_classes = []
        for instances_per_image in instances:
            if len(instances_per_image) == 0:
                continue

            if not cls_agnostic_mask:
                gt_classes_per_image = instances_per_image.gt_classes.astype('int64')
                gt_classes.append(gt_classes_per_image)

    gt_mask_logits = point_labels
    point_ignores = point_labels == -1
    if gt_mask_logits.shape[0] == 0:
        return mask_logits.sum() * 0

    assert gt_mask_logits.numel() > 0, gt_mask_logits.shape

    if cls_agnostic_mask:
        mask_logits = mask_logits[:, 0]
    else:
        indices = paddle.arange(total_num_masks)
        gt_classes = paddle.concat(gt_classes, axis=0)
        mask_logits = mask_logits[indices, gt_classes]

    # Log the training accuracy (using gt classes and 0.0 threshold for the logits)
    mask_accurate = (mask_logits > 0.0) == gt_mask_logits.astype('uint8')
    mask_accurate = mask_accurate[~point_ignores]
    mask_accuracy = mask_accurate.nonzero().shape[0] / max(mask_accurate.numel(), 1.0)
    get_event_storage().put_scalar("point/accuracy", mask_accuracy)

    point_loss = F.binary_cross_entropy_with_logits(
        mask_logits, gt_mask_logits.astype('float32'), weight=~point_ignores, reduction="mean"
    )
    return point_loss

@POINT_HEAD_REGISTRY.register()
class StandardPointHead(nn.Layer):
    """
    A point head multi-layer perceptron which we model with conv1d layers with kernel 1. The head
    takes both fine-grained and coarse prediction features as its input.
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        """
        The following attributes are parsed from config:
            fc_dim: the output dimension of each FC layers
            num_fc: the number of FC layers
            coarse_pred_each_layer: if True, coarse prediction features are concatenated to each
                layer's input
        """
        super(StandardPointHead, self).__init__()
        # fmt: off
        num_classes                 = cfg.MODEL.POINT_HEAD.NUM_CLASSES      #19
        fc_dim                      = cfg.MODEL.POINT_HEAD.FC_DIM           #256
        num_fc                      = cfg.MODEL.POINT_HEAD.NUM_FC           #3
        cls_agnostic_mask           = cfg.MODEL.POINT_HEAD.CLS_AGNOSTIC_MASK
        self.coarse_pred_each_layer = cfg.MODEL.POINT_HEAD.COARSE_PRED_EACH_LAYER
        input_channels              = input_shape.channels      #'512'
        # fmt: on

        fc_dim_in = input_channels + num_classes
        self.fc_layers = []
        w_attr_1, b_attr_1 = self._init_weights()
        for k in range(num_fc):
            fc = nn.Conv1D(fc_dim_in, fc_dim, kernel_size=1, stride=1, padding=0, bias_attr=b_attr_1)
            self.add_sublayer("fc{}".format(k + 1), fc)
            self.fc_layers.append(fc)
            fc_dim_in = fc_dim
            fc_dim_in += num_classes if self.coarse_pred_each_layer else 0

        num_mask_classes = 1 if cls_agnostic_mask else num_classes
        self.predictor = nn.Conv1D(fc_dim_in, num_mask_classes, kernel_size=1, stride=1, padding=0, weight_attr=w_attr_1)

    def _init_weights(self):
        weight_attr = paddle.ParamAttr(initializer=nn.initializer.Normal(std=0.001))
        bias_attr = paddle.ParamAttr(initializer=nn.initializer.KaimingNormal())
        return weight_attr, bias_attr


    def forward(self, fine_grained_features, coarse_features):
        x = paddle.concat((fine_grained_features, coarse_features), axis=1)
        for layer in self.fc_layers:
            x = F.relu(layer(x))
            if self.coarse_pred_each_layer:
                x = paddle.concat((x, coarse_features), axis=1)
        return self.predictor(x)


def build_point_head(cfg, input_channels):
    """
    Build a point head defined by `cfg.MODEL.POINT_HEAD.NAME`.
    """
    head_name = cfg.MODEL.POINT_HEAD.NAME
    return POINT_HEAD_REGISTRY.get(head_name)(cfg, input_channels)
