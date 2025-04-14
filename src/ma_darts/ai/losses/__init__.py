__all__ = []

xst_weight = 400.0
cls_weight = 2000.0
pos_weight = 0.5

__all__.append("xst_weight")
__all__.append("cls_weight")
__all__.append("pos_weight")

from .existence_loss import ExistenceLoss
from .classes_loss import ClassesLoss
from .positions_loss import PositionsLoss
from .diou_loss import DIoULoss
from .yolo_loss import YOLOv8Loss

__all__.append("ExistenceLoss")
__all__.append("ClassesLoss")
__all__.append("PositionsLoss")
__all__.append("DIoULoss")
__all__.append("YOLOv8Loss")
