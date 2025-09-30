from .loss import CELoss, MultiClassBCELoss, SmoothL1Loss
from .distill_loss import TeacherDistillLoss1D, TeacherDistillLoss2D

__all__ = [
    "CELoss", "MultiClassBCELoss", "SmoothL1Loss",
    "TeacherDistillLoss1D", "TeacherDistillLoss2D"
    ]