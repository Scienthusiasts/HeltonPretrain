from .loss import CELoss, MultiClassBCELoss, SmoothL1Loss, KLDivLoss, JSDivLoss
from .distill_loss import TeacherDistillLoss1D, TeacherDistillLoss2D

__all__ = [
    "CELoss", "MultiClassBCELoss", "SmoothL1Loss", "KLDivLoss", "JSDivLoss",
    "TeacherDistillLoss1D", "TeacherDistillLoss2D"
    ]