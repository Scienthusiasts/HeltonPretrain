from .timm_backbone import TIMMBackbone

# __all__的作用, 当使用from ... import *时, 只会导入 TIMMBackbone, 不会导入其他类或函数
__all__ = ["TIMMBackbone"]