import torch.nn as nn



class Register(dict):
    """实现一个自动注册类(继承自python字典类), 可以作为装饰器自动注册nn.Module"""
    def __init__(self, *args, **kwargs):
        super(Register, self).__init__(*args, **kwargs)
        # dict类的self本身就是字典, 无需再新建一个self._dict = {}成员
        # 字典里存放所有被注册的类对象: key=类名, value=类本身(未实例化)

    def add_item(self, key, value):
        """注册的核心方法, 将key(类名), value(未实例化的类对象)存入字典中"""
        if not callable(value):
            raise Exception(f"Error:{value} must be callable!")
        if key in self:
            print(f"\033[31mWarning:\033[0m {value.__name__} already exists and will be overwritten!")
        self[key] = value
        return value

    def register(self, target):
        """注册的调用方法"""
        # 传入的target是函数或类
        if callable(target):    
            return self.add_item(target.__name__, target)
        # 如果传入的是字符串，返回一个装饰器函数，用这个字符串作为注册名
        else:                   
            return lambda x : self.add_item(target, x)

    def build(self, key, *args, **kwargs):
        """根据key(类名称)实例化对应的nn.Module"""
        if key not in self:
            raise KeyError(f"{key} is not registered!")
        cls = self[key]
        # 类必须是一个nn.Module类或其子类, 否则报错
        if not issubclass(cls, nn.Module):
            raise TypeError(f"{key} is not a subclass of nn.Module!")
        return cls(*args, **kwargs)

    def build_from_cfg(self, cfg: dict):
        """根据配置字典递归实例化模块(因为有时候存在嵌套的情况，一个模块的初始化参数包含另一个模块)
            cfg 示例: 
                {"type": "MyLinear", "in_features": 10, "out_features": 5}
        """
        if "type" not in cfg:
            raise KeyError("cfg must contain the key 'type'")
        # cfg.pop("type") 会修改原字典, 如果外部还有引用, 会产生错误, 因此用copy避免外部再次调用时已被更改
        cfg = cfg.copy()
        module_type = cfg.pop("type")

        # 递归的实例化子模块
        for k, v in cfg.items():
            # 子模块
            if isinstance(v, dict) and "type" in v:   
                cfg[k] = self.build_from_cfg(v)
            # 模块列表
            elif isinstance(v, list):                 
                cfg[k] = [self.build_from_cfg(i) if isinstance(i, dict) and "type" in i else i for i in v]

        return self.build(module_type, **cfg)



# 关键点：模块级单例注册器, 这样其他文件中直接import就行, 无需初始化
MODELS = Register()