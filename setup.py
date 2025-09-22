from setuptools import setup, find_packages

setup(
    name="heltondl",  # 包名，可随意改
    version="0.1",
    packages=find_packages(include=['models', 'models.*']),
    py_modules=['register'],  # 根目录的单文件模块
    install_requires=[
        "torch>=1.13.1",
        "torchvision>=0.14.1",
        "timm>=0.9.0",
    ],
    python_requires=">=3.8",
    description="HeltonDL Project",
)
