from setuptools import setup, find_packages
import os

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname), encoding='utf-8').read()

setup(
    name="heltonx",
    version="0.1",
    author="Yan Helton",
    author_email="1041440961@qq.com",
    description="A unified deep learning framework.",
    long_description=read("README.md") if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/Scienthusiasts/heltonx",
    license="MIT",
    # 声明当前目录即 heltonx 包
    package_dir={"heltonx": "."}, 
    # 查找所有子模块 会返回 ["configs", "tools", "utils"等]
    packages=find_packages(where="."),
    include_package_data=True,  # 包含非.py文件
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13",
        "torchvision>=0.15",
        "accelerate>=0.21",
        "numpy==2.2.6",
        "opencv-python>=4.5",
        "tqdm",
        "pyyaml",
        "matplotlib",
        "Pillow",
    ],
    extras_require={
        "dev": ["black", "isort", "flake8", "pytest"],
    },
    entry_points={
        "console_scripts": [
            "heltonx-train=tools.train:main",
            "heltonx-train-acc=tools.train_accelerate:main",
            "heltonx-eval=tools.eval:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="deep-learning pytorch pretrain detection generation",
)