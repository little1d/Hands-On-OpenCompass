# Hands-On-OpenCompass

OpenCompass 可以根据用户需要，自定义数据集和评测方式，并提供评估工具包以及丰富的数据集等。本仓库旨在介绍 OpenCompass的基础内容，帮助大家快速上手。

OpenCompass基础的数据集配置，评测配置和进阶的运行后端等都是基于写配置文件完成的，区别于传统的纯文本风格的配置文件(`json/yaml`)，
OpenCompass 基于 MMEngine，使用纯 Python风格的配置文件，详情可了解 [MMEngine-CONFIG](https://mmengine.readthedocs.io/zh-cn/latest/advanced_tutorials/config.html#python-beta)

OpenCompass 2.0 由三个部分构成
1. CompassRank：排行榜
2. CompassHub：基准测试资源库
3. CompassKit：评估工具 toolkit

如下所示，仓库整体分为三部分，推荐您按照这个顺序阅读
```bash
.
├── README.md    # 介绍 OpenCompass的前置知识以及环境的配置
├── datasets     # 数据集配置
├── models       # 模型配置
└── evaluation   # 评测任务发起

```

# 环境搭建
* 创建虚拟环境

使用 `conda` 管理 `Python` 环境

```python
conda create -n opencompass python=3.10 -y
conda activate opencompass
```

* 通过 pip 安装 OpenCompass 基础版工具包
```python
pip install -U opencompass
```

* 基于源码安装 OpenCompass
```python
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
# pip install -e ".[full]"
# pip install -e ".[vllm]"
```

> 本仓库绝大多数内容参考自 [opencompass官方文档](https://opencompass.readthedocs.io/zh-cn/latest/)，加上自己的代码实操与理解，源文档很详尽，如有时间或问题还请移步源文档





