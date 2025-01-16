### 准备模型
OpenCompass 中支持以下三种方式的模型评测 

#### 基于 HuggingFace 的模型
直接从 Huggingface 的 AutoModel.from_pretrained 和 AutoModelForCausalLM.from_pretrained 接口构建评测模型。如果需要评测的模型符合 HuggingFace 模型通常的生成接口， 则不需要编写代码，直接在配置文件中指定相关配置即可。
对于 HuggingFace的封装，源代码可以参考：[opencompass/models/huggingface.py](https://github.com/open-compass/opencompass/blob/main/opencompass/models/huggingface.py)

这里用 `baichuan2_13b_chat` 作为理解

[源代码](https://github.com/open-compass/opencompass/blob/main/configs/models/baichuan/hf_baichuan2_13b_chat.py)

![](../images/figure2.png)

按照 Python风格的配置文件，对于模型的配置可以大致分为三部分：
1. type：使用的模型类型，具体封装在[opencompass/opencompass/models](https://github.com/open-compass/opencompass/tree/main/opencompass/models)之下，例如上图展示的 `type=VLLM`,
VLLM类封装在对应的 `opencompass/opencompass/models/vllm.py`下
2. 初始化参数
    与模型本身直接相关的，用于实例化模型，例如上面的 path--指定模型路径,tokenizer_path, tokenizer_kwargs等
3. 其他参数
    用于模型推理及其他个性化的设置，与模型本身并无相关性，例如 abbr用于结果展示时模型的简称，batch

评测开始时，OpenCompass首先会使用配置文件的 type和初始化参数实例化模型，再用其他参数进行推理和总结。例如上图中，我们首先执行的是实例化过程
```python
model = HuggingFaceCausalLM(
    type=HuggingFaceCausalLM,
    path='baichuan-inc/Baichuan2-13B-Chat',
    tokenizer_path='baichuan-inc/Baichuan2-13B-Chat',
    tokenizer_kwargs=dict(padding_side='left', truncation_side='left'),
    tokenizer_kwargs=dict(
        padding_side='left',
        truncation_side='left',
        trust_remote_code=True,
        use_fast=False,
    )
    .......
)
```

具体该使用哪些参数，需要去对应 type模型的封装文件中查阅（模型太多，可能 opencompass也没法提供详尽的 API文档），例如`HuggingFaceCausalLM`，我们可以找到以下参数

![](../images/figure4.png)
#### 基于 API 的模型



#### 自定义模型