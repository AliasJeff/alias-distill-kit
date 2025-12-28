# DistillKit 项目概览

## 项目简介

DistillKit 是一个灵活且可用于生产环境的大语言模型知识蒸馏工具包，支持在线和离线蒸馏工作流，并提供先进的 logit 压缩技术。该项目由 Arcee AI 开发，已被用于训练多个流行的开源模型，包括 Virtuoso、SuperNova Medius 和 Blitz。

## 核心功能

### 1. 知识蒸馏模式

#### 在线蒸馏 (Online Distillation)
- **实时推理**：在训练学生模型的同时，实时运行教师模型进行推理
- **无存储开销**：不需要预先存储教师模型的输出
- **适用场景**：当有足够的 VRAM 同时容纳教师和学生模型，且使用密集分布时

#### 离线蒸馏 (Offline Distillation)
- **预捕获输出**：教师模型的输出预先捕获并压缩存储
- **多学生训练**：可以从同一个教师模型训练多个学生模型
- **适用场景**：VRAM 受限、需要重用教师信号，或大规模训练时

### 2. 先进的 Logit 压缩系统

DistillKit 的核心创新在于其压缩系统，能够在保持蒸馏质量的同时大幅降低存储成本：

1. **多项式逼近**：对 logit 分布曲线进行多项式拟合
2. **误差扩散量化**：对残差进行量化以保持质量
3. **位级打包**：支持任意位宽（1-64 位）的位级打包

**压缩效果**：
- 推荐配置：~300 字节/令牌（约为未压缩分布大小的 0.15%）
- 预算配置：~114 字节/令牌，比存储 top-32 logprobs 更小且质量更好

### 3. 灵活的损失函数

支持多种可组合的损失函数，每种损失函数都有独立的权重：

#### 分布型损失
- **KL 散度** (`kl`)：标准的蒸馏损失
- **Jensen-Shannon 散度** (`jsd`)：KL 散度的对称替代方案
- **总变差距离** (`tvd`)：另一种分布距离度量

#### 排序损失
- **Hinge 损失** (`hinge`)：用于排序任务的 hinge 损失
- **Logistic 排序损失** (`logistic_ranking`)：用于排序任务的 logistic 损失

#### 隐藏状态对齐
- **隐藏状态 MSE** (`hs_mse`)：教师和学生隐藏状态之间的均方误差
- **隐藏状态余弦相似度** (`hs_cosine`)：隐藏状态之间的余弦相似度

#### 标准损失
- **交叉熵** (`cross_entropy`)：标准的语言建模损失

### 4. 稀疏与密集分布支持

- **密集分布**：包含完整词汇表的概率，更准确但内存密集
- **稀疏分布**：仅存储 top-k 令牌，作为完整密集分布的有损但高效的近似
- **自动分块**：支持长序列的内存高效处理

### 5. 跨架构蒸馏

- 支持不同架构之间的知识蒸馏
- 使用学习的线性映射投影隐藏状态
- 可与 mergekit-tokensurgeon 结合使用进行跨分词器、跨架构蒸馏

## 技术架构

### 核心技术栈

- **深度学习框架**：PyTorch (>=2.0.0)
- **模型库**：HuggingFace Transformers (>=4.50.3)
- **训练框架**：TRL (~0.25.1)
- **加速框架**：Accelerate (~1.11.0)
- **数据处理**：Datasets (>=3.5.0)
- **配置管理**：Pydantic (~2.12.4)
- **CLI 工具**：Click (~8.3.0)
- **实验跟踪**：Weights & Biases (wandb)

### 项目结构

```
distillkit/
├── main.py                    # 主入口文件，包含 CLI 命令
├── configuration.py           # 配置管理模块
├── trainer.py                 # 蒸馏训练器
├── data_processing.py         # 数据处理模块
├── signals.py                 # 信号源（在线/离线）
├── evaluation.py              # 模型评估模块
├── hsd_mapping.py             # 隐藏状态映射
├── logging_utils.py           # 日志工具
├── monkey_patch_packing.py     # 打包补丁
├── pack_logits.py             # Logit 打包
├── sample_common.py           # 采样通用功能
├── sample_logits_vllm.py      # 使用 vLLM 采样 logits
├── compression/               # 压缩模块
│   ├── compressor.py          # 压缩器主类
│   ├── config.py              # 压缩配置
│   ├── bitpack.py             # 位打包实现
│   ├── densify.py             # 密集化处理
│   ├── legacy.py              # 传统压缩实现
│   └── monotonic_logprobs.py  # 单调 logprobs 压缩
└── lossfuncs/                 # 损失函数模块
    ├── __init__.py            # 损失函数注册
    ├── common.py              # 通用损失基类
    ├── cross_entropy.py       # 交叉熵损失
    ├── kl.py                  # KL 散度损失
    ├── jsd.py                 # JSD 损失
    ├── tvd.py                 # TVD 损失
    ├── hingeloss.py           # Hinge 损失
    ├── logistic_ranking.py # Logistic 排序损失
    └── hidden_state.py        # 隐藏状态损失
```

## 核心模块详解

### 1. main.py - 主入口模块

**功能**：提供命令行接口，协调整个蒸馏流程

**主要函数**：
- `do_distill()`: 执行蒸馏训练的主函数
- `train_teacher()`: 训练教师模型
- `load_student_model()`: 加载学生模型
- `create_signal_source()`: 创建信号源（在线或离线）
- `load_tokenizer()`: 加载分词器

**CLI 命令**：
- `distill`: 执行知识蒸馏
- `train-teacher`: 训练教师模型
- `evaluate`: 评估模型性能
- `infer`: 模型推理

**关键特性**：
- 支持模型嵌入大小调整
- 支持模块冻结（通过名称或正则表达式）
- 支持 Flash Attention 2
- 支持 Functionary 打包模式

### 2. configuration.py - 配置管理模块

**功能**：使用 Pydantic 进行类型安全的配置管理

**主要配置类**：
- `DistillationRunConfig`: 蒸馏运行的主配置
- `DatasetConfiguration`: 数据集配置
- `TeacherModelConfig`: 在线教师模型配置
- `TeacherDatasetConfig`: 离线教师数据集配置
- `LossFunctionConfig`: 损失函数配置
- `EvaluationConfig`: 评估配置

**配置特性**：
- 类型验证和自动转换
- 支持 YAML 配置文件
- 配置继承和覆盖
- 默认值管理

### 3. trainer.py - 蒸馏训练器

**功能**：继承自 TRL 的 SFTTrainer，实现知识蒸馏训练逻辑

**核心类**：`DistillationTrainer`

**主要方法**：
- `compute_loss()`: 计算蒸馏损失
- `total_distillation_loss()`: 组合多个损失函数

**关键特性**：
- 支持多个损失函数的加权组合
- 自动处理稀疏和密集分布
- 支持隐藏状态对齐
- 集成信号源（在线/离线）

### 4. data_processing.py - 数据处理模块

**功能**：加载和预处理训练数据

**主要函数**：
- `load_data()`: 加载训练和验证数据集
- `_load_dataset()`: 内部数据集加载逻辑
- `_format_row()`: 行格式化函数
- `gpt_format()`: GPT 格式转换
- `leet10k_format()`: Leet10k 格式转换

**支持的数据源**：
- HuggingFace Hub 数据集
- 本地数据集
- 预打包数据集（跳过 TRL 打包）

**数据处理特性**：
- 自动格式化（支持 messages、conversations 等格式）
- 数据集缓存（基于哈希）
- 支持预打包数据集
- 可配置的采样和打乱

### 5. signals.py - 信号源模块

**功能**：定义教师模型信号的抽象接口和实现

**核心类**：
- `SignalSource`: 抽象基类
- `OnlineSignalSource`: 在线信号源（实时推理）
- `OfflineSignalSource`: 离线信号源（从压缩数据加载）

**信号类型**：
- `DenseSignal`: 密集分布信号（完整 logits）
- `SparseSignal`: 稀疏分布信号（top-k logits）

**关键特性**：
- 统一的信号接口
- 自动处理稀疏/密集转换
- 支持隐藏状态提取
- 温度参数管理

### 6. compression/ - 压缩模块

#### compressor.py - 压缩器主类

**功能**：提供 logit 压缩和解压缩的统一接口

**核心类**：`LogprobCompressor`

**主要方法**：
- `compress()`: 压缩完整 logits
- `compress_from_sparse()`: 从稀疏表示压缩
- `decompress_to_sparse()`: 解压缩为稀疏表示

**支持的格式**：
- 新格式：使用多项式逼近和位打包
- 传统格式：完全基于多项式的压缩

#### config.py - 压缩配置

**功能**：定义压缩算法的配置参数

**主要配置类**：
- `DistributionQuantizationConfig`: 分布量化配置
- `LegacyLogitCompressionConfig`: 传统压缩配置
- `QuantizationBin`: 量化桶配置

**配置参数**：
- `d`: 词汇表大小
- `k`: 总 logprobs 数量
- `exact_k`: 精确存储的 logprobs 数量
- `polynomial_terms`: 多项式项（整数或特殊项如 "sqrt"）
- `residual_bins`: 残差量化桶
- `delta_encoding`: 是否使用增量编码
- `error_diffusion`: 是否使用误差扩散

#### bitpack.py - 位打包

**功能**：实现高效的位级打包和解包

**关键特性**：
- 支持任意位宽（1-64 位）
- 高效的字节级打包
- 支持不同数据类型的索引

#### monotonic_logprobs.py - 单调 Logprobs 压缩

**功能**：利用 logprobs 的单调性进行压缩

**算法**：
- 排序 logprobs
- 多项式拟合
- 残差量化
- 位打包

### 7. lossfuncs/ - 损失函数模块

**功能**：实现各种知识蒸馏损失函数

**损失函数基类**：`LossFunctionBase`

**实现的损失函数**：

1. **KLDLoss** (`kl.py`): Kullback-Leibler 散度
   - 支持稀疏和密集模式
   - 可配置温度参数
   - 支持缺失概率处理

2. **JSDLoss** (`jsd.py`): Jensen-Shannon 散度
   - 对称的分布距离度量
   - 数值稳定性更好

3. **TVDLoss** (`tvd.py`): 总变差距离
   - L1 范数的分布距离

4. **HingeLoss** (`hingeloss.py`): Hinge 排序损失
   - 用于学习排序关系

5. **LogisticRankingLoss** (`logistic_ranking.py`): Logistic 排序损失
   - 平滑的排序损失

6. **HiddenStateMSELoss** (`hidden_state.py`): 隐藏状态 MSE
   - 对齐教师和学生的隐藏状态

7. **HiddenStateCosineLoss** (`hidden_state.py`): 隐藏状态余弦损失
   - 基于余弦相似度的隐藏状态对齐

8. **CrossEntropyLoss** (`cross_entropy.py`): 标准交叉熵
   - 语言建模的基础损失

**损失函数特性**：
- 统一的接口
- 支持稀疏和密集分布
- 可配置的温度和权重
- 自动处理缺失概率

### 8. evaluation.py - 评估模块

**功能**：评估蒸馏后模型的性能

**评估指标**：
- **PPL (Perplexity)**: 困惑度，衡量语言建模质量
- **BLEU**: 机器翻译质量指标
- **ROUGE**: 文本摘要质量指标（ROUGE-1, ROUGE-2, ROUGE-L）
- **F1 Score**: 分类任务的 F1 分数

**主要函数**：
- `evaluate_all_models()`: 评估多个模型
- `calculate_ppl()`: 计算困惑度
- `calculate_bleu()`: 计算 BLEU 分数
- `calculate_rouge()`: 计算 ROUGE 分数
- `calculate_f1()`: 计算 F1 分数
- `generate_texts()`: 生成文本用于评估

### 9. hsd_mapping.py - 隐藏状态映射

**功能**：处理跨架构蒸馏时的隐藏状态对齐

**核心类**：`HiddenStateMapping`

**主要功能**：
- 定义教师和学生层之间的映射关系
- 创建线性投影层（当隐藏状态大小不匹配时）
- 支持不同的初始化策略（Xavier、Kaiming、Identity、Zero）

**初始化策略**：
- `xavier`: Xavier 均匀初始化
- `kaiming`: Kaiming 均匀初始化
- `identity`: 截断单位矩阵初始化
- `zero`: 零初始化

### 10. 其他工具模块

#### logging_utils.py
- 文件日志记录
- 训练过程回调

#### monkey_patch_packing.py
- 为特定模型打补丁以支持打包

#### pack_logits.py
- Logit 打包相关功能

#### sample_logits_vllm.py
- 使用 vLLM 进行高效的 logit 采样
- 用于创建离线蒸馏数据集

#### sample_common.py
- 采样通用功能

## 工作流程

### 离线蒸馏流程

1. **捕获教师输出**（可选）：
   ```bash
   python -m distillkit.sample_logits_vllm \
     --model teacher-model \
     --dataset dataset-name \
     --output output-path \
     --compression-config config.yaml
   ```

2. **配置蒸馏**：
   - 创建 YAML 配置文件
   - 指定教师数据集路径
   - 配置压缩参数（必须与捕获时一致）
   - 设置损失函数和权重

3. **执行蒸馏**：
   ```bash
   distill config.yaml
   ```

4. **评估模型**：
   ```bash
   evaluate config.yaml
   ```

### 在线蒸馏流程

1. **配置蒸馏**：
   - 创建 YAML 配置文件
   - 指定教师模型路径
   - 配置损失函数

2. **执行蒸馏**：
   ```bash
   distill config.yaml
   ```

3. **评估模型**：
   ```bash
   evaluate config.yaml
   ```

## 技术亮点

### 1. 高效的压缩算法

- **多项式逼近**：利用 logit 分布的平滑性
- **自适应量化**：根据重要性分配不同精度
- **位级打包**：最大化存储效率

### 2. 灵活的架构设计

- **插件式损失函数**：易于扩展新的损失函数
- **统一的信号接口**：在线和离线模式无缝切换
- **类型安全的配置**：使用 Pydantic 确保配置正确性

### 3. 生产级特性

- **内存优化**：支持长序列分块处理
- **分布式训练**：集成 Accelerate 和 DeepSpeed
- **实验跟踪**：集成 Weights & Biases
- **错误处理**：完善的异常处理和日志记录

### 4. 跨架构支持

- **隐藏状态投影**：支持不同架构之间的蒸馏
- **分词器适配**：可与 tokensurgeon 结合使用

## 使用场景

1. **模型压缩**：将大模型的知识转移到小模型
2. **多学生训练**：从一个教师训练多个学生模型
3. **资源受限环境**：在 VRAM 受限时使用离线蒸馏
4. **大规模训练**：预捕获教师输出，避免重复推理
5. **跨架构迁移**：在不同架构之间转移知识

## 性能优化建议

### 内存优化
- 使用 `sparse_chunk_length` 处理长序列
- 使用 DeepSpeed ZeRO Stage 1 或 2
- 启用梯度检查点
- 使用 8-bit 优化器

### 计算优化
- 启用 Flash Attention 2
- 使用 bfloat16 而不是 float32
- 减少批次大小，增加梯度累积
- 使用预打包数据集

### 存储优化
- 选择合适的压缩配置
- 使用推荐的压缩参数
- 考虑使用预算配置以节省空间

## 依赖关系

### 核心依赖
- PyTorch >= 2.0.0
- Transformers >= 4.50.3
- TRL ~ 0.25.1
- Accelerate ~ 1.11.0
- Datasets >= 3.5.0
- Pydantic ~ 2.12.4
- Click ~ 8.3.0

### 可选依赖
- **vLLM** (>=0.12.0): 用于高效的 logit 采样
- **Weights & Biases**: 用于实验跟踪
- **评估工具**: sacrebleu, rouge-score, nltk

## 扩展性

### 添加新的损失函数

1. 在 `lossfuncs/` 目录创建新文件
2. 继承 `LossFunctionBase` 类
3. 实现 `__call__()` 方法
4. 在 `lossfuncs/__init__.py` 中注册

### 添加新的压缩算法

1. 在 `compression/` 目录实现新算法
2. 在 `compressor.py` 中集成
3. 更新配置类以支持新参数

### 添加新的数据格式

1. 在 `data_processing.py` 中添加格式化函数
2. 在 `FORMAT_FUNCTIONS` 字典中注册

## 最佳实践

1. **配置管理**：
   - 使用版本控制管理配置文件
   - 为不同实验创建配置副本
   - 记录配置变更历史

2. **实验跟踪**：
   - 使用 Weights & Biases 跟踪实验
   - 记录所有超参数
   - 保存模型检查点

3. **数据准备**：
   - 预处理和缓存数据集
   - 使用预打包数据集以提高效率
   - 验证数据格式正确性

4. **模型训练**：
   - 从小规模实验开始
   - 逐步增加数据量和模型大小
   - 监控训练指标和内存使用

5. **评估验证**：
   - 使用多个评估指标
   - 在验证集上定期评估
   - 保存评估结果用于比较

## 总结

DistillKit 是一个功能强大、设计灵活的知识蒸馏工具包，通过创新的压缩技术和灵活的架构设计，使得大规模离线蒸馏成为可能。无论是用于研究还是生产环境，DistillKit 都提供了必要的工具和灵活性来实现高效的知识蒸馏。

