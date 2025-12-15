# REAP剪枝分析指南

本文档全面分析了REAP（路由器加权专家激活剪枝）方法，解答数据需求、可复现性和关键剪枝参数等问题。

## 目录
1. [REAP剪枝的数据需求](#data-requirements)
2. [开源完整度与可复现性](#reproducibility)
3. [关键剪枝参数](#key-parameters)
4. [快速入门示例](#quick-start)

---

## 1. REAP剪枝的数据需求 {#data-requirements}

### 剪枝时是否需要引入数据？

**是的，REAP剪枝需要校准数据。** REAP是一种数据驱动的剪枝方法，需要校准数据集来计算专家显著性分数。

### 为什么需要数据

REAP剪枝需要数据的原因如下：

1. **记录专家激活**：该方法需要观察在真实数据样本上进行前向传播时哪些专家被激活，以计算：
   - 专家频率（每个专家被选择的频率）
   - 专家激活范数（专家输出的幅度）
   - 路由器门控值（路由器分配的路由权重）

2. **显著性分数计算**：REAP结合路由器权重和激活范数来计算专家重要性：
   ```
   REAP分数 = 路由器权重 × 专家激活范数
   ```
   这个计算需要在校准数据上运行推理来收集统计信息。

3. **逐层统计**：该方法在校准数据集上收集每一层的统计信息，为每个MoE层独立做出明智的剪枝决策。

### 使用什么类型的数据？

本仓库支持多种校准数据集：

- **代码数据集**（推荐用于代码模型）：
  - `theblackcat102/evol-codealpaca-v1`（默认）
  - `ise-uiuc/Magicoder-Evol-Instruct-110K`
  - `m-a-p/CodeFeedback-Filtered-Instruction`

- **通用文本数据集**：
  - `allenai/c4`
  
- **数学数据集**：
  - `allenai/tulu-3-sft-personas-math`

- **创意写作数据集**：
  - `euclaise/WritingPrompts_curated`

- **组合数据集**：使用多个数据集的预记录观测值

### 数据量需求

默认配置使用：
- **每个类别1024个样本**（`samples_per_category=1024`）
- **最大序列长度**：2048个token（`model_max_length=2048`）

这些设置在以下方面取得平衡：
- 显著性分数的统计可靠性
- 计算效率
- 内存限制

---

## 2. 开源完整度与可复现性 {#reproducibility}

### REAP剪枝能否完全复现？

**可以，REAP剪枝可以完全复现**，使用本开源仓库即可。代码库提供了完整的实现，包括：

### 仓库包含的内容

#### ✅ 核心剪枝实现
- **完整的剪枝流程**（`src/reap/prune.py`）
- **用于激活记录的观察器钩子**（`src/reap/observer.py`）
- **多种剪枝方法**，包括REAP和基线方法
- **显著性分数计算**，支持多种指标

#### ✅ 支持的模型
仓库包含多个MoE架构的配置：
- Qwen3-MoE（Qwen3-30B-A3B、Qwen3-Coder-480B）
- Llama-4 MoE
- Mixtral
- DeepSeek-V2
- GLM-4.5-Air、GLM-4.6
- ERNIE-4.5
- MiniMax-M2
- Kimi-Linear

完整的模型支持请查看`src/reap/model_util.py`中的`MODEL_ATTRS`和`src/reap/observer.py`中的`OBSERVER_CONFIG_REGISTRY`。

#### ✅ 评估套件
- **LM-Eval**：用于多项选择任务
- **EvalPlus**：用于代码生成（HumanEval、MBPP）
- **LiveCodeBench**：用于最新的代码生成评估
- **数学评估**
- **WildBench**：用于智能体任务

#### ✅ 预训练的剪枝模型
在HuggingFace上可用：[Cerebras REAP合集](https://huggingface.co/collections/cerebras/cerebras-reap)

### 可复现性清单

要完全复现REAP剪枝实验：

- [x] **源代码**：完整实现已提供
- [x] **模型支持**：开箱即用支持多种架构
- [x] **校准数据集**：提供多个选项
- [x] **评估工具**：包含综合评估套件
- [x] **超参数**：提供默认配置
- [x] **文档**：包含示例的README
- [x] **Docker支持**：提供容器化环境
- [x] **脚本**：提供可直接使用的bash脚本
- [x] **预训练模型**：可下载的剪枝检查点

### 复现步骤

1. **安装**：
   ```bash
   bash scripts/build.sh  # 或使用Docker
   ```

2. **运行剪枝**：
   ```bash
   bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1
   ```

3. **结果**：剪枝模型保存到`artifacts/`目录，包含完整的配置YAML文件

---

## 3. 关键剪枝参数 {#key-parameters}

### REAP剪枝的关键参数

#### 3.1 剪枝方法（`--prune-method`）

**最重要的参数**：确定专家选择的显著性标准。

可用方法：
- **`reap`**（推荐）：路由器加权的专家激活范数（均值）
  - 公式：`mean(router_weight × ||expert_activation||_2)`
  - 根据论文，这是性能最好的方法

- **`reap_l2`**：REAP的L2范数变体
  - 公式：`sqrt(sum((router_weight × ||expert_activation||_2)^2))`

- **`frequency`**：专家选择频率（基线）
  - 简单计数每个专家被选择的次数
  - 不考虑激活幅度

- **`ean_sum`**：专家激活范数求和
  - 专家输出的L2范数之和
  - 不考虑路由器权重

- **`ean_mean`**：专家激活范数均值
  - 激活的平均L2范数

- **`weighted_ean_sum`**：路由器加权的EAN求和
  - 公式：`sum(router_weight × ||expert_activation||_2)`

- **`weighted_ean_sum_l2`**：加权EAN的L2范数

- **`ean_ca`**：基于特征激活的EAN

- **`max_activations`**：最大激活值（用于超级专家检测）

**建议**：使用`reap`（默认）以获得最佳性能。

#### 3.2 压缩比（`--compression-ratio`）

**定义**：从每层**移除**的专家比例。

- **范围**：0.0到1.0
- **常用值**： 
  - `0.25` = 移除25%的专家（保留75%）
  - `0.5` = 移除50%的专家（保留50%）- **论文推荐**
  - `0.75` = 移除75%的专家（保留25%）

**权衡**：更高的压缩率→更小的模型尺寸，但可能会损失精度

**论文发现**：REAP在0.5压缩比下对代码生成任务实现了近乎无损的压缩。

#### 3.3 校准数据集（`--dataset-name`）

**目的**：用于计算专家统计信息的数据。

**重要考虑因素**：
- 选择与目标任务领域对齐的数据集
- 代码模型→代码数据集
- 数学模型→数学数据集
- 通用模型→多样化数据集或`combined`

**默认**：`theblackcat102/evol-codealpaca-v1`（专注于代码）

#### 3.4 样本大小（`--samples-per-category`）

**定义**：从校准数据集处理的样本数量。

- **默认**：`1024`
- **范围**：100-2048+，取决于数据集大小和内存
- **权衡**： 
  - 更多样本→更可靠的统计数据→更好的剪枝决策
  - 更多样本→更长的校准时间→更高的内存使用

**建议**：1024个样本为大多数模型提供了良好的平衡。

#### 3.5 随机种子（`--seed`）

**目的**：确保结果的可复现性。

- **默认**：`42`
- **影响**：影响数据集采样和任何随机操作
- **建议**：使用相同的种子进行公平比较

#### 3.6 超级专家/异常值保护

> **注意**：参数名使用"perserve"（"preserve"的拼写错误）以匹配实际代码库实现。

**超级专家保护**：
- `--perserve-super-experts` / `--singleton_super_experts true`
- 防止剪枝"超级专家"（具有持续高激活的专家）
- 排除最后25%的层进行超级专家检测

**异常值专家保护**：
- `--perserve-outliers` / `--singleton_outlier_experts true`
- 在所有层保护异常值专家
- 有助于保持模型稳定性

**互斥性**：一次只能启用一个。

**论文参考**：参见[揭示MoE LLM中的超级专家](https://arxiv.org/abs/2507.23279)

#### 3.7 路由器权重重归一化（`--renormalize-router-weights`）

**目的**：将top-k路由器权重重归一化，使其总和为1。

- **默认**：`false`
- **何时启用**：如果`model.config.norm_topk_prob == True`
- **效果**：确保在计算显著性之前路由器权重被正确归一化

#### 3.8 观测记录模式（`--record-pruning-metrics-only`）

**目的**：优化校准期间的内存使用。

- **默认**：`true`（在pruning-cli.sh中）
- **启用时**：仅记录剪枝所需的指标（不包括合并指标）
- **内存节省**：显著 - 跳过记录相似度矩阵和特征激活

### 参数优先级排序

对于大多数使用场景，按以下顺序调整这些参数：

1. **`--prune-method`**：选择正确的显著性标准
2. **`--compression-ratio`**：设置目标模型大小
3. **`--dataset-name`**：使校准数据与目标领域匹配
4. **`--samples-per-category`**：平衡精度与效率
5. **`--perserve-super-experts` / `--perserve-outliers`**：如果出现稳定性问题则启用（注意："perserve"拼写匹配代码库）

### 次要参数

这些通常使用默认值即可：

- **`--model-max-length`**：2048个token（默认）
- **`--distance-measure`**："cosine"（默认）
- **`--output-file-name`**：根据设置自动生成
- **`--overwrite-observations`**：false（重用缓存的激活）
- **`--overwrite-pruned-model`**：false（不覆盖现有的剪枝模型）

---

## 4. 快速入门示例 {#quick-start}

### 示例1：基本REAP剪枝（50%压缩）

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false
```

参数说明：
- `0`：GPU设备
- `Qwen/Qwen3-30B-A3B`：模型名称
- `reap`：剪枝方法（REAP）
- `42`：随机种子
- `0.5`：压缩比（50%剪枝）
- `theblackcat102/evol-codealpaca-v1`：校准数据集
- `true true true false false`：启用lm_eval、evalplus、livecodebench；禁用math、wildbench

### 示例2：保守剪枝（25%压缩）

```bash
bash experiments/pruning-cli.sh 0,1 Qwen/Qwen3-30B-A3B reap 42 0.25 theblackcat102/evol-codealpaca-v1 true true false false false
```

变化：
- `0,1`：使用2个GPU
- `0.25`：仅剪枝25%的专家（更保守）

### 示例3：基于频率的基线

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B frequency 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false
```

变化：
- `frequency`：使用简单的频率计数而不是REAP

### 示例4：启用超级专家保护

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false true false
```

变化：
- `true false`：启用超级专家保护（第12个参数）

### 示例5：Python API使用

```python
from reap.prune import main
import sys

# 设置命令行参数
sys.argv = [
    "prune.py",
    "--model-name", "Qwen/Qwen3-30B-A3B",
    "--dataset-name", "theblackcat102/evol-codealpaca-v1",
    "--compression-ratio", "0.5",
    "--prune-method", "reap",
    "--seed", "42",
    "--samples-per-category", "1024",
    "--record-pruning-metrics-only", "true",
    "--do-eval", "true",
]

# 运行剪枝
main()
```

---

## 输出和结果

运行剪枝后，您将找到：

1. **剪枝模型**：保存到`artifacts/{model_name}/{dataset_name}/pruned_models/{method}-seed_{seed}-{ratio}/`
   - 包含模型权重、配置、分词器
   - 包含所有实验参数的`reap_args.yaml`

2. **评估结果**：保存到`{pruned_model_dir}/eval/`
   - LM-Eval分数
   - 代码生成指标
   - 基准测试结果

3. **观测数据**：缓存在`artifacts/{model_name}/{dataset_name}/all/observations_*.pt`
   - 用于相同数据集的后续运行
   - 包含专家统计信息

---

## 获得最佳结果的技巧

1. **从默认设置开始**：默认的REAP配置对大多数模型效果很好
2. **匹配校准到目标任务**：代码模型使用代码数据，数学模型使用数学数据
3. **监控评估指标**：检查剪枝是否降低了目标任务的性能
4. **尝试不同的压缩比**：0.25、0.5、0.75以找到最佳权衡
5. **启用超级专家保护**：如果观察到不稳定或性能下降
6. **重用观测数据**：设置`--overwrite-observations false`以节省重复实验的时间

---

## 其他资源

- **论文**：[REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/abs/2510.13999)
- **HuggingFace模型**：[Cerebras REAP合集](https://huggingface.co/collections/cerebras/cerebras-reap)
- **源代码**：`src/reap/prune.py`、`src/reap/observer.py`、`src/reap/args.py`
- **问题反馈**：在GitHub Issues上报告错误或提问

---

## 总结

- ✅ **需要数据**：REAP需要校准数据来计算专家显著性分数
- ✅ **完全可复现**：完整的开源实现，包含所有必要组件
- ✅ **关键参数**：重点关注`prune-method`、`compression-ratio`和`dataset-name`
- ✅ **易于使用**：提供bash脚本和Python API，便于快速实验
- ✅ **支持良好**：包含多个模型、数据集和评估基准

如有问题或想要贡献，请在GitHub上提交issue或pull request。
