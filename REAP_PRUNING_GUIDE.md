# REAP Pruning Analysis Guide

This document provides a comprehensive analysis of the REAP (Router-weighted Expert Activation Pruning) methodology, addressing data requirements, reproducibility, and key parameters for pruning Mixture-of-Experts (MoE) language models.

## Table of Contents
1. [Data Requirements for REAP Pruning](#data-requirements)
2. [Open-Source Completeness and Reproducibility](#reproducibility)
3. [Key Pruning Parameters](#key-parameters)
4. [Quick Start Examples](#quick-start)

---

## 1. Data Requirements for REAP Pruning {#data-requirements}

### Is Data Required for REAP Pruning?

**Yes, calibration data is required for REAP pruning.** REAP is a data-driven pruning method that requires a calibration dataset to compute expert saliency scores.

### Why Data is Needed

REAP pruning requires data for the following reasons:

1. **Expert Activation Recording**: The method needs to observe which experts are activated during forward passes on real data samples to calculate:
   - Expert frequency (how often each expert is selected)
   - Expert activation norms (the magnitude of expert outputs)
   - Router gate values (the routing weights assigned by the router)

2. **Saliency Score Calculation**: REAP combines router weights and activation norms to compute expert importance:
   ```
   REAP Score = Router Weight × Expert Activation Norm
   ```
   This calculation requires running inference on calibration data to collect statistics.

3. **Layer-wise Statistics**: The method collects per-layer statistics across the calibration dataset to make informed pruning decisions for each MoE layer independently.

### What Type of Data is Used?

The repository supports various calibration datasets:

- **Code Datasets** (recommended for code models):
  - `theblackcat102/evol-codealpaca-v1` (default)
  - `ise-uiuc/Magicoder-Evol-Instruct-110K`
  - `m-a-p/CodeFeedback-Filtered-Instruction`

- **General Text Datasets**:
  - `allenai/c4`
  
- **Math Datasets**:
  - `allenai/tulu-3-sft-personas-math`

- **Creative Writing Datasets**:
  - `euclaise/WritingPrompts_curated`

- **Combined**: Use pre-recorded observations from multiple datasets

### Data Volume Requirements

Default configuration uses:
- **1024 samples per category** (`samples_per_category=1024`)
- **Maximum sequence length**: 2048 tokens (`model_max_length=2048`)

These settings balance between:
- Statistical reliability of saliency scores
- Computational efficiency
- Memory constraints

---

## 2. Open-Source Completeness and Reproducibility {#reproducibility}

### Can REAP Pruning Be Fully Reproduced?

**Yes, REAP pruning can be fully reproduced** using this open-source repository. The codebase provides complete implementations for:

### What's Included in the Repository

#### ✅ Core Pruning Implementation
- **Complete pruning pipeline** (`src/reap/prune.py`)
- **Observer hooks for activation recording** (`src/reap/observer.py`)
- **Multiple pruning methods** including REAP and baselines
- **Saliency score calculations** with various metrics

#### ✅ Supported Models
The repository includes configuration for multiple MoE architectures:
- Qwen3-MoE (Qwen3-30B-A3B, Qwen3-Coder-480B)
- Llama-4 MoE
- Mixtral
- DeepSeek-V2
- GLM-4.5-Air, GLM-4.6
- ERNIE-4.5
- MiniMax-M2
- Kimi-Linear

See `MODEL_ATTRS` in `src/reap/model_util.py` and `OBSERVER_CONFIG_REGISTRY` in `src/reap/observer.py` for full model support.

#### ✅ Evaluation Suite
- **LM-Eval** for multiple choice tasks
- **EvalPlus** for coding (HumanEval, MBPP)
- **LiveCodeBench** for up-to-date code generation
- **Math evaluations**
- **WildBench** for agentic tasks

#### ✅ Pre-trained Pruned Models
Available on HuggingFace: [Cerebras REAP Collection](https://huggingface.co/collections/cerebras/cerebras-reap)

### Reproducibility Checklist

To fully reproduce REAP pruning experiments:

- [x] **Source code**: Complete implementation available
- [x] **Model support**: Multiple architectures supported out-of-the-box
- [x] **Calibration datasets**: Multiple options provided
- [x] **Evaluation harness**: Comprehensive evaluation suite included
- [x] **Hyperparameters**: Default configurations provided
- [x] **Documentation**: README with examples
- [x] **Docker support**: Containerized environment available
- [x] **Scripts**: Ready-to-use bash scripts for experiments
- [x] **Pre-trained models**: Pruned checkpoints available for download

### Steps to Reproduce

1. **Installation**:
   ```bash
   bash scripts/build.sh  # or use Docker
   ```

2. **Run Pruning**:
   ```bash
   bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1
   ```

3. **Results**: Pruned models saved to `artifacts/` directory with full configuration YAML

---

## 3. Key Pruning Parameters {#key-parameters}

### Critical Parameters for REAP Pruning

#### 3.1 Pruning Method (`--prune-method`)

**Most Important Parameter**: Determines the saliency criterion for expert selection.

Available methods:
- **`reap`** (RECOMMENDED): Router-weighted Expert Activation Norm (mean)
  - Formula: `mean(router_weight × ||expert_activation||_2)`
  - Best performing method according to paper

- **`reap_l2`**: L2 norm variant of REAP
  - Formula: `sqrt(sum((router_weight × ||expert_activation||_2)^2))`

- **`frequency`**: Expert selection frequency (baseline)
  - Simple count of how often each expert is selected
  - No activation magnitude consideration

- **`ean_sum`**: Expert Activation Norm sum
  - Sum of L2 norms of expert outputs
  - No router weight consideration

- **`ean_mean`**: Expert Activation Norm mean
  - Average L2 norm across activations

- **`weighted_ean_sum`**: Router-weighted EAN sum
  - Formula: `sum(router_weight × ||expert_activation||_2)`

- **`weighted_ean_sum_l2`**: L2 norm of weighted EAN

- **`ean_ca`**: Characteristic Activation based EAN

- **`max_activations`**: Maximum activation values (for super expert detection)

**Recommendation**: Use `reap` (default) for best performance.

#### 3.2 Compression Ratio (`--compression-ratio`)

**Definition**: Fraction of experts to **remove** from each layer.

- **Range**: 0.0 to 1.0
- **Common values**: 
  - `0.25` = Remove 25% of experts (75% remaining)
  - `0.5` = Remove 50% of experts (50% remaining) - **Recommended in paper**
  - `0.75` = Remove 75% of experts (25% remaining)

**Trade-off**: Higher compression → smaller model size but potential accuracy loss

**Paper findings**: REAP achieves near-lossless compression at 0.5 compression ratio for code generation tasks.

#### 3.3 Calibration Dataset (`--dataset-name`)

**Purpose**: Data used to compute expert statistics.

**Important Considerations**:
- Choose dataset aligned with target task domain
- Code models → code datasets
- Math models → math datasets
- General models → diverse datasets or `combined`

**Default**: `theblackcat102/evol-codealpaca-v1` (code-focused)

#### 3.4 Sample Size (`--samples-per-category`)

**Definition**: Number of samples to process from calibration dataset.

- **Default**: `1024`
- **Range**: 100-2048+ depending on dataset size and memory
- **Trade-off**: 
  - More samples → more reliable statistics → better pruning decisions
  - More samples → longer calibration time → higher memory usage

**Recommendation**: 1024 samples provides good balance for most models.

#### 3.5 Random Seed (`--seed`)

**Purpose**: Ensures reproducibility of results.

- **Default**: `42`
- **Impact**: Affects dataset sampling and any stochastic operations
- **Recommendation**: Use same seed for fair comparisons

#### 3.6 Super Expert / Outlier Preservation

**Super Experts Protection**:
- `--perserve-super-experts` / `--singleton_super_experts true`
- Prevents pruning of "super experts" (experts with consistently high activations)
- Excludes last 25% of layers from super expert detection

**Outlier Experts Protection**:
- `--perserve-outliers` / `--singleton_outlier_experts true`
- Preserves outlier experts across all layers
- Useful for maintaining model stability

**Mutual Exclusivity**: Only one can be enabled at a time.

**Paper Reference**: See [Unveiling Super Experts in MoE LLMs](https://arxiv.org/abs/2507.23279)

#### 3.7 Router Weight Renormalization (`--renormalize-router-weights`)

**Purpose**: Renormalize top-k router weights to sum to 1.

- **Default**: `false`
- **When to enable**: If `model.config.norm_topk_prob == True`
- **Effect**: Ensures router weights are properly normalized before computing saliency

#### 3.8 Observation Recording Mode (`--record-pruning-metrics-only`)

**Purpose**: Optimize memory usage during calibration.

- **Default**: `true` (in pruning-cli.sh)
- **When enabled**: Only records metrics needed for pruning (not merging)
- **Memory savings**: Significant - skips recording similarity matrices and characteristic activations

### Parameter Priority Ranking

For most use cases, prioritize tuning these parameters in order:

1. **`--prune-method`**: Choose the right saliency criterion
2. **`--compression-ratio`**: Set target model size
3. **`--dataset-name`**: Match calibration data to target domain
4. **`--samples-per-category`**: Balance accuracy vs. efficiency
5. **`--perserve-super-experts` / `--perserve-outliers`**: Enable if stability issues occur

### Secondary Parameters

These typically work well with defaults:

- **`--model-max-length`**: 2048 tokens (default)
- **`--distance-measure`**: "cosine" (default)
- **`--output-file-name`**: Auto-generated based on settings
- **`--overwrite-observations`**: false (reuse cached activations)
- **`--overwrite-pruned-model`**: false (don't overwrite existing pruned models)

---

## 4. Quick Start Examples {#quick-start}

### Example 1: Basic REAP Pruning (50% compression)

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false
```

Parameters explained:
- `0`: GPU device
- `Qwen/Qwen3-30B-A3B`: Model name
- `reap`: Pruning method (REAP)
- `42`: Random seed
- `0.5`: Compression ratio (50% pruning)
- `theblackcat102/evol-codealpaca-v1`: Calibration dataset
- `true true true false false`: Enable lm_eval, evalplus, livecodebench; disable math, wildbench

### Example 2: Conservative Pruning (25% compression)

```bash
bash experiments/pruning-cli.sh 0,1 Qwen/Qwen3-30B-A3B reap 42 0.25 theblackcat102/evol-codealpaca-v1 true true false false false
```

Changes:
- `0,1`: Use 2 GPUs
- `0.25`: Only prune 25% of experts (more conservative)

### Example 3: Frequency-based Baseline

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B frequency 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false
```

Changes:
- `frequency`: Use simple frequency counting instead of REAP

### Example 4: With Super Expert Preservation

```bash
bash experiments/pruning-cli.sh 0 Qwen/Qwen3-30B-A3B reap 42 0.5 theblackcat102/evol-codealpaca-v1 true true true false false true false
```

Changes:
- `true false`: Enable super expert preservation (12th argument)

### Example 5: Python API Usage

```python
from reap.prune import main
import sys

# Set command line arguments
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

# Run pruning
main()
```

---

## Output and Results

After running pruning, you will find:

1. **Pruned Model**: Saved to `artifacts/{model_name}/{dataset_name}/pruned_models/{method}-seed_{seed}-{ratio}/`
   - Contains model weights, config, tokenizer
   - Includes `reap_args.yaml` with all experiment parameters

2. **Evaluation Results**: Saved to `{pruned_model_dir}/eval/`
   - LM-Eval scores
   - Code generation metrics
   - Benchmark results

3. **Observation Data**: Cached in `artifacts/{model_name}/{dataset_name}/all/observations_*.pt`
   - Reused for subsequent runs with same dataset
   - Contains expert statistics

---

## Tips for Best Results

1. **Start with defaults**: The default REAP configuration works well for most models
2. **Match calibration to target task**: Use code data for code models, math for math models
3. **Monitor evaluation metrics**: Check if pruning degrades target task performance
4. **Try different compression ratios**: 0.25, 0.5, 0.75 to find optimal trade-off
5. **Enable super expert preservation**: If you observe instability or performance drops
6. **Reuse observations**: Set `--overwrite-observations false` to save time on repeated experiments

---

## Additional Resources

- **Paper**: [REAP the Experts: Why Pruning Prevails for One-Shot MoE compression](https://arxiv.org/abs/2510.13999)
- **HuggingFace Models**: [Cerebras REAP Collection](https://huggingface.co/collections/cerebras/cerebras-reap)
- **Source Code**: `src/reap/prune.py`, `src/reap/observer.py`, `src/reap/args.py`
- **Issues**: Report bugs or ask questions on GitHub Issues

---

## Summary

- ✅ **Data is required**: REAP needs calibration data to compute expert saliency scores
- ✅ **Fully reproducible**: Complete open-source implementation with all necessary components
- ✅ **Key parameters**: Focus on `prune-method`, `compression-ratio`, and `dataset-name`
- ✅ **Easy to use**: Provided bash scripts and Python API for quick experimentation
- ✅ **Well-supported**: Multiple models, datasets, and evaluation benchmarks included

For questions or contributions, please open an issue or pull request on GitHub.
