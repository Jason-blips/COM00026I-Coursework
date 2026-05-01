# 对比实验分析报告（20260430_123220）

## 1. 实验目的

在**不修改模型结构**、固定数据集与训练轮数（`30 epoch`）的前提下，通过调整训练超参数提升多分类任务的 `test_acc`，并比较两阶段训练与单阶段训练的效果差异。

---

## 2. `run.py` 的对比实验设计

本轮实验由 `run.py` 自动执行，核心策略是：

- 固定公共参数：`epochs=30`、`batch_size=32`、`val_ratio=0.1`、`seed=42`
- 每组实验仅改训练超参数（学习率、warmup、两阶段开关等）
- 每组独立目录保存日志与权重
- 自动解析日志中的 `best_val_acc / test_loss / test_acc`
- 最终生成 `summary.csv` 并按 `test_acc` 排名

### 使用到的关键模块与方法

- **实验配置模块**：`EXPERIMENTS`（定义 S1~S5 配置）
- **命令执行模块**：`subprocess.Popen(...)`（逐组调用 `train.py`）
- **日志采集模块**：实时写入 `<experiment>/train.log`
- **结果解析模块**：正则 `TEST_RE`、`VAL_RE` 提取关键指标
- **结果汇总模块**：`write_summary(...)` 写出 `summary.csv`
- **排序展示模块**：按 `test_acc` 进行 ranking 输出

---

## 3. 本轮实验配置（S1~S5）

- `S1_v2_reproduce`：两阶段，`stage1=3ep@3e-4`，`stage2 lr=0.05`
- `S2_two_stage_lr045`：两阶段，`stage2 lr=0.045`
- `S3_two_stage_lr055`：两阶段，`stage2 lr=0.055`
- `S4_two_stage_long_stage1`：两阶段，`stage1=5ep`（更长预热）
- `S5_single_stage_strong_sgd`：单阶段 SGD，`lr=0.05`

> 本轮共同设置了 `patience=5`（较激进的早停）。

---

## 4. 实验结果表格

数据来源：`results/auto_runs/20260430_123220/summary.csv`

| Rank | Experiment | Return Code | Best Val Acc | Test Loss | Test Acc |
|---|---|---:|---:|---:|---:|
| 1 | S5_single_stage_strong_sgd | 0 | 0.6712 | 1.3219 | **0.6402** |
| 2 | S3_two_stage_lr055 | 0 | **0.6793** | 1.3343 | 0.6312 |
| 3 | S1_v2_reproduce | 0 | 0.6522 | 1.3739 | 0.6271 |
| 4 | S2_two_stage_lr045 | 0 | 0.6332 | 1.4447 | 0.6023 |
| 5 | S4_two_stage_long_stage1 | 0 | 0.3016 | 2.7054 | 0.2434 |

---

## 5. 结果分析

### 5.1 主要发现

- **单阶段强 SGD（S5）目前最优**，`test_acc=0.6402`，优于所有两阶段配置。
- 两阶段中，`lr=0.055`（S3）优于 `0.05`（S1）和 `0.045`（S2），说明当前有效区间偏高学习率。
- `S4`（拉长 stage1）显著失败，表明在该配置下 AdamW 预阶段过长，削弱了后续 SGD 收敛效率。

### 5.2 验证集与测试集的关系

- `S3` 的 `best_val_acc` 最高（0.6793），但 `test_acc` 低于 `S5`，说明仅看 `val_acc` 不足以保证最终 test 最优。
- 该现象提示：当前划分下存在一定验证-测试偏差，建议后续至少做多 seed 复验。

### 5.3 当前瓶颈判断

- 在“模型结构不变”的约束下，纯靠本轮这组超参，模型大致稳定在 `0.60~0.64` 区间；
- 要冲击 `0.75+`，需要更系统的训练策略升级（loss、增强、调度策略、种子复验），仅微调当前 5 组参数不够。

---

## 6. 下一轮实验建议

### 6.1 继续保持不改模型结构

- 以 `S5` 为基线，做小范围网格：
  - `lr`: `0.045 / 0.05 / 0.055`
  - `warmup_epochs`: `3 / 5 / 7`
  - `weight_decay`: `3e-4 / 5e-4 / 8e-4`

### 6.2 训练流程层面的改进（仍不改模型结构）

- 将“快筛实验”保留 `patience=5`，但“正式复验”改 `patience=10~12`
- 对 Top2 配置使用 `seed=42/7/123` 重复训练，比较均值与方差
- 在 `train.py` 中增加 `best_acc` 权重保存与最终双模型测试

---

## 7. 复现实验命令

```bash
python run.py
```

只跑指定实验：

```bash
python run.py --only S5_single_stage_strong_sgd S3_two_stage_lr055
```

