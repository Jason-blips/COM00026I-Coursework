# 30 Epoch 对比实验计划（S5 Baseline）

## 目标

在不修改 `model.py` 的前提下，以当前最优的 `S5` 配置为 baseline，继续通过训练超参数微调提升 `test_acc`。

## 可复现基线（已写入 `train.py` 默认参数）

当前 `train.py` 默认已切到 S5 风格：

- `--no-two_stage`（默认关闭两阶段）
- `--lr 0.05`
- `--warmup_epochs 5`
- `--weight_decay 5e-4`
- `--momentum 0.9`
- `--patience 5`

因此你直接运行即可复现 baseline：

```bash
python train.py
```

## 本轮对比实验（`run.py`）

### B0 baseline
- `B0_s5_baseline`：单阶段 SGD，`lr=0.05, warmup=5, wd=5e-4`

### 学习率局部搜索（吸取“高学习率有效”经验）
- `B1_single_stage_lr045`
- `B2_single_stage_lr055`

### warmup 消融（吸取“收敛稳定性重要”经验）
- `B3_single_stage_warmup3`（对比 baseline 的 warmup=5）

### 正则强度搜索（控制过拟合）
- `B4_single_stage_wd3e4`
- `B5_single_stage_wd8e4`

### 两阶段兜底验证（吸取“两阶段有时不稳定”经验）
- `B6_two_stage_short_rescue`：缩短 stage1，仅做轻量两阶段验证

## 运行命令

跑全部：

```bash
python run.py
```

只跑关键组：

```bash
python run.py --only B0_s5_baseline B2_single_stage_lr055 B5_single_stage_wd8e4
```

## 输出

- 每组日志：`results/auto_runs/<timestamp>/<experiment>/train.log`
- 汇总表：`results/auto_runs/<timestamp>/summary.csv`

## 判定策略

1. 以 `test_acc` 为主排序；
2. 若差距小于 `0.01`，优先选 `best_val_acc` 更高且曲线更平滑的配置；
3. 若本轮最佳仍 < 0.65，下一轮转向：
   - loss 对比（CE vs Focal）
   - 多 seed 复验（42/7/123）
