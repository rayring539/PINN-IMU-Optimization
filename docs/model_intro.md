# PINN-IMU 模型简介

## 任务

根据 **加计 / 陀螺原始读数（LSB）与温度**，预测 **六轴温漂校正量 δ**（LSB），使校正后读数接近理想零偏（见 `core/data_pipeline.py` 中 `IDEAL6` 与标签定义）。

## 输入与输出

| 项目 | 说明 |
|------|------|
| **输入** | 每样本 7 列：`[ax, ay, az, gx, gy, gz, T_raw]`；若开启 `use_dTdt`，再拼接温度变化率 `Tdot`，共 **8 维**。 |
| **输出** | **6 维 δ**（与寄存器同量纲的 LSB），加到原始 6 轴上得到校正读数。 |

网络内部对输入做归一化（`x_mean` / `x_std` 等），**监督目标与评测主指标在 LSB 空间**（见 `eval_pinn.py`）。

## 网络结构（`IMUNet` / `PINN_IMU`）

多分支相加：**δ = 物理分支 + 迟滞分支（可选）+ 残差分支**。

1. **物理分支**  
   温度相对参考值的低阶多项式（及可选的 `Tdot` 项），在 **物理单位**（加计 g、陀螺 °/s）中建模再乘回 LSB，体现文档中的三因子 / 温漂物理形状。

2. **热特征分支**  
   小型 MLP（`thermal_net`）：由归一化温度得到热特征，供残差使用。

3. **迟滞分支（可选，`use_hysteresis`）**  
   GRU 对温度相关量建模，输出 6 维修正；无该权重时须与 checkpoint 一致（见 `load_imunet_from_dde_model_save` 中对权重的推断）。

4. **残差分支**  
   MLP：输入为 **归一化 raw6 + 热特征**，输出 6 维 δ，并乘以 `y_std` 与数据尺度对齐。

## 训练与约束

- **DeepXDE**（`train_pinn_dde.py`）：数据损失 + 多类物理正则（热光滑、材料先验、三因子项等，权重见 `config/pinn_train.yaml` 中 `physics_loss`）。
- **纯 PyTorch**（`train_pinn.py`）：同一套物理损失思想，循环自写。

## 权重文件说明

| 文件 | 含义 |
|------|------|
| `pinn_model_best.pt` | 训练结束由 `save_dde_checkpoint` 写出，含 `model_state` / `meta` / `ext_vars`，**推荐评测与 ONNX**。 |
| `dde_ckpt-<iter>.pt` | DeepXDE `Model.save()` 的中间 checkpoint，仅 `model_state_dict`；需同目录 **`train_config.json`**，`eval_pinn.py` 可自动加载（见 `load_eval_checkpoint`）。 |

## 评测指标（摘要）

- **`delta_lsb`**：δ 预测误差在 **LSB** 下的 RMSE / bad（阈值默认与 `val_bad_threshold` 一致）。  
- **物理单位**：将误差除以 `acc_scale` / `gyro_scale` 得到 **g** 与 **°/s**（`metrics_corrected_physical`、`delta_physical`）；同一阈值在 LSB 与物理空间等价，比例一致。

更细的公式与分支含义见 `core/pinn_model.py`、`core/pinn_dde.py` 及仓库根目录 `README.md`。
