# IMU 温漂 PINN

基于物理信息神经网络（PINN）的 IMU 六轴温度漂移校正项目：在 **PyTorch** 与 **DeepXDE** 两套训练流程下，联合数据拟合与物理约束（热传导光滑、材料先验、三因子误差模型等），细节见 `config/pinn_train.yaml` 与 `core/` 中损失实现。

## 功能概览

- **传感器标度**：加速度 `÷2048`（g）、角速度 `÷16`（°/s）、温度 `÷256`（℃），物理损失在物理单位下计算。
- **DeepXDE 训练**（`train_pinn_dde.py`）：`dde.Model`、多损失项、`dde.Variable` 物理参数、TensorBoard、进度条、可选 L-BFGS、**ONNX 导出**、多 GPU `DataParallel`。
- **纯 PyTorch 训练**（`train_pinn.py`）：同等物理损失与配置 YAML。
- **数据划分**：支持 `explicit`（默认 `1.txt`–`4.txt` 训练，`5.txt` 测试）或 `random`（多文件中随机留一作测试）。

## 环境要求

- Python ≥ 3.10（推荐 3.10–3.12；3.13 需安装带 CUDA 的 wheel 时自行核对 [PyTorch 官网](https://pytorch.org/get-started/locally/)）
- **GPU**：需安装 **CUDA 版** PyTorch（`torch.cuda.is_available()` 为 `True`），CPU 版 wheel（`+cpu`）无法使用 GPU。
- 依赖安装：

```bash
pip install -r requirements.txt
```

主要依赖：`torch`、`numpy`、`matplotlib`、`pyyaml`、`tensorboard`、`tqdm`、`onnx`、`scipy`、`scikit-learn`、`scikit-optimize`（后三者亦为本地 `deepxde/` 导入所需）。

## 仓库结构

```
IMU/
├── README.md
├── requirements.txt
├── config/
│   └── pinn_train.yaml       # 训练与数据划分主配置
├── core/                     # 模型、数据 IO、DeepXDE 封装
├── deepxde/                  # 随仓库提供的 DeepXDE 源码（仅包含包本体，不含上游 docs/examples）
├── tools/                    # 画图、ONNX 导出脚本等
├── train_pinn.py             # PyTorch 训练入口
├── train_pinn_dde.py         # DeepXDE 训练入口（推荐）
└── eval_pinn.py / eval_*.py
```

## 数据格式

- 文本文件，每行 7 列（无表头），制表符或空格分隔：
  `[ax, ay, az, gx, gy, gz, T_raw]`（与 `config` 中 `sensor` 标度一致）。
- 配置中 `data.data_dir` 指向数据目录；默认 **explicit** 划分见 `config/pinn_train.yaml` 中 `train_files` / `test_files`。

## 快速开始

### 1. 配置路径

编辑 `config/pinn_train.yaml`：

- `data.data_dir`：数据目录（如 `D:\IMU_data`）
- `output.out_dir`：日志与模型输出目录
- `training.gpu_ids`：多卡如 `[0, 1, 2, 3]`，单卡 `[0]`

### 2. DeepXDE 训练

```bash
python train_pinn_dde.py --config config/pinn_train.yaml
```

常用覆盖示例：

```bash
python train_pinn_dde.py --config config/pinn_train.yaml --epochs 100 --gpus 0,1
python train_pinn_dde.py --split_mode explicit --train_files "1.txt,2.txt,3.txt,4.txt" --test_files "5.txt"
```

后台训练（服务器推荐）：

```bash
nohup python train_pinn_dde.py --config config/pinn_train.yaml > train.log 2>&1 &
tail -f train.log
```

产出（在 `out_dir`）：`pinn_model_best.pt`、`pinn_model.onnx`、`train_test_split.json`、TensorBoard 日志 `tb/run_*` 等。

### 3. 评测与可视化

```bash
python eval_pinn.py --model_path <out_dir>/pinn_model_best.pt --split_meta <out_dir>/train_test_split.json
python tools/plot_pinn_results.py --model_path <out_dir>/pinn_model_best.pt --split_meta <out_dir>/train_test_split.json --out_png <out_dir>/pinn_corrected_6d.png
```

### 4. 仅导出 ONNX（已有 checkpoint）

```bash
python tools/export_onnx.py --ckpt <out_dir>/pinn_model_best.pt
```

## TensorBoard

每次训练自动在 `<out_dir>/tb/run_<时间戳>/` 下创建独立日志，多次训练不会互相覆盖。

```bash
tensorboard --logdir <out_dir>/tb
```

远程服务器通过 SSH 端口转发查看：

```bash
ssh -L 6006:127.0.0.1:6006 user@server -p <port> -i <key>
tensorboard --logdir <out_dir>/tb --port 6006 --host 127.0.0.1
# 本机浏览器打开 http://127.0.0.1:6006
```

TensorBoard 面板说明：

| 标量组 | 含义 | 更新频率 |
|--------|------|----------|
| `train/*` | 训练集各项 loss（data、physics 子项、total） | 每个优化步 |
| `train/lr` | 学习率 | 每个优化步 |
| `val/*` | 验证集各项 loss | 每 `log_interval` 个 epoch |
| `meta/epoch` | 近似 epoch 数 | 每个优化步 |

## 配置说明

关键训练参数（`config/pinn_train.yaml`）：

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `training.epochs` | 训练轮数 | 50 |
| `training.batch_size` | 批大小 | 8000 |
| `training.lr` | 初始学习率 | 1e-3 |
| `training.log_interval` | 每 N 个 epoch 做一次验证 | 1 |
| `training.save_best_checkpoint_only` | 仅保存验证 loss 最优的 checkpoint | false |
| `training.physics_warmup` | 物理损失从 0 升至满权重的 epoch 数 | 30 |
| `training.lbfgs_iters` | Adam 后 L-BFGS 精调步数（0=关闭） | 0 |
| `training.gpu_ids` | GPU 索引列表 | [0] |

## 说明

- 本地 **大文件数据、训练权重、ONNX、PDF** 等已通过 `.gitignore` 排除，请勿将含密钥或隐私的路径提交到公开仓库。
- 更细的物理项与损失权重含义见 `config/pinn_train.yaml` 注释与 `core/pinn_dde.py`、`core/pinn_model.py`。

## 许可证

项目代码与用法请遵循各依赖库许可证；`deepxde/` 内容遵循其上游项目许可证。
