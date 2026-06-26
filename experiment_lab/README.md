# C3Box Experiment Lab

这个目录用于做“模型 x 数据集”的适配性实验。核心思路是先做静态检查，再生成小规模 smoke-test 配置，确认能跑通后再扩展到完整矩阵。

## 1. 数据目录

`utils/data.py` 已支持环境变量配置数据路径。默认会从 `./data/<dataset>` 读取；推荐显式设置：

```bash
export C3BOX_DATA_ROOT=/path/to/c3box_data
```

ImageFolder 类数据默认布局：

```text
$C3BOX_DATA_ROOT/aircraft/train/<class_name>/*.jpg
$C3BOX_DATA_ROOT/aircraft/val/<class_name>/*.jpg
```

也可以单独覆盖某个数据集：

```bash
export C3BOX_AIRCRAFT_ROOT=/path/to/aircraft
export C3BOX_AIRCRAFT_TRAIN_DIR=/path/to/aircraft/train
export C3BOX_AIRCRAFT_VAL_DIR=/path/to/aircraft/val
```

CIFAR 类数据使用 torchvision 数据集根目录，例如 `C3BOX_CIFAR224_ROOT`。

## 2. 适配性检查

```bash
cd C3Box
python experiment_lab/check_environment.py
```

检查器会报告：

- Python 包是否存在，例如 `open_clip`、`timm`、`easydict`。
- 数据集是否有 `labels.json` 和 `templates.json` 条目。
- 本地 train/val 数据目录是否存在。
- 特殊模型资源是否存在，例如 `c.pth`、ENGINE 的 `*_des.json`、CLG-CBM 的 concept 文件。

注意：当前 `DataManager` 会无条件读取 `labels.json` 和 `templates.json`，所以即使是图像-only 模型，数据集也需要这两个 metadata 条目。

## 3. 生成 smoke-test 配置

```bash
python experiment_lab/make_matrix_configs.py \
  --preset smoke \
  --models simplecil,zs_clip,finetune,ease,tuna,proof,engine,bofa \
  --datasets cifar224,aircraft,cars,food101
```

smoke 配置会：

- 设置 `max_tasks=1`，只跑第一个增量任务。
- 把常见 epoch 字段压到 `1`。
- 把 batch size 降到最多 `8`。
- 保留原始模型的大部分超参数。

生成文件位于：

```text
experiment_lab/generated_configs/smoke/
```

## 4. 运行矩阵

先 dry-run 看命令：

```bash
python experiment_lab/run_matrix.py \
  --configs experiment_lab/generated_configs/smoke \
  --dry-run
```

确认后正式跑：

```bash
python experiment_lab/run_matrix.py \
  --configs experiment_lab/generated_configs/smoke \
  --stop-on-error
```

## 5. 如何理解“适配什么数据”

可以先按数据需求把模型分组：

- 图像分类基础流程：`finetune`、`simplecil`、`foster`、`memo`、`l2p`、`dualprompt`、`coda`。
- 需要类别文本模板：`zs_clip`、`rapf`、`proof`、`bofa`、`mind`、`engine`、`clg_cbm`。
- 需要额外语义资源：`engine` 需要 `utils/engine/chat/<dataset>_des.json`；`clg_cbm` 需要 concept JSON。
- 需要本地 CLIP checkpoint：`ease`、`tuna`、`dualprompt`、`coda`、`aper_*` 目前会读 `c.pth` 或 `.p/c.pth`。

建议流程：

1. 用 `check_environment.py` 找出 blocked pair。
2. 对 ready pair 生成 smoke 配置。
3. smoke 跑通后，把 preset 换成 `full` 生成完整实验配置。
4. 对比每个模型在不同数据集上的最终平均准确率、遗忘率和训练稳定性。
