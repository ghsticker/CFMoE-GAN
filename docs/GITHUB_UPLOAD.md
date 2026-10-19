# GitHub 上传前的文件整理建议

本文根据当前工作目录和完整模型配置整理。此次只新增文档、补充忽略规则及缺失依赖，没有删除现有代码，也没有恢复工作区中已经删除的脚本。

## 1. 建议保留

| 文件或目录 | 原因 |
| --- | --- |
| `README.md`、`environment.yml`、`.gitignore` | 使用说明、环境安装与上传排除规则 |
| `LICENSE` | 保留上游版权及许可证 |
| `train.py`、`test.py` | 训练和推理入口 |
| `models/` | 生成器、判别器、DINO 编码器和模型训练逻辑 |
| `data/` | 数据读取、配对和预处理 |
| `options/` | 参数定义与解析 |
| `util/` 中被训练/测试入口导入的模块 | 损失、日志、可视化与 HTML 输出 |
| 可复现的训练/测试脚本 | 如需提供一键运行，可根据 README 命令整理脚本，路径应可配置 |

`data/` 是 Python 源码目录，不能与存放真实数据的 `datasets/` 一起排除。

## 2. 建议不上传的生成文件

| 文件或目录 | 原因 | 忽略规则 |
| --- | --- | --- |
| `.DS_Store`、`.idea/`、`.vscode/` | 操作系统和编辑器配置 | 已覆盖 |
| `__pycache__/`、`*.pyc`、`*.swp` | 缓存和编辑临时文件 | 已覆盖 |
| `datasets/`、数据压缩包 | 数据单独通过 README 中的网盘链接提供 | 已覆盖 |
| `checkpoints/`、`pretrained/`、`*.pth`、`*.pt` 等 | 训练权重或第三方预训练模型 | 已覆盖 |
| `results/`、`results_archive/`、`wandb/`、TensorBoard 日志 | 训练和推理产物 | 已覆盖 |
| `bridge_moe_gates.csv` | 已生成的路由记录，约 2.2 MiB | 已新增 |
| `metrics_results.csv` | 本地指标汇总文件，当前工作区已删除 | 已新增 |
| `plot/sar2opt_frequency_sankey.html` | 已生成的交互图，约 4.6 MiB | 已新增 |
| `third_party/` | 用户自行获取的第三方源码 | 已新增 |

不再全局忽略 `*.png`，方便以后提交 README 方法图和少量展示图片；数据和结果仍按目录排除。

**注意：`.gitignore` 不会自动取消已被 Git 跟踪的文件。** 当前 `bridge_moe_gates.csv` 和 `plot/sar2opt_frequency_sankey.html` 已被跟踪；若决定不再发布，可在项目根目录执行以下命令，仅取消跟踪，保留本地文件：

```bash
git rm --cached -- bridge_moe_gates.csv plot/sar2opt_frequency_sankey.html
```

该命令尚未执行。现有 Git 历史中的文件也不会因取消跟踪而消失。

## 3. 与 SAR 转光学主流程无关，可不纳入发布

下列文件未被当前完整模型的训练/推理入口直接依赖，可按发布范围选择移除。此次未修改或删除它们。

| 文件或目录 | 建议 |
| --- | --- |
| `.replit` | 原项目的 Replit 配置，无需随本项目发布 |
| `CycleGAN.ipynb` | 原版 CycleGAN 演示，与 CFMoE-GAN 复现无关 |
| `imgs/horse2zebra.gif`、`imgs/edges2cats.jpg` | 原版示例图片；其中 GIF 约 7.3 MiB，可改为本项目结果图 |
| `scripts/edges/` | HED 边缘预处理脚本，非当前 SAR 配对训练流程 |
| `scripts/eval_cityscapes/` | Cityscapes 评估脚本，非当前三个数据集的评估流程 |
| `docs/README_es.md`、`docs/docker.md`、`docs/Dockerfile`、`docs/datasets.md` | 主要介绍上游项目；旧 Dockerfile 使用 CUDA 10.1 并克隆原版仓库，不适合作为本项目环境说明 |
| `docs/overview.md`、`docs/qa.md`、`docs/tips.md` | 可作为上游参考保留，也可在精简发布中移除 |
| `plot/sankey.py`、`plot/scatter_plot.py` | 分析画图脚本，使用本机绝对路径；如无对应图表复现需求，可不发布 |
| `util/rename_sen12.py` | 面向 SEN1-2 的本机重命名工具，非本次三个数据集的主流程 |
| `util/get_data.py` | 上游通用数据下载辅助代码，README 已单独提供当前数据链接 |
| `models/vision_transformer.py` | 当前 `networks.py` 使用自己的 Swin 包装器，此文件未被主流程导入 |

`models/cycle_gan_model.py`、`models/colorization_model.py`、`models/test_model.py`、`models/template_model.py` 及对应的其他数据读取器是上游扩展功能。若只发布当前 pix2pix + paired 流程，可进一步精简；保留它们也不影响主流程。`data/aligned_dataset.py` 仍是 pix2pix 的默认读取模式，除非同步修改默认参数，否则建议保留。

## 4. 不能仅因主实验未启用就直接删除的模块

`models/pix2pix_model.py` 在文件顶部直接导入以下内容。即使完整模型没有启用对应实验分支，直接删除这些文件仍会导致模型导入失败：

- `models/projected_discriminator.py`
- `models/sam_align_modules.py`
- `util/sam_rfl.py`
- `util/wavelet_loss.py`
- `util/loss.py`
- `util/dino_losses.py`

此外，`models/networks.py` 直接导入 `models/standalone_discriminator.py`。`train.py` 直接导入 `util/gate_logger.py`，需要 TensorBoard；`pix2pix_model.py` 导入的 `util/sam_rfl.py` 又直接导入 OpenCV。因此环境文件中补充了 `einops`、`opencv-python-headless` 和 `tensorboard`。

如果需要大幅精简 `pix2pix_model.py` 中的 SAM、Projected GAN、其他损失等分支，应同步修改导入、参数和分支代码，并在具备依赖和权重的环境中做训练/推理验证。

## 5. 可保留但需整理的辅助工具

- `util/handle_unicorn.py`：可能有助于说明 UNICORN 的目录转换；使用本机路径，且在顶层执行复制操作。建议参数化并增加入口保护后再作为通用工具介绍，不能直接当作可导入模块使用。
- `util/sample_paired_subset.py`：如果实验使用子集，建议保留并公布实际划分、随机种子或文件清单；当前含本机数据路径。
- `util/sar_nlm_despeckle.py`：若实际实验使用该预处理，应保留并记录设置；默认输入路径需改为用户自己的路径。
- `util/move_fake.py`、`util/move_real.py`：已接受命令行路径，适合保留用于整理评估图像。
- `util/loss_plot.py`：可选的日志画图工具；如要使用需安装 Matplotlib。
- `util/sam_cache_builder.py`：只用于 SAM 分支，需额外安装 `segment_anything`；完整模型配置无需运行。
- `run_test_and_metrics.py`：默认测试脚本已不存在，还保留本机结果路径。它解析外部脚本打印的指标，本身没有计算指标。若要公开承诺自动评估，应先补齐测试脚本和指标实现，再整理这些默认值。

## 6. 当前复现说明中的关键点

1. 参考的 `scripts/ablation/train_ablation_06_full.sh` 在开始整理前已从工作区删除。本次从 `HEAD` 读取其参数，将可运行命令直接展开写入 README，没有恢复该脚本。
2. 原完整配置为 `n_epochs=400`，代码默认 `n_epochs_decay=100`，总计 500 轮。
3. `options/base_options.py` 中 DINO 源码和 SAM 权重默认值仍为本机绝对路径。README 的完整训练命令显式覆盖 DINO 路径；SAM 分支才需要指定 SAM 权重。该文件原有修改保持原样。
4. 当前工作区未包含 `calculate-metrics-master/` 和原测试脚本，不能把批量指标评估描述为开箱即用。
5. 当前基础环境为 PyTorch 2.4.0；新的 DINOv3 源码需另行确认兼容性。README 链接了官方要求，没有将该环境表述为完整实验环境锁文件。
6. 当前机器缺少 PyTorch 等运行依赖和实验权重，因此此次验证限于参数、语法、文档和忽略规则，未执行模型训练或推理。

仓库保留了较多工作区删除和少量既有修改。提交前请检查拟上传的变更，避免把仍需保留的实验材料误删；原始许可证应始终保留。
