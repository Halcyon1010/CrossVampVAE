# CrossVampVAE: 基于双向交叉注意力流与 VampPrior 的变分自编码器

**CrossVampVAE** 是一个针对图像生成任务（主要是 CIFAR-10）的高级变分自编码器（VAE）实现。本项目结合了 **VampPrior** (Variational Mixture of Posteriors Prior) 和 **基于双向交叉注意力的归一化流 (Bi-directional Cross-Attention Normalizing Flow)**，旨在提高 VAE 的后验近似能力，缓解后验坍塌问题，并生成高质量的图像。

## 📖 项目简介

传统的 VAE 通常使用标准高斯分布作为先验，这限制了模型的表达能力。本项目通过以下核心技术改进了 VAE：

1. **VampPrior**: 使用一组可学习的“伪输入”（Pseudo-inputs）通过编码器生成的混合分布作为先验，使其比标准高斯先验更灵活。
2. **Cross-Attention Flow**: 在潜在空间引入了基于 Transformer 交叉注意力机制的可逆流模型 (`BiCrossAttnFlow`)。通过多层双向耦合层，增强了后验分布  的复杂度和灵活性。
3. **LPIPS 感知损失**: 结合传统的像素级重建损失（L1）和 LPIPS 感知损失，显著提升生成图像的视觉清晰度。

## 📂 文件结构

* **`models/vamp_flow.py`**: 核心模型定义。包含 `CrossFlowVampVAE` 类、`BiCrossAttnFlow` 流模型以及 ResNet 风格的编码器/解码器。
* **`train.py`**: 模型训练脚本。支持 FID 在线评估、伪输入可视化、LPIPS 监控等。
* **`test.py`**: 测试与生成脚本。用于生成大量样本以计算 FID，或重建测试集图像。
* **`PR_Calculate.py`**: 计算 Precision (精度) 和 Recall (召回率) 指标，评估生成多样性和真实性。
* **`csv_analysis.py`**: 单张图像质量分析工具（拉普拉斯方差、Tenengrad 梯度、频谱高频比）。
* **`comparison.py`**: 比较不同模型实验结果的脚本，生成箱线图和小提琴图。
* **`utils.py`**: 辅助工具，包含日志记录器等。

## 🛠️ 环境依赖

请确保安装了 Python 3.8+ 及 PyTorch。

```bash
# 安装依赖
pip install -r requirements.txt
```

**核心依赖库：**

* `torch`, `torchvision`
* `lpips` (用于感知损失)
* `pytorch-fid` (用于 FID 计算)
* `pandas`, `numpy`, `matplotlib`, `scipy` (用于数据分析与绘图)

## 🚀 快速开始

### 1. 数据准备

代码默认会自动下载 CIFAR-10 数据集。请确保在参数中指定正确的数据路径。

### 2. 训练模型 (Training)

使用 `train.py` 启动训练。你可以配置流模型的深度 (`flow_length`) 和 VampPrior 的组件数 (`num_components`)。

```bash
python train.py \
    --exp_name CrossVampVAE_Experiment \
    --data_dir /path/to/data \
    --save_dir ./results \
    --batch_size 128 \
    --lr 1e-4 \
    --latent_dim 128 \
    --num_components 100 \
    --epochs 1000 \
    --warmup_epochs 20 \
    --beta_max 0.2 \
    --fid_every 10

```

**主要参数说明：**

* `--exp_name`: 实验名称，结果将保存在 `save_dir/exp_name` 下。
* `--warmup_epochs`: KL 散度权重  的预热轮数。
* `--fid_every`: 每隔多少个 Epoch 计算一次 FID 分数。
* 在 `models/vamp_flow.py` 中默认启用了 `flow_length=8` (见 `train.py` 中的模型初始化部分)。

### 3. 生成与测试 (Testing & Generation)

使用 `test.py` 加载训练好的权重，生成用于 FID 计算的伪造图像或重建图像。

```bash
python test.py \
    --resume ./results/CrossVampVAE_Experiment/best_fid.pth \
    --data_dir /path/to/data \
    --save_dir ./results \
    --num_samples 50000 \
    --gen_batch_size 100 \
    --real_dirname real_cifar \
    --fake_dirname fake_cifar \
    --do_recon False

```

该脚本会：

1. 导出 CIFAR-10 真实图片到 `real_cifar` 文件夹。
2. 使用模型生成 50,000 张图片到 `fake_cifar` 文件夹。
3. (可选) 重建训练集图片。
4. 打印计算 FID 的命令。

### 4. 评估 (Evaluation)

#### 计算 FID

推荐使用官方 `pytorch-fid` 工具：

```bash
python -m pytorch_fid /path/to/data/real_cifar /path/to/data/fake_cifar
```

#### 计算 Precision & Recall

使用 `PR_Calculate.py` 计算生成分布的流形覆盖率（Recall）和质量（Precision）。

```bash
python PR_Calculate.py \
    --real_dir /path/to/data/real_cifar \
    --fake_dir /path/to/data/fake_cifar \
    --device cuda \
    --pca_dim 256
```

#### 图像质量统计分析

使用 `csv_analysis.py` 计算图像的锐度和其他统计指标，并缓存结果。

```bash
python csv_analysis.py \
    --gen /path/to/data/fake_cifar \
    --out_dir ./results/single_image_quality \
    --tag CrossVampVAE
```

#### 模型对比

如果有多个模型的分析缓存文件，可以使用 `comparison.py` 生成对比图表。

```bash
python comparison.py \
    --cache_dir ./results/single_image_quality \
    --out_dir ./results/compare \
    --model_order "VAE,CrossVampVAE,real"
```

## 🧠 模型细节 (Model Details)

### CrossFlowVampVAE (`models/vamp_flow.py`)

主要流程如下：

1. **Encoder**: 将图像 $x$ 映射到 $\mu$ 和 $\log \sigma$ 。
2. **Reparameterization**: 采样得到 $z_0$。
3. **Flow (`BiCrossAttnFlow`)**:
* 若 `flow_length > 0`，将  通过一系列可逆的双向交叉注意力耦合层转换为 $z_K$。
* 计算雅可比行列式的对数 `log_det` 用于修正 KL 散度。


4. **Decoder**: 将 $z_K$ 解码为重构图像 。
5. **Loss**:
* **Reconstruction**: L1 Loss + LPIPS Loss。
* **KL Divergence**: 计算 $q(z_K|x)$ 与 VampPrior $p_{vamp}(z_K)$ 之间的 KL 散度。由于引入了流模型，后验概率 $q(z_K|x)$ 通过变量代换公式计算：$$ \log q(z_K|x) = \log q(z_0|x) - \log \left| \det \frac{\partial z_K}{\partial z_0} \right| $$ 。


## 📊 结果示例

训练过程中，日志会记录以下关键指标：

* `LPIPS`: 感知距离，越低越好。
* `KL_Raw`: 原始 KL 散度值。
* `FID`: 生成图像与真实图像的分布距离，越低越好。

代码会自动保存 `pseudo_inputs`（VampPrior 学到的锚点）和生成的样本网格图以便目视检查。

---
