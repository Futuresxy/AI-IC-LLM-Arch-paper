# LUT Tensor Core：一种面向基于 LUT 的低比特 LLM 推理的软件–硬件协同设计

Zhiwen Mo*

Imperial College London

London, United Kingdom

Microsoft Research

Beijing, China

[zhiwen.mo25@imperial.ac.uk](mailto:zhiwen.mo25@imperial.ac.uk)

Lei Wang*

Peking University

Beijing, China

Microsoft Research

Beijing, China

[leiwang1999@outlook.com](mailto:leiwang1999@outlook.com)

Jianyu Wei*

University of Science and Technology
 of China

Hefei, China

Microsoft Research

Beijing, China

[noob@mail.ustc.edu.cn](mailto:noob@mail.ustc.edu.cn)

Zhichen Zeng*

University of Washington

Seattle, USA

Microsoft Research

Beijing, China

[zczeng@uw.edu](mailto:zczeng@uw.edu)

Shijie Cao†

Microsoft Research

Beijing, China

[shijiecao@microsoft.com](mailto:shijiecao@microsoft.com)

Lingxiao Ma

Microsoft Research

Beijing, China

[lingxiao.ma@microsoft.com](mailto:lingxiao.ma@microsoft.com)

Naifeng Jing

Shanghai Jiao Tong University

Shanghai, China

[sjtuj@sjtu.edu.cn](mailto:sjtuj@sjtu.edu.cn)

Ting Cao

Microsoft Research

Beijing, China

[Ting.Cao@microsoft.com](mailto:Ting.Cao@microsoft.com)

Jilong Xue

Microsoft Research

Beijing, China

[jxue@microsoft.com](mailto:jxue@microsoft.com)

Fan Yang

Microsoft Research

Beijing, China

[fanyang@microsoft.com](mailto:fanyang@microsoft.com)

Mao Yang

Microsoft Research

Beijing, China

[maoyang@microsoft.com](mailto:maoyang@microsoft.com)

# Abstract

大语言模型（LLM）的推理变得越来越消耗资源，这推动了社区向低比特模型权重方向发展，以减小内存占用并提升效率。此类低比特 LLM 需要使用混合精度矩阵乘（mpGEMM），即用低精度权重与高精度激活相乘，这是一类重要但尚未被充分研究的运算。现成硬件并不原生支持这种操作，只能通过间接的、因而低效的基于反量化的实现方式来完成。

在本文中，我们研究了面向 mpGEMM 的查找表（LUT）方法，并发现传统的 LUT 实现难以达到理论上的收益。为释放 LUT 基 mpGEMM 的全部潜力，我们提出 LUT TENSOR CORE，这是一种面向低比特 LLM 推理的软件–硬件协同设计。与传统 LUT 设计相比，LUT TENSOR CORE 的区别在于：1）**软件侧优化**，通过最小化表预计算开销以及权重重解释（reinterpretation）来减少表存储；2）**基于 LUT 的 Tensor Core 硬件设计**，采用拉长的（elongated）分块形状最大化表复用，并采用类似比特串行（bit-serial）的设计以支持 mpGEMM 中多种精度组合；3）**新的指令集与编译优化**，专门面向 LUT 基 mpGEMM。与现有纯软件 LUT 实现相比，LUT Tensor Core 显著提升性能；与此前最先进的 LUT 加速器相比，其计算密度与能效提升了 $1.44\times$。

# CCS Concepts

- 计算机系统组织  $\rightarrow$  神经网络；体系结构；
- 硬件  $\rightarrow$  算术与数据通路电路。

# Keywords

低比特 LLM，软硬件协同设计，LUT，加速器

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/170b3e2c3a2d7bd440aa777620bb4902cad7eb81f502dc000276172e862b34e8.jpg)

本工作基于 Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License 进行许可。

ISCA '25，日本东京

© 2025 版权所有者 / 作者保留所有权利。

ACM ISBN 979-8-4007-1261-6/25/06

https://doi.org/10.1145/3695053.3731057

# ACM Reference Format:

Zhiwen Mo, Lei Wang, Jianyu Wei, Zhichen Zeng, Shijie Cao, Lingxiao Ma, Naifeng Jing, Ting Cao, Jilong Xue, Fan Yang, and Mao Yang. 2025. LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference. In Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA '25), June 21-25, 2025, Tokyo, Japan. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3695053.3731057

# 1 Introduction

大语言模型（LLM）的出现为各类 AI 应用带来了变革性的机遇 [1, 3, 28, 65]。然而，LLM 的部署需要大量硬件资源 [21, 54, 55]。为降低推理成本，低比特 LLM 成为一种有前景的方向 [10, 15, 31, 40]。在诸多方案中，**权重量化**（即使用低精度权重和高精度激活对 LLM 进行量化）尤为受到关注，因为它在保持模型精度的同时，既减少了内存开销又降低了计算成本 [39, 75, 81]。当前，4 比特权重量化已经十分普遍 [12, 32, 64]，学术界和工业界也在积极探索 2 比特甚至 1 比特的方案，以进一步提升效率 [4, 14, 29, 42, 44, 49, 68]。

权重量化将 LLM 推理的核心计算模式从传统的通用矩阵乘（GEMM）转移到**混合精度 GEMM（mpGEMM）**：其中一个输入矩阵为低精度（如 INT4/2/1 权重），另一个则保持高精度（如 FP16/8、INT8 激活）。目前，现成硬件并不原生支持混合精度运算。因此，大多数低比特 LLM 推理系统不得不采用基于反量化的 mpGEMM 方法 [16, 39, 51, 69]。反量化将低比特表示放大到硬件支持的 GEMM 精度。对于大 batch 场景，这些额外操作可能成为性能瓶颈。

查找表（LUT）是另一种常见的低比特计算方案，非常适合用于 mpGEMM [26, 38, 45, 53, 71]。通过预计算低精度权重与高精度激活之间的乘积，LUT 方法可以消除反量化需求，并用简单的表查找替代复杂运算。在实践中，LUT 按 tile（小块）维度实现。对于 mpGEMM 的每个小 tile，会针对该 tile 内的激活预先计算并构建查找表，然后在权重矩阵列之间复用，从而在保持效率的同时大幅减少存储开销。

尽管如此，基于 LUT 的 mpGEMM 在软硬件实现上仍面临显著的性能缺口与挑战。在软件方面，LUT kernel 受到有限的指令支持以及低效的内存访问模式的制约，导致其在 GPU 上的性能低于基于反量化的 kernel，如图 4 所示。在硬件方面，传统 LUT 设计没有针对 mpGEMM 进行专门优化，往往难以达到预期的性能提升。其主要挑战包括：表预计算与存储开销大、多比特宽组合支持受限、非最优分块形状导致效率下降，以及缺乏专门的指令集与编译支持等；详见 §2.3。

LUT TENSOR CORE 通过系统性的软硬件协同设计来解决这些问题。通过在软件中优化对硬件不友好的任务（例如表预计算和存储管理），LUT TENSOR CORE 将硬件侧的工作量降到更低，简化硬件设计并提升其紧凑性和效率。具体而言：

**软件优化（§ 3.1）**。为摊销查找表预计算开销，我们观察到传统设计在多个单元之间存在大量重复预计算。LUT TENSOR CORE 将预计算拆分为一个独立算子，从而避免冗余，并与前一算子进行融合，以进一步减少内存访问。为降低存储开销，LUT TENSOR Core 暴露并利用 mpGEMM 查找表固有的对称性，通过将 ${0,1}$ 重新解释为 ${-1,1}$，将表大小减半。LUT TENSOR Core 还通过对查找表进行量化来减小表宽度，并支持多种激活比特宽度。

**硬件定制化（§ 3.2）**。LUT TENSOR CORE 定制了基于 LUT 的 Tensor Core 设计。上述软件优化将部分电路任务迁移到软件端，从而简化了硬件设计，将广播与多路选择器（multiplexer）的需求降低一半。同时，LUT TENSOR CORE 引入类似比特串行的灵活电路，以适配各种混合精度组合。此外，LUT TENSOR CORE 还针对 LUT 基 Tensor Core 的形状进行了设计空间探索（DSE），识别出一种拉长的分块形状，可以更加高效地复用查找表。

**新指令与编译支持（§ 3.3）**。LUT TENSOR Core 将传统的矩阵乘加（MMA）指令扩展为 LUT 基的矩阵乘加（LMMA）指令集，其中包含了操作数类型与形状等必要元信息。借助这一扩展，LUT TENSOR Core 能利用 LMMA 中提供的形状信息，通过 tile 级深度学习编译器 [5, 62, 84] 重新编译 LLM 工作负载，从而为新硬件生成高效的 kernel。

我们的 LUT 基 Tensor Core 相比传统 Tensor Core 实现了 $4 \times$ 到 $6 \times$ 的功耗和面积（PPA）降低。为验证其在 mpGEMM 上的性能提升，我们将 LUT 基 Tensor Core 设计与指令集集成到 GPU 硬件模拟器 Accelsim [30] 中。结果显示，在仅占用传统 Tensor Core 16% 面积的前提下，我们的 LUT 基 Tensor Core 就能在 mpGEMM 性能上实现超越。与最先进的 LUT 软件实现 [53] 相比，LUT TENSOR CORE 在通用矩阵向量乘（GEMV）中可获得最高 $1.42 \times$ 的加速，在 GEMM 中则可达到最高 $72.2 \times$ 的加速。相较于最先进的 LUT 加速器 [38]，LUT TENSOR CORE 在计算密度与能效上提升了 $1.44 \times$，这得益于软硬件协同优化。我们的代码已开源于：https://github.com/microsoft/T-MAC/tree/LUTTensorCore_ISCA25。

我们的主要贡献总结如下：

- 我们提出 LUT TENSOR CORE，这是一种面向 LUT 基 mpGEMM 的软硬件协同设计，用于提升低比特 LLM 推理效率。
- 实验表明，所提出的 LUT 基 Tensor Core 能在功耗、性能与面积（PPA）方面取得 $4 \times$ 到 $6 \times$ 的收益。对于 BitNet 以及量化后的 LLaMA 等低比特 LLM，LUT TENSOR CORE 在保证面积与精度可比的前提下，可以获得 $2.06 \times$ 到 $5.51 \times$ 的推理加速。
- 除了效率之外，我们的设计还可以支持广泛的权重精度（如 INT4/2/1）和激活精度（如 FP16/8、INT8）。此外，借助扩展的 LMMA 指令与编译优化，LUT TENSOR Core 可以无缝集成到现有推理硬件与软件栈中。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/c0a62dd8f523f380b90fd9ffa8cfa96caaad4f4795cf27f4780bdb24ebd8cf23.jpg)
 图 1：LLM 中的仅解码器（decoder-only）Transformer 块。主要计算为 GEMM 操作（或在权重量化场景下的 mpGEMM 操作）。

# 2 Background and Motivation

# 2.1 LLM Inference and Low-Bit Quantization

近期的 LLM 主要采用仅解码器的 Transformer 架构 [66]，如图 1 所示。具体而言，LLM 由多个顺序堆叠的 Transformer 层构成，每个 Transformer 层包含一个多头注意力块，后接一个前馈块。在这两类模块中，主要的计算都是 GEMM，或在权重量化场景下的 mpGEMM。随着 LLM 规模的扩大，其对硬件资源的需求也急剧增加 [21, 28]。例如，LLAMA-2-70B [65] 仅模型权重（FP16）就需要 140GB 内存，远超当前 NVIDIA A100 或 H100 等现代 GPU 的显存容量，这给 LLM 部署带来了巨大的挑战。

为降低 LLM 部署的推理成本，低比特量化逐渐成为主流方案 [10, 12, 64, 76]。量化通过降低模型数值表示的精度，从而减少内存占用并缩短计算时间。在 LLM 量化中，**权重量化**相较于激活量化更受青睐 [37, 39]。这是因为模型权值是静态的，可以离线进行量化；权重可以被量化为 4 比特、2 比特甚至 1 比特。在 4 比特权重场景下，后训练量化（PTQ）几乎不会带来精度损失 [12, 64, 76]。近期的研究与实践表明，在相同内存预算下，使用量化感知训练（QAT）时，2 比特权重量化在模型精度上优于 4 比特 [14, 42, 49]。BitNet 进一步表明，从头训练 1.58 比特（三值）乃至 1 比特（二值）权重的模型，其精度可以与 16 比特模型相当 [44, 68]。ParetoQ [42] 也指出，在考虑硬件约束时，2 比特量化在内存减少与加速方面具有很有前景的潜力。

相对而言，激活是在推理过程中即时生成的，高方差且存在动态离群值 [10, 18, 73]。由于这些离群值的存在，将激活精度量化到 8 比特以下非常困难。不同的权重与激活比特宽组合已经在多种模型和场景下进行了探索 [10, 14, 15, 19, 68]，结果表明不存在一种在所有场景下通用的最佳组合。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/e730852f9bbe2aa4d9c8853c84c87b7591d98643e6b57dd8649bf08253e1f4a4.jpg)
 图 2：（a）GEMM，（b）带反量化的间接 mpGEMM，（c）面向低比特 LLM 推理的直接 mpGEMM。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/10c9d95b4a615beac6eb8d112847e13e3f2dd757dbe8e6e281ffc4fb9fdb9495.jpg)

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/b091d9ad72a8331ab5c6f22c3ade52b817706e10b515ad9175106e3d0782e824.jpg)

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/3920775a0c25696df758f7cac07a70df0b4c4ca8b72dc9d0ddc32498bcdfa39f.jpg)
 图 3：一个针对 FP16 激活和 INT1 权重的朴素 LUT 基 mpGEMM tile 示例。通过预计算表，表查找可以替代 4 元向量的点积运算。

# 2.2 LUT-based mpGEMM for Low-Bit LLM

权重与激活的不同比特宽度带来了独特的混合精度 GEMM（mpGEMM）需求，例如 INT4/2/1 乘以 FP16，如图 2 所示。当前商业 LLM 推理硬件（如 GPU 和 TPU）并不原生支持 mpGEMM，而是专注于输入格式统一的传统 GEMM。基于反量化的 mpGEMM 通过将低精度权重放大到与高精度激活匹配来弥补这一缺口 [50, 69]。然而，这种方法引入了额外的反量化操作，丧失了低精度计算在效率上的优势。

基于 LUT 的 mpGEMM 是一种越来越受关注的低比特 LLM 推理方法 [26, 38, 45, 53, 71]。其核心思想是预计算高精度激活与低精度权重之间的点积，并将结果存储在查找表（LUT）中以供快速访问。与为所有可能的高精度与低精度取值组合预计算一个巨大的查找表（例如 FP16  $\times$  INT4，对应表大小为 $(2^{16} \times 2^{4})$）不同，LUT 基 mpGEMM 采用分块（tile）方式组织计算。对于 mpGEMM 操作中的每个小 tile，也即一小组激活值，会针对这些激活值专门预计算一个 LUT，并在权重列之间复用。该方法在计算过程中动态地为每个 tile 构建 LUT，从而在保持效率的同时将表大小控制在可接受范围内。图 3 展示了一个基本示例：一个小 tile 包含 $1 \times 4$ 的 FP16 激活和 $4 \times N$ 的 INT1 权重。对于长度为 4 的激活向量，其查找表大小为 16。在这种情况下，每个长度为 4 的点积结果都可以通过一次简单的表查找获得。该表可以复用 N 次，对于大规模权重矩阵，这非常关键。更长的激活向量或更高比特宽度的权重将需要按比例更大的查找表。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/613d4c82141647464142109cfb293632cdc0cd556939ea9c04b64b2827ae47da.jpg)
 图 4：从 LLAMA2-70B 中提取的 M0–M3 形状下的 mpGEMM kernel 性能。$W_{INT4}A_{FP16}$ 表示 INT4 权重和 FP16 激活。LUT 基软件 kernel（LUT-GEMM）在 A100 GPU 上性能弱于基于反量化的 kernel（CUTLASS）。

# 2.3 Gaps in Current LUT-based Solutions

基于 LUT 的 mpGEMM 通过用简单的表查找替代反量化与乘法运算，并减少加法运算，理论上具有明显优势。但现有的软件和硬件实现仍面临诸多挑战和缺口：

**软件侧 LUT kernel。** LUT 基 mpGEMM 的软件 kernel 通常受到指令支持有限和内存访问低效的困扰。主要体现在两个方面：第一，GPU 对表查找的指令支持有限。目前最有效的指令是 prmt（permute），但其宽度有限，无法在单条指令中完成一次完整的表查找，降低了吞吐率。第二，查找表的存储位置对性能影响很大。将 LUT 存放在寄存器文件中，由于 LUT 方法具有“广播”的本质，表项会在多个线程之间被大量复制，面对较大表时会导致寄存器溢出。相反，如果将查找表放在共享内存中，则由于同一 warp 内线程对表的访问呈随机模式，往往容易产生 bank 冲突，从而严重影响内存带宽。上述问题导致 LUT kernel 在现有 LLM 推理硬件（如 GPU）上，相比基于反量化的 kernel 效果欠佳。图 4 对比了 A100 GPU 上，文献 [53] 中的 LUT 基 mpGEMM kernel 与 CUTLASS [50] 中基于反量化的 mpGEMM kernel 的性能。结果显示，后者始终优于前者。尤其是在大 batch 场景下，LUT kernel 受到表访问开销的严重影响，性能会下降几个数量级。“Seg. Error” 标注则表明我们在 LUT-GEMM [53] 中观察到了段错误。

**硬件侧 LUT 加速器。** 乍一看，定制的 LUT 硬件因结构简单（只需寄存器用于表存储和多路选择器用于查找）而似乎能带来可观的效率收益。但我们的研究表明，传统 LUT 硬件设计并没有充分兑现这些收益。图 5 展示了一个用于 mpGEMM 的传统三步 LUT 硬件方案：表预计算、表查找和部分和累加。这其中存在大量挑战与未充分探索的设计因素，会显著影响整体性能：（1）**表预计算与存储。** 预计算查找表可能占用大量存储资源，引入额外的面积与时延开销，从而吞噬本应获得的效率收益。（2）**比特宽灵活性。** 若要支持多种比特宽组合（如 INT4/2/1 × FP16/FP8/INT8），可能会消耗过多芯片面积。（3）

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/1d3ca6fa61868a15f4503274c1edc99c0a270e581144196ff1455f6cac34261d.jpg)

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/3567fbad3bc63b5099d1ed4188e0732cac88bf96670a892b76850b25cfd9509d.jpg)
 图 5：传统 LUT 硬件的三步流程。表预计算与存储带来了沉重开销。
 图 6：LUT TENSOR CORE 的整体工作流程。

**LUT 分块形状。** 非最优的分块形状会增加存储成本，并限制表复用机会，最终影响性能。（4）**指令与编译。** LUT 基 mpGEMM 需要新的指令集。然而传统编译栈是围绕标准 GEMM 硬件优化的，难以高效地映射与调度新的指令集，从而使其与现有软件栈的集成更加复杂。

# 3 LUT TENSOR CORE Design

我们提出 LUT TENSOR CORE，一种旨在应对上述效率、灵活性与集成挑战（§2.3）的软硬件协同设计。图 6 给出了 LUT TENSOR CORE 的总体结构。区别于传统的硬件侧 LUT 方案——其表预计算与表存储引入了较大的硬件开销——LUT TENSOR CORE 通过软件侧优化（§3.1）对表预计算与存储过程进行改造：对输入激活张量的 LUT 预计算通过算子融合进行，而对输入权重张量则通过重解释来支持表存储优化。在硬件端，基于 LUT 的 Tensor Core 微架构（§3.2）为 mpGEMM 计算提供高效实现，并支持多种比特宽数据类型。为了将 LUT TENSOR CORE 集成进现有深度学习生态，LUT TENSOR CORE 设计了 LMMA 指令集，用来暴露基于 LUT 的 Tensor Core 供 mpGEMM 编程，并实现了完整的编译栈来调度端到端 LLM 执行（§3.3）。

# 3.1 Software-based Table Optimization

如 §2 所述，基于 LUT 的 mpGEMM 需要额外的表预计算过程以及用于存储预计算结果的存储空间。朴素地看，对于长度为 $K$ 的激活向量、权重比特宽为 $W_{BIT}$ 的情况，其预计算点积查找表需要 $(2^{W_{BIT}})^K$ 个表项。

对于每个激活元素，将其与 $W_{BIT}$ 比特权重相乘有 $2^{W_{BIT}}$ 种可能结果，这些结果构成该激活元素的预计算表。因此，对于长度为 $K$ 的激活向量，预计算表共有 $(2^{W_{BIT}})^K$ 个表项。图 3 中展示了在 $K = 4$、$W_{BIT} = 1$ 时，大小为 $2^4$ 的查找表。

一种常用优化是**比特串行** [27]，将一个 $W$ 比特整数表示为 $W$ 个 1 比特整数，并通过移位实现基于 1 比特整数的乘法。这种范式可以在 1 比特表上复用预计算，从而将表大小降低为 $2^{K}$。尽管如此，即便是这种缩小后的表大小，在硬件上也带来不小开销。LUT TENSOR CORE 提出了**数据流图（DFG）变换与算子融合**来消除表预计算开销，同时通过**权重重解释与表量化**来减少表大小。

## 3.1.1 Precomputing lookup table with DFG transformation and operator fusion

LUT 基 mpGEMM 需要预先计算高精度激活与一组低精度权重之间的点积结果，并用查找表保存，以供后续查找使用。传统实现通常将预计算单元放置在 LUT 单元旁边，为每个 LUT 单元实时进行表预计算。这种做法由于存在大量重复计算，会引入显著的硬件成本。例如，对 OPT-175B 中 [4096,12288]  $\times$  [12288,12288] 的 GEMM 而言，一个朴素的直接预计算单元会在阵列大小为 $N = 4$ 的 LUT 基 Tensor Core 内共享表。在这种配置下，每个表会在整个计算过程中，由不同的 LUT 单元重复计算 12288/4 = 3072 次，计算负担极重。

为缓解这一效率问题，我们首先对 DFG 进行变换，将表预计算拆分为一个独立 kernel，从而使预计算可以仅执行一次，然后广播给所有 LUT 单元。该改动将预计算开销降低了数百倍，使其可以由现有的向量单元（如 CUDA Core）完成。为摊销广播带来的额外内存流量，LUT TENSOR CORE 将预计算算子与前一算子进行融合，利用前一算子逐元素（element-wise）的计算模式，如图 6 所示，详细内容见 §3.3.2。该融合减少了内存访问，使预计算开销几乎降为零，其效果在 §4.6.1 中进行了评估。

## 3.1.2 Reinterpreting weight for table symmetrization

预计算长度为 $K$ 的激活向量所需的 $2^{K}$ 大小的查找表，在存储和查找访问两方面都代价不菲。为此，我们观察并利用了整数表示的对称性属性。

假设原量化权重表示为：

$$
 r _ {w} = s _ {w} \left(q _ {w} - z _ {w}\right) \tag {1}
$$

其中，$r_w$ 为实值权重，$s_w$ 为缩放因子，$z_w$ 为偏置，$q_w$ 为 $K$ 比特整数表示。

我们的目标是对 $q_{w}$ 做映射，在保持数学等价的前提下，使其在零点附近呈对称分布。为此，我们需要同时调整 $s_{w}$ 和 $z_{w}$。当将无符号整数 $q_{w}$ 映射为关于零点对称的整数时，需要进行如下变换：

$$
 q _ {w} ^ {\prime} = 2 q _ {w} - \left(2 ^ {K} - 1\right), \quad s _ {w} ^ {\prime} = s _ {w} / 2, \quad z _ {w} ^ {\prime} = 2 z _ {w} + 1 - 2 ^ {K} \tag {2}
$$

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/72e3cba70b5bc2463faaac08a7d497bd7ff9ed0c68e85c28c1f22824ea66d7d4.jpg)
 图 7：将 0、1 重新解释为 -1、1 以实现对称性，从而将表大小减半。

图 7 说明了这一过程，以 4 比特无符号整数的转换为例。通过计算 $s_w'$ 和 $z_w'$，$q_w'$ 被映射为 ${0, 1, \dots, 14, 15}$ 到 ${-15, -13, \dots, 13, 15}$，实现了关于零点的对称。

接下来，点积可以写成：

$$
 D P = \Sigma A c t _ {i} s _ {w} \left(q _ {w i} - z _ {w}\right) = \Sigma A c t _ {i} s _ {w} ^ {\prime} \left(q _ {w i} ^ {\prime} - z _ {w} ^ {\prime}\right) \tag {3}
$$

其中 $DP$ 为点积结果，$Act_i$ 为激活值。因此，量化过程与之前保持一致，只是在离线阶段多了一步：将权重的 $s_w(q_{wi} - z_w)$ 映射为 $s_w'(q_{wi}' - z_w')$。考虑一个二进制表示为 $W_3W_2W_1W_0 = 0.010$ 的权重向量，与变量 $A, B, C, D$ 的点积。最初，二进制值 ${0',1}$ 被解释为 ${0,1}$。假设 $s_w = 2$，$z = 1/2$，则计算如下：

$$
 \begin{array}{l} D P = \sum A c t _ {i} s _ {w} \left(q _ {w i} - z _ {w}\right) \ = A \cdot 2 \cdot (0 - 0. 5) + B \cdot 2 \cdot (1 - 0. 5) \ + C \cdot 2 \cdot (1 - 0. 5) + D \cdot 2 \cdot (1 - 0. 5) \ = - A + B - C - D \ \end{array}
$$

经重解释后，二进制 ${^{\prime}0^{\prime},1^{\prime}}$ 被重新映射为 ${-1,1}$，并调整缩放因子 $s_w' = 1$、偏置 $z_w' = 0$。新的计算为：

$$
 \begin{array}{l} D P = \sum A c t _ {i} s _ {w} ^ {\prime} \left(q _ {w i} ^ {\prime} - z _ {w} ^ {\prime}\right) \ = A \cdot 1 \cdot (- 1 - 0) + B \cdot 1 \cdot (1 - 0) \ + C \cdot 1 \cdot (- 1 - 0) + D \cdot 1 \cdot (- 1 - 0) \ = - A + B - C - D \ \end{array}
$$

可以看到，两种表达形式在数学上是等价的。由于表项以零点为中心呈对称分布，查找表具有类似奇函数的性质。假设索引是一个 4 比特值 $W_{3}W_{2}W_{1}W_{0}$，朴素实现需要 $2^{4} = 16$ 个表项。然而，我们可以观察到类似奇函数的关系成立：

$$
\mathrm {L U T} \left[ W _ {3} W _ {2} W _ {1} W _ {0} \right] = - \mathrm {L U T} [ \sim \left(W _ {3} W _ {2} W _ {1} W _ {0}\right) ] \tag {4}
$$

因此，LUT 表项数可以减半为 $2^{4 - 1} = 8$，关系式变为：

$$
\operatorname {L U T} \left[ W _ {3} W _ {2} W _ {1} W _ {0} \right] = \left{ \begin{array}{l l} - \operatorname {L U T} \left[ \sim \left(W _ {2} W _ {1} W _ {0}\right) \right], & \text {i f} W _ {3} = 1 \ \operatorname {L U T} \left[ W _ {2} W _ {1} W _ {0} \right], & \text {i f} W _ {3} = 0 \end{array} \right. \tag {5}
$$

这里，$\sim$ 表示按位取反操作。因此，对于长度为 $K$ 的激活向量，表对称化可以将表长度压缩为 $2^{K - 1}$。表大小不仅影响预计算阶段所需的计算量，还决定了多路选择器的规模。此外，每个表项还需要广播到 $N$ 个处理单元（PE），通常为 64 或 128，用于完成点积计算。上述优化显著降低了广播开销和多路选择器的选择开销。注意，式（5）中的 $W_{3}W_{2}W_{1}W_{0}$ 是静态权重，比特级的取反可以在离线阶段完成，以进一步简化设计：

$$
\mathrm {L U T} \left[ W _ {3} ^ {\prime} W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right] = \left{ \begin{array}{l l} - \mathrm {L U T} \left[ W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right], & \text {i f} W _ {3} ^ {\prime} = 1 \ \mathrm {L U T} \left[ W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right], & \text {i f} W _ {3} ^ {\prime} = 0 \end{array} \right. \tag {6}
$$

这一简化可以在电路设计中去掉取反逻辑，其详细设计将在 §3.2 中介绍。

## 3.1.3 Table quantization

对于 FP32 或 FP16 等高精度激活，我们采用**表量化**技术，将预计算表项转换为较低且统一的精度，例如 INT8。该方法一方面通过支持多种激活精度提高灵活性，另一方面通过减小表大小提升效率。

与传统的激活量化相比，表量化允许在表预计算阶段进行更动态、更细粒度的量化。例如，当分组大小为 4 个激活元素时，我们会对每个包含 8 个预计算点积结果的表进行单独量化。正如 § 4.6.2 中的实验所示，基于 INT8 的表量化在保持高精度的同时简化了硬件设计，证明了该方法的有效性。

# 3.2 LUT-based Tensor Core Microarchitecture

## 3.2.1 Simplified LUT unit design with bit-serial

借助软件侧的预计算融合与权重重解释，每个独立 LUT 单元所需的硬件成本得以降低。图 8 展示了我们的 LUT 单元设计。与朴素设计相比，用于 LUT 存储的寄存器数量以及表广播与多路选择器的开销均降低了一半。如式（6）所示，电路中的比特级取反逻辑可以从每个 LUT 单元中移除，进一步提升效率。为支持灵活的权重比特宽，我们采用了一种**比特串行电路架构** [27, 74]。这种设计将权重比特宽度映射为 W_BIT 个周期，以串行方式处理不同比特宽。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/d25b78db60b1fd07da945f33f95233542e6553037f030148f8e40dbd8b5515e3.jpg)
 图 8：带比特串行的优化 LUT 单元。

## 3.2.2 Elongated LUT tiling

在 LUT 基 Tensor Core 中，$M$、$N$ 和 $K$ 维度的选择对性能至关重要；若沿用传统基于 MAC 的 Tensor Core 的形状，可能得到次优性能。如图 9 所示，一个 MNK tile 的 LUT 阵列包含 $M$ 个查找表、$N$ 组权重以及 $M*N$ 个基于多路选择器的单元。每个表有 $M \times 2^{K-1}$ 个表项，每个表项需要广播到 $N$ 个多路选择器单元。每组 Grouped Binary Weights 包含 $K$ 个比特，需要广播到 $M$ 个多路选择器单元作为选择信号。总表大小由下式给出：

$$
 \text {T o t a l} = M \times 2 ^ {K - 1} \times \text {L U T} _ {\text {B I T}} \tag {7}
$$

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/99aa0a6b3c361ae4e0f9bc83a634ba1131978d4d732d4e5ce9cee6e8a63241cc.jpg)
 图 9：LUT 基 Tensor Core 的拉长 $MNK$ 分块。为最大化表复用，LUT 基 Tensor Core 需要更大的 $N$（例如 64/128），以及适当的 $K$（例如 4）以保持表大小成本可接受。

而 Grouped Binary Weights 的大小为：

$$
 \text {G r o u p e d B i n a r y W e i g h t s S i z e} = K \times N \times \mathrm {W} _ {\text {B I T}} \tag {8}
$$

其中 LUT_BIT 为 LUT 表项的比特宽，W_BIT 为权重比特宽。

LUT 基 Tensor Core 从**拉长分块形状**中获益。$K$ 越大，表项数量呈指数增长；而 $N$ 则决定每个表项可以被多少个多路选择器单元复用。最优配置需要在 $K$、较大的 $N$ 和较小的 $M$ 之间取得平衡，这与传统 GPU Tensor Core 的设计不同。此外，分块形状也会影响 I/O 流量，更趋近正方形的分块形状可以降低数据搬运开销。在 §4.2.2 中，我们对 $MNK$ 分块进行了设计空间探索，验证了拉长分块在效率上的优势。

# 3.3 Instruction and Compilation

为将 LUT TENSOR CORE 集成到现有 GPU 架构与生态中，我们提出了一套新的指令集，并基于 tile 级 DNN 编译器 [5, 62, 84] 构建了相应的编译栈。

## 3.3.1 LUT-based MMA instructions

为支持基于 LUT 的 Tensor Core 编程，我们定义了一组 LMMA 指令，将 GPU 中的 MMA 指令集进行扩展。

Imma.{M}{N}{K}.  ${A_{\mathrm{dtype}}} {W_{\mathrm{dtype}}} {Accum_{\mathrm{dtype}}} {O_{\mathrm{dtype}}}$

上式给出了 LMMA 指令的格式，其形式类似于 MMA。具体来说，$M,N,K$ 表示 LUT 基 Tensor Core 的形状；$A_{dtype},W_{dtype},Accum_{dtype}$ 和 $O_{dtype}$ 则分别表示输入、累加与输出的数据类型。与 MMA 指令类似，每条 LMMA 指令会被调度到一个 warp 的线程中执行。每个 warp 计算如下公式：$O_{dtype}[M,N] = A_{dtype}[M,K]\times W_{dtype}[N,K] + Accum_{dtype}[M,N]$。

## 3.3.2 Compilation support and optimizations

我们基于 TVM [5]、Roller [84] 和 Welder [62] 实现了 LUT-mpGEMM kernel 的自动生成以及基于 LUT 基 Tensor Core 的端到端 LLM 编译。编译栈主要包括以下几个关键方面，图 10 展示了在 LLAMA 模型上的一个编译示例：

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/cda3f23ed307cb24bb9fc632adf498395c0b589d1101fb35fa7ca4cfe1f5edc6.jpg)
 图 10：LUT-mpGEMM 的编译流程。整体数据流类似 cutlass [50]。采用拉长 tile 以加强数据复用。

- **DFG 变换。** 给定 DFG 形式的模型，我们将混合精度 GEMM 算子拆解为“预计算算子”和“LUT-mpGEMM 算子”。这一变换作为图优化 pass 集成在 Welder [62] 中。
- **算子融合。** 算子融合是编译器中广泛使用的技术，用于通过减少内存流量与运行时开销来优化端到端模型执行。我们在 Welder 中复用算子融合能力，将预计算算子与 LUT-mpGEMM 算子注册为需要 tile 级表示的算子。如图 10 所示，逐元素的预计算算子会与前一个逐元素算子融合。
- **LUT-mpGEMM 调度。** 对 LUT-mpGEMM 算子进行调度时，需要在内存层次上的分块策略上进行精细设计，以获得最佳性能。传统 GEMM 分块策略 [5, 82, 84] 假定激活与权重使用相同数据类型，而 mpGEMM 则在激活与权重的数据类型上存在差异，从而影响内存传输模式。为解决这一问题，我们以**内存大小**而非**形状**来描述分块，并在 Roller 的 rTile [84] 接口中注册 LMMA 的形状与分块计算方式，以便自动搜索最优配置。
- **代码生成。** 在调度计划确定后，我们使用 TVM 完成代码生成。具体而言，我们将 LMMA 指令注册为 TVM 的 intrinsic，TVM 会依据调度策略生成包含 LMMA 指令的 kernel 代码。

# 4 Evaluation

本节中，我们对 LUT TENSOR CORE 进行评估，以验证其在加速低比特 LLM 推理方面的效率。首先，我们通过详细的 PPA 基准测试评估其硬件效率收益（$\S 4.2$）。然后，我们在 kernel 级别上评估 mpGEMM 的加速效果（$\S 4.3$）。接着，我们对常用 LLM 进行端到端推理评估，以展示其实际性能提升（$\S 4.4$）。最后，我们将 LUT TENSOR CORE 与先前的 LUT 基工作进行对比（§4.5），并分析软件侧优化的效果，重点关注表预计算融合与表量化（§4.6）。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/cb47d66b95f11f0a1500d25e6501c1040e6e898b108bf57766db392f038fe384.jpg)
 图 11：沿 K 轴对 LUT 基点积单元进行设计空间探索。整体上 $K = 4$ 为最优选择。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/41af76ca18834ca16cca5db5bf8b692c24d52021a27ea5124eb47858e06521b6.jpg)
 图 12：MAC/ADD/LUT 基 DP4 实现的 PPA 对比。我们的 LUT 基 DP4 单元在计算密度与功耗上具有优势。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/9c7cdae72d1165f5cb359d73c3b3cbf7a56331b575f58fea9fee05bac9d53fae.jpg)
 图 13：在 $W_{\mathrm{INTX}} \times A_{\mathrm{FP16}}$ 场景下，不同权重比特宽度下 MAC、ADD 与 LUT 基 DP4 单元的面积对比。传统的 LUT 实现并不具有面积优势。

## 4.1 Experimental Setup and Methodology

### 4.1.1 Hardware PPA benchmarks

我们将 LUT 基 Tensor Core 与两个基线进行对比：基于乘加（MAC）的 Tensor Core 与基于加法（ADD）的 Tensor Core。MAC 代表当前 GPU 中典型的设计，需要通过反量化来支持 mpGEMM。ADD 则采用文献 [27] 中提出的比特串行计算来支持 mpGEMM，即每一比特权重对应一次加法操作。我们用 Verilog 实现了 LUT 基 Tensor Core 与两个基线设计，并采用 Synopsys Design Compiler [63] 与 TSMC 28nm 工艺库进行电路综合和 PPA 数据生成。为保证比较公平，我们在面向 1GHz 的目标频率下，统一使用 DC 的中等优化力度。

### 4.1.2 Kernel-level evaluation

在 mpGEMM 的 kernel 级评估中，我们以 NVIDIA A100 GPU 为基线，并采用开源的最先进 GPU 模拟器 Accel-Sim [30]。通过修改 Accel-Sim 中的配置与 trace 文件，我们可以同时模拟原始 A100 以及配备 LUT TENSOR Core 的 A100。

### 4.1.3 Model end-to-end evaluation and analysis

为将评估扩展到真实 LLM，我们选取了四个广泛使用的开源 LLM：LLAMA-2 [65]、OPT [80]、BLOOM [36] 和 BitNet [68]。由于在大 trace 文件下，Accel-Sim 的仿真速度极慢，不适合进行端到端 LLM 实验，我们基于 tile 级别开发了一个模拟器，以支持端到端推理评估，具体见 §4.4。

## 4.2 Hardware PPA Benchmarks

### 4.2.1 Dot product unit microbenchmark

在这一实验中，我们将 $M$ 和 $N$ 固定为 1，并改变 $K$（即长度为 $K$ 的向量点积单元）来探索其对计算密度的影响。过大的 $K$ 会导致查找表项数指数级增长，而过小的 $K$ 则意味着仍有 $1 / K$ 的计算需要由加法器完成。如图 11 所示，我们发现整型运算在 $K = 4$ 时计算密度达到峰值，而浮点运算在 $K = 5$ 时表现最佳，但在 $K = 4$ 时表现也很好。因此，我们在后续所有 LUT 基设计中采用 $K = 4$。

我们在多种数据格式下，对基于 MAC、ADD 以及 LUT 的点积实现进行 PPA 基准测试。包括基于 MAC 的统一精度实现（如 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$），以及在 ADD 和 LUT 两种方法下的混合精度实现（如 $W_{\mathrm{INT1}}A_{\mathrm{FP8}}$）。如图 12 所示，在 $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 场景下，LUT 基方法的计算密度可达 61.55 TFLOPs/mm²，而传统的 MAC 实现（$W_{\mathrm{FP16}}A_{\mathrm{FP16}}$）仅为 3.39 TFLOPs/mm²。在功耗效率方面，LUT 基方法同样显著优于其他方案。

此外，我们还对 $W_{\mathrm{INTX}} \times A_{\mathrm{FP16}}$ 的 DP4 单元进行了权重比特宽度缩放实验，分别评估 MAC、ADD 与 LUT 实现。实验采用的 Tensor Core 的 N 维度设置为 4，与 A100 的配置一致。如图 13 所示，当权重比特宽超过 2 比特时，传统 LUT 实现的面积并不优于 MAC 基线。其面积效率瓶颈主要来自表预计算与存储开销。ADD 实现也仅在 1 比特与 2 比特场景下优于 MAC。通过软硬件协同设计，LUT TENSOR Core 在权重比特宽度最高达到 6 比特时，都能优于上述基线，并在面积效率上显著改善传统 LUT 实现。

### 4.2.2 Tensor Core benchmark

我们将评估从点积单元扩展到 Tensor Core 级别，并进行设计空间探索以确定最优 $MNK$ 配置。为匹配 A100 中 INT8 Tensor Core 的配置 $M, N, K = 8, 4, 16$，我们将阵列大小设置为 $M \times N \times K = 512$。实验中采用了多种激活数据类型，包括 $A_{\mathrm{FP16}}$、$A_{\mathrm{INT16}}$、$A_{\mathrm{FP8}}$ 与 $A_{\mathrm{INT8}}$，以及多种权重比特宽度，如 $W_{\mathrm{INT1}}$、$W_{\mathrm{INT2}}$ 与 $W_{\mathrm{INT4}}$。我们将 LUT 基方法与基于 MAC 和 ADD 的实现进行了对比。

如图 14 所示，我们对不同的 $M, N, K$ 组合进行 sweeping 以探索设计空间，确保不同方法之间比较公平。图中 y 轴为“面积”，x 轴为“功耗”。虚线表示在各个设计方法中，其 Area×Power 最小点所在的等值线。结果表明，在 12 组不同激活格式与权重比特宽度的实验中，LUT 基方法在除 $W_{\mathrm{INT8}}A_{\mathrm{INT4}}$ 情况之外，均实现了最小面积与最低功耗。尤其是对于 1 比特权重，LUT 基方法在功耗与面积上相对于 MAC 基 Tensor Core 设计实现了 $4\times -6\times$ 的降低。我们最终确定 LUT 基 Tensor Core 的最优 $MNK$ 配置为 $M2N64K4$。这主要是因为激活为高比特、权重为低比特；在整体比特宽上，M 维对应 $2\times 16 = 32$ 比特，N 维对应 $64\times 1 = 64$ 比特，总体比特配置仍接近一个方阵。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/fa41a61936e8799ab32d669963cc5bed17a4d4e7dbc9272a1172e674110e9865.jpg)
 图 14：面向 mpGEMM 的 LUT/ADD/MAC 基 Tensor Core 实现的 PPA 对比。

## 4.3 mpGEMM Kernel-level Evaluation

我们利用最先进 GPU 模拟器 Accel-Sim 验证 LUT TENSOR CORE 在 mpGEMM 运算上的效率以及其与现有 GPU 架构的兼容性。mpGEMM 的矩阵形状从 LLAMA2-13B 中提取，为 $M = 2048$、$N = 27648$、$K = 5120$。mpGEMM 的数据流采用类似 CUTLASS 的输出驻留（output-stationary）方案，并对分块形状进行优化以实现高效数据复用。例如，在 $W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 场景下，一个较优的分块方案为 Thread Block tile [128, 512, 32]，Warp tile [64, 256, 32]。

如图 15 所示，在 mpGEMM 运算中，LUT 基 Tensor Core 的性能优于传统的 MAC 基 Tensor Core。每个子图最左边两根柱分别表示 A100 的理想峰值性能与基于 cuBLAS 的实测性能；其余柱则表示 LUT 基结果：理想峰值性能、模拟性能以及增大寄存器容量后的模拟性能。增大寄存器容量是为缓解寄存器不足带来的瓶颈，因为寄存器不足会限制分块规模，并将性能限制在内存带宽上。例如，在 $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 情况下，LUT 基方法在仅使用 MAC 基 Tensor Core 14.3% 面积的情况下，就能提供略高的 mpGEMM 性能。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/a78837e9c39a35412a5fde5e62a2afef1ad884fdcb91f0ea7e2b02a0913b4f9b.jpg)
 图 15：在 $A_{\mathrm{FP16}}$ 与 $A_{\mathrm{INT8}}$ Tensor Core 设计上的 Accel-Sim 运行时间与面积。符号 $\times$ 表示相对于 $1 \times$ 基线的 Tensor Core 阵列规模，其中 $1 \times$ 对应于 NVIDIA A100 中 $M \times N \times K = 512$ 的阵列规模。

## 4.4 Model End-to-End Evaluation

尽管 Accel-Sim 可以提供详细的体系结构级仿真，但其速度极慢，约为真实硬件的五百万分之一，会将 A100 GPU 上 10 秒的任务变为约 579 天的仿真时间，并产生超过 79TB 的 trace 文件。

为解决这一问题，我们开发了一个端到端模拟器，用于以 tile 级粒度进行快速且准确的仿真。我们的关键洞见是：在 LLM 场景中，高度优化的、几乎无停顿的大型 GPU kernel 的行为可以被视作“加速器”，这一观点也得到了 NVIDIA 在 NVAS [67] 中的支持，即将 GPU 仿真“哲学性地”视为“动态互相作用的 roofline 组件”，而非“周期级的进程”。因此，我们借鉴 Timeloop [52]、Maestro [34] 和 Tileflow [83] 等已有加速器建模框架中的分析方法，构建了一个 tile 级 GPU 模拟器。该工具支持对数据流、内存带宽、计算资源以及算子融合进行细致而准确的评估。我们计划在未来将该模拟器开源。

### 4.4.1 Simulator accuracy evaluation

如图 16 所示，我们在 OPT-175B、BLOOM-176B 和 LLAMA2-70B 上对单层进行多种配置的仿真，并在 A100 与 RTX 3090 GPU 上验证模拟器的准确性。我们的模拟器相对于真实 GPU 性能的平均绝对百分误差仅为 $5.21%$，且仿真速度远快于 Accel-Sim。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/beb7a358017b517a90bcb0bd0bbaddbd2abd71e9787b05da218a05fceb8bcfe5.jpg)
 图 16：端到端模拟器准确性的评估结果。

### 4.4.2 End-to-end inference simulation results

图 17 展示了 OPT、BLOOM 与 LLAMA 模型的基准测试结果。实验表明，相比传统的 $W_{\mathrm{FP16}} A_{\mathrm{FP16}}$ Tensor Core，LUT TENSOR CORE 在端到端推理中最高可以获得 $8.2 \times$ 的速度提升，且占用更少面积。值得注意的是，即便在 $8 \times$ 配置下，LUT TENSOR CORE 的面积也仅为传统 $W_{\mathrm{FP16}} A_{\mathrm{FP16}}$ MAC 基 Tensor Core 的 38.3%。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/5815fcb8bd380aeae9b497af2396e0efdf5c9ebb74b6c2a6fcda9698891b6d0b.jpg)
 图 17：在 LLM 上的端到端仿真结果（A100 与 3090）。R：真实 GPU，M：模型仿真，DRM：双寄存器模型仿真。

### 4.4.3 Overall comparison

如表 1 所示，在配备 LUT 与 BitNet 的 A100 上，我们可以在仅使用原 Tensor Core 38.3% 面积的情况下，实现最高 $5.51 \times$ 的推理速度加速。这带来了最高 $20.9 \times$ 的计算密度提升与 $11.2 \times$ 的能效提升，这得益于量化 LUT 表以及通过软硬件协同设计高度优化的 LUT 电路。与 H100 上原生的 $W_{FP8}A_{FP8}$ Tensor Core 相比，LUT Tensor Core 的面积效率最高可提升 $2.02 \times$。

![img](https://cdn-mineru.openxlab.org.cn/result/2025-12-08/c91186c8-741d-4809-9e5d-3d1c25299580/be146d07f9835db8868c3f0e76f3268245bbc755907108c73fe9f7c82c221016.jpg)
 图 18：LUT TENSOR CORE 与 LUT 基软件工作 LUT-GEMM [53] 在 GEMM 与 GEMV 上的对比。

# 4.5 Compared to Prior Works

## 4.5 与现有工作的比较

4.5.1 基于 LUT 的软件。LUT-GEMM [53] 和 T-MAC [71] 分别是此前在 GPU 和 CPU 上的 SOTA（state-of-the-art）LUT 软件方案。由于 T-MAC 面向 CPU 设计，我们在 GPU 场景下采用 LUT-GEMM 作为更合适的对比基线。LUT TENSOR CORE 的配置仅占传统 FP16 Tensor Core 面积的 $57.2\%$。图 18 展示了 LUT TENSOR CORE 和 LUT-GEMM 相对于 A100 上 $W_{FP16}A_{FP16}$ cuBLAS 的加速比对比结果。LUT-GEMM 仅在 GEMV 场景中带来性能提升，而在 GEMM 场景中相较于 cuBLAS 慢了数十倍。与纯软件方案 LUT-GEMM 相比，LUT TENSOR CORE 在 GEMV 上最高可实现 $1.42\times$ 的加速，在 GEMM 上可实现高达 $72.2\times$ 的加速。

4.5.2 基于 LUT 的硬件。UNPU [38] 是面向 DNN 工作负载的 SOTA LUT 硬件加速器。由于缺乏公开代码，我们基于其论文对 UNPU 进行了重新实现，并在此基础上进行优化以保证公平对比。我们在 Tensor Core 级别上对 UNPU 和 LUT TENSOR CORE 进行了 DSE（设计空间探索）。以 Tensor Core 配置为 $M\times N\times K = 512$，并采用 $W_{\mathrm{INT8}}A_{\mathrm{INT2}}$ 为例，通过消融实验评估各项优化带来的影响。表 2 显示，多比特权重的重新解释以及对称化可将计算密度和能效提升约 $30\%$。进一步的优化，包括离线权重重解释、消除求反电路、DFG 变换以及内核融合，使 LUT TENSOR CORE 相比 UNPU 在这些指标上整体实现了 $1.44\times$ 的提升。

4.5.3 量化 DNN 加速器。既有工作如 Ant [19]、FIGNA [25] 和 Mokey [78]，主要围绕专用低比特精度（例如 int8×int8 或 int4×fp16）设计基于 MAC 的 PE。尽管针对特定数据类型时效率较高，这类设计在适配不同精度需求时缺乏足够灵活性：要么在转为更低精度时牺牲模型精度，要么在转向更高精度时错失潜在的效率收益。相比之下，我们采用 LUT 方案，通过不同的 LMMA 指令同时支持 1–4 比特 INT 权重和 FP/INT 16/8 激活，覆盖了大多数低比特 LLM 的使用场景。表 3 将 LUT TENSOR CORE 与其他加速器进行了对比。

# 4.6 Software Optimization Analysis

## 4.6 软件优化分析

4.6.1 查找表预计算融合分析。表 4 展示了将预计算融入 DNN 编译器 Welder [62] 后的效果。Welder 通过优化算子融合来提升推理性能。本实验在 OPT-175B、BLOOM-176B 和 LLAMA2-70B 模型的单层上进行，分别考虑批量预取（batch prefetching）和解码（decoding）两种配置。最初，将预计算放在 CUDA Cores 上执行会带来平均 $16.47\%$ 和 $24.41\%$ 的额外开销。然而，当在 Welder 的搜索空间中将预计算视作一个独立算子后，该开销被降低到 $2.62\%$ 和 $2.52\%$，在整体执行时间中几乎可以忽略。

4.6.2 查找表量化分析。为了评估表量化的影响，我们在具有 2 比特量化权重的 LLAMA2-7B 模型 [65] 上进行了对比实验。首行数据是原始的 $W_{FP16}A_{FP16}$ LLAMA2-7B 模型；第二行是 BitNet-b1.58 论文 [44] 中报告的 LLAMA-3B 模型。随后的 2 比特模型基于 BitDistiller [14] 得到，该框架是一个开源的 QAT（量化感知训练）框架，用于提升超低比特 LLM 的性能。其原始配置为 INT2 权重和 FP16 激活。在 BitDistiller 的开源代码基础上，我们进一步实现了基于 LUT 的 mpGEMM 的 INT8 查找表量化。评测指标与 BitDistiller 保持一致，包括在 WikiText-2 数据集 [46] 上的困惑度（perplexity）、在 MMLU [20] 上的 5-shot 准确率，以及在多个任务 [2, 7, 48, 59, 79] 上的 zero-shot 准确率。实证结果汇总在表 5 中。第二行中的 “N/A” 表示 [44] 中未报告 MMLU 准确率。尽管 2 比特权重量化相较原始 $W_{FP16}A_{FP16}$ LLAMA2-7B 模型略有劣化，但其性能仍优于 $W_{FP16}A_{FP16}$ 的 LLAMA-3B 模型。值得注意的是，引入 INT8 查找表量化并未损失模型精度，仅在困惑度上有极小的变化，同时任务准确率还略有提升，这可能与量化带来的正则化效应有关。

Table 1: Overall comparison.  

表 1：总体比较。  

<table><tr><td>HW. Config.</td><td>Model</td><td>BS1
SEQ2048
Latency</td><td>BS1024
SEQ1
Latency</td><td>Peak
Perf.</td><td>TC. Area
Per SM</td><td>TC. Compute
Density</td><td>TC. Energy
Efficiency</td></tr><tr><td>A100† FP16 TC.</td><td>LLAMA 3B
(WFP16AFP16)</td><td>49.7%</td><td>106.71ms</td><td>41.15ms</td><td>312 TFLOPs</td><td>0.975mm²</td><td>2.96 TFLOPs/mm²</td></tr><tr><td>A100† INT8 TC</td><td>BitNet b1.58 3B
(WINT2AINT8)</td><td>49.4%</td><td>67.06ms</td><td>21.70ms</td><td>624 TOPs</td><td>0.312mm²</td><td>17.73 TOPs/mm²</td></tr><tr><td>A100†-LUT-4X*</td><td>BitNet b1.58 3B
(WINT2AINT8)</td><td>49.4%</td><td>42.49ms</td><td>11.41ms</td><td>1248 TOPs</td><td>0.187mm²</td><td>61.84 TOPs/mm²</td></tr><tr><td>A100†-LUT-8X*</td><td>BitNet b1.58 3B
(WINT2AINT8)</td><td>49.4%</td><td>38.02ms</td><td>7.47ms</td><td>2496 TOPs</td><td>0.373mm²</td><td>61.95 TOPs/mm²</td></tr><tr><td>H100† FP8 TC</td><td>BitNet b1.58 3B
(WFP8AFP8)</td><td>-</td><td>38.20ms</td><td>12.30ms</td><td>1525 TFLOPs</td><td>0.918mm²</td><td>12.59TFLOPs/mm²</td></tr><tr><td>H100†-LUT-4X*</td><td>BitNet b1.58 3B
(WINT2AFP8)</td><td>-</td><td>28.70ms</td><td>9.90ms</td><td>1525 TFLOPs</td><td>0.488mm²</td><td>23.69TFLOPs/mm²</td></tr><tr><td>H100†-LUT-8X*</td><td>BitNet b1.58 3B
(WINT2AFP8)</td><td>-</td><td>23.48ms</td><td>5.97ms</td><td>3049 TFLOPs</td><td>0.909mm²</td><td>25.40TFLOPs/mm²</td></tr></table>


Due to the lack of public data on A100/H100 Tensor Cores and their 7/4nm processes,  $\dagger$  indicates that the data are normalized to 28nm at 1.41GHz and optimized to the best of our ability for fair comparison. -LUT* denotes LUT TENSOR CORE-equipped GPU with Double Register Modeling.  $\times$  means that of A100 FP16 Tensor Core array size. TC. refers to Tensor Core. Model accuracy for  $A_{FP8}$  is not reported, as BitNet is trained from scratch in the  $A_{INT8}$  format. Prior works [33, 47, 81] show that  $A_{FP8}$  generally outperforms  $A_{INT8}$  in terms of accuracy.

由于缺乏关于 A100/H100 Tensor Core 以及其 7/4nm 工艺的公开数据，$\dagger$ 表示这些数据已在 28nm、1.41GHz 条件下进行归一化，并在我们能力范围内尽可能保证公平比较。-LUT* 表示配备 LUT TENSOR CORE 并采用 Double Register Modeling 的 GPU。$\times$ 表示相对于 A100 FP16 Tensor Core 阵列规模的倍数。TC. 指 Tensor Core。对于 $A_{FP8}$，未报告模型精度，因为 BitNet 是在 $A_{INT8}$ 格式下从零开始训练的。先前工作 [33, 47, 81] 表明，就精度而言，$A_{FP8}$ 通常优于 $A_{INT8}$。

Table 2: LUT TENSOR CORE compared with UNPU [38]:  ${W}_{\mathrm{{INT}}2}{A}_{\mathrm{{INT}}8}$  Tensor Core case.  

表 2：LUT TENSOR CORE 与 UNPU [38] 的比较：${W}_{\mathrm{{INT}}2}{A}_{\mathrm{{INT}}8}$ Tensor Core 场景。  

<table><tr><td>Configuration</td><td>Area (mm2)</td><td>Normalized Compute Intensity</td><td>Power (mW)</td><td>Normalized Power Efficiency</td></tr><tr><td>UNPU (DSE Enabled)</td><td>17,271.71</td><td>1×</td><td>23.39</td><td>1×</td></tr><tr><td>+ Weight Reinterpretation</td><td>13,116.60</td><td>1.317×</td><td>17.98</td><td>1.301×</td></tr><tr><td>+ Negation Circuit Elimination</td><td>12,780.05</td><td>1.351×</td><td>17.37</td><td>1.347×</td></tr><tr><td>+ DFG Trans. + Kernel Fusion =LUT TENSOR CORE (Proposed)</td><td>11,991.29</td><td>1.440×</td><td>16.22</td><td>1.442×</td></tr></table>

Table 3: LUT TENSOR CORE compared with accelerators for quantized models.  

表 3：LUT TENSOR CORE 与若干量化模型加速器的比较。  

<table><tr><td></td><td>UNPU[38]</td><td>Ant[19]</td><td>Mokey[78]</td><td>FIGNA[25]</td><td>LUT TENSOR CORE</td></tr><tr><td>Act. Format</td><td>INT16</td><td>flint4</td><td>FP16/32, INT4</td><td>FP16/32, BF16</td><td>FP/INT8, FP/INT16</td></tr><tr><td>Wgt. Format</td><td>INT1~INT16</td><td>flint4</td><td>INT3/4</td><td>INT4/8</td><td>INT1~INT4</td></tr><tr><td>Compute Engine</td><td>LUT</td><td>flint-flint MAC</td><td>Multi Counter</td><td>Pre-aligned INT MAC</td><td>LUT</td></tr><tr><td>Process</td><td>65nm</td><td>28nm</td><td>65nm</td><td>28nm</td><td>28nm</td></tr><tr><td>PE Energy Eff.</td><td>27TOPs/W @0.9V (WINT1AINT16)</td><td>N/A</td><td>N/A</td><td>2.19× FP16-FP16 (WINT4AFP16)</td><td>63.78TOPs/W @0.9V DC (WINT1AINT8)</td></tr><tr><td>Compiler Stack</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td></tr><tr><td>Eval. Models</td><td>VGG-16, AlexNet</td><td>ResNet, BERT</td><td>BERT, Ro/DeBERTa</td><td>BERT, BLOOM, OPT</td><td>LLAMA, BitNet, BLOOM, OPT</td></tr></table>

a slight increase in task accuracy, which may be attributed to the regularizing effect of quantization.

任务精度的轻微提升可能归因于量化带来的正则化效应。

# 5 Discussion and Limitations

## 5 讨论与局限性

低比特训练与微调。目前，LUT TENSOR CORE 仅用于加速低比特 LLM 的推理阶段。近期工作显示，针对 LLM 的低比特训练和微调需求正在快速上升 [11, 72]。虽然 LUT TENSOR CORE 的 mpGEMM 思路可以直接应用于低比特训练的前向传播阶段，但训练过程的复杂性与稳定性仍要求在反向传播中进行更多高精度计算，涉及梯度、优化器状态等张量和计算，这些尚未完全与低比特格式兼容。此外，训练效率还受到诸多因素影响，例如内存效率与通信效率，而不仅仅是 GEMM 性能。因此，优化低比特训练过程需要更全面的策略，可能需要新的训练算法来适配更低精度，并配套硬件创新以满足训练工作流的复杂需求。我们将这些挑战视为未来扩展 LUT TENSOR CORE 以支持训练的重要方向。

长上下文注意力与 KV 缓存量化。支持长上下文是提升 LLM 能力的重要方向之一 [13, 56]。在长上下文场景中，注意力机制往往成为计算瓶颈。当前研究和实践表明，在预填充（prefilling）阶段，将注意力计算量化为 FP8 对模型精度影响有限 [60]，但超低比特精度对模型精度的影响尚缺乏系统研究。在解码阶段，多项工作表明，将 KV 缓存量化到 4 比特甚至 2 比特对模型性能的影响可以忽略不计 [22, 41]。在此过程中，Q 矩阵仍保持高精度，计算形式本质上仍是 mpGEMM。因此，将 LUT TENSOR CORE 拓展到长上下文场景，是一个颇具潜力的未来研究方向。

更灵活的数据格式与非整数权重。我们认为，LUT 方案在本质上非常适合支持多种精度组合，因为其将主计算（点积）替换为查找表访问。目前，LUT TENSOR CORE 支持 $W_{\mathrm{INT}}A_{\mathrm{FP}}$ 和 $W_{\mathrm{INT}}A_{\mathrm{INT}}$ 的组合。若要扩展到 $W_{\mathrm{FP}}$，我们的初步策略是将尾数（mantissa）与符号位类似于 $W_{\mathrm{INT}}$ 处理，将其作为查找表索引，而将指数位作为移位器（shifter）的输入。此外，LUT 也天然适配非整数权重格式。例如对于三值（ternary）权重，LUT 方案可以将三个三值权重打包到 5 个比特中，而基于 ADD/MAC 的方法通常需要 6 个比特才能表示相同的信息。

mpGEMM 支持的前沿趋势。新一代 GPU（如 B100 [8]）已经在 Tensor Core 中原生支持混合精度 GEMM [9, 50]。Blackwell 进一步引入了 FP4、FP6、FP8 及其变体 NVFP4、MXFP4、MXFP6 和 MXFP8 等窄精度格式，可支持包括 $A_{FP4,FP6,FP8} \times W_{FP4,FP6,FP8}$ 和 $A_{MXF4,MXF6,MXF8} \times W_{MXF4,MXF6,MXF8}$ 在内的一系列混合精度 GEMM，并在吞吐上保持与 $W_{FP8}A_{FP8}$ Tensor Core 相同。LUT Tensor Core 可以通过位串行（bit-serial）的方式支持这些运算，并在不同格式间实现可扩展的性能。随着 NVIDIA 等主流厂商提供原生支持，mpGEMM 很可能成为一个关键且被广泛采用的计算模式。

![](./assets/f78a2eaa0fae0102b538cb0296f8146da4cd6364dfc835c14f46ee1dddabb596.jpg)  
Figure 19: Roofline analysis of conventional  $W_{FP16}A_{FP16}$  Tensor Core and  $W_{INT1}A_{FP16}$  from LUT Tensor Core.

图 19：传统 $W_{FP16}A_{FP16}$ Tensor Core 与 LUT Tensor Core 中 $W_{INT1}A_{FP16}$ 的 Roofline 分析。

LUT TENSOR CORE 的 Roofline 分析。图 19 展示了在 A100 内存系统下，传统 $W_{FP16}A_{FP16}$ Tensor Core 与基于 LUT 的 $W_{INT1}A_{FP16}$ Tensor Core 的 Roofline 图。横轴表示基于主存流量的运算强度（operational intensity）。LUT TENSOR CORE 中 $W_{INT1}A_{FP16}$ Tensor Core 所占面积仅为传统 $W_{FP16}A_{FP16}$ Tensor Core 的 $58.4\%$，但理论 FLOPs 却提升了 $4\times$。在原始设计中，$W_{FP16}A_{FP16}$ 处于计算受限（compute-bound）状态，而朴素的 LUT 实现则受制于内存带宽（memory-bound）。通过软硬件协同优化——包括通过权重重解释将表大小减半并减少激活内存流量，采用拉长的平铺（elongated tiling）提升数据复用，以及使用 thread block “swizzling” 提升 L2 命中率——LUT TENSOR CORE 显著提升了运算强度，使优化点逼近 Roofline 的“脊点”（ridge point）。

# 6 Related work

## 6 相关工作

低比特 DNN 加速器。随着 LLM 规模持续增大，低比特量化技术在减小模型规模和降低计算需求方面的重要性日益凸显。为支持量化模型推理中更低比特宽度的数据类型，业界已经提出了多种硬件加速器。NVIDIA GPU 架构也顺应这一趋势，逐步引入更低精度格式：从 Fermi 时代对 FP32 和 FP64 的支持，到 Pascal 中的 FP16，再到 Turing 中的 INT4 和 INT8，以及 Ampere 中的 BF16。在 LLM 时代，Hopper 引入了 FP8 [47]，Blackwell 更是进一步发展到 FP4 [57]。除 GPU 外，许多研究还提出了面向低比特量化 DNN 的定制加速器 [19, 35, 43, 58, 77, 78]。尽管这些工作取得了显著进展，但它们主要关注于两侧输入（权重与激活）具有相同数据类型和比特宽度的 GEMM。FIGNA [25] 通过定制 $W_{INT4}A_{FP16}$ 算术单元提升低比特 LLM 推理效率。而 LUT TENSOR CORE 则采用 LUT 计算范式提升 mpGEMM 的效率，并在无需复杂硬件重设计的前提下，提供对多种精度组合的灵活支持。

稀疏 DNN 加速器。与低比特量化类似，稀疏化也是降低模型规模、加速 DNN 推理的常用策略。稀疏化利用 DNN 权重矩阵或激活中天然存在的零元素，通过跳过这些零值的存储与计算来提升效率。随着 NVIDIA A100 GPU 的问世，Sparse Tensor Core 被引入以原生支持稀疏性，提供了 2:4 结构化稀疏 [6]。在商用 GPU 之外，定制稀疏 DNN 加速器也日益受到关注，这些设计通过剪枝（pruning）、跳零（zero-skipping）、稀疏矩阵格式等技术，在不同程度上挖掘稀疏性以优化存储与计算 [17, 23, 24, 61, 70, 74, 85]。稀疏性在低比特 LLM 中同样广泛存在。当稀疏化与量化结合时，有潜力带来更可观的效率收益。但如何在保证模型精度的前提下，同时兼顾量化与稀疏性，并设计高效的微架构，仍面临不小挑战。在 LUT TENSOR CORE 中引入稀疏性支持是一个颇具前景的研究方向，我们将其留作未来工作。

# 7 Conclusion

## 7 总结

本文提出了 LUT TENSOR CORE，这是一种基于 LUT 计算范式的软件–硬件协同设计，用于实现高效的混合精度 GEMM，从而加速低比特 LLM 推理。LUT TENSOR CORE 在提升性能的同时，提供了对多种精度组合的广泛支持，并且能够与现有加速器架构及软件生态无缝集成。