# LUT Tensor Core：一种面向基于LUT的低比特LLM推理的软硬件协同设计

Zhiwen Mo*

伦敦帝国理工学院

英国，伦敦

微软研究院

中国，北京

zhiwen.mo25@imperial.ac.uk

Lei Wang*

北京大学

中国，北京

微软研究院

中国，北京

leiwang1999@outlook.com

Jianyu Wei*

中国科学技术大学

中国，合肥

微软研究院

中国，北京

noob@mail.ustc.edu.cn

Zhichen Zeng*

华盛顿大学

美国，西雅图

微软研究院

中国，北京

zczeng@uw.edu

Shijie Cao†

微软研究院

中国，北京

shijiecao@microsoft.com

Lingxiao Ma

微软研究院

中国，北京

lingxiao.ma@microsoft.com

Naifeng Jing

上海交通大学

中国，上海

sjtuj@sjtu.edu.cn

Ting Cao

微软研究院

中国，北京

Ting.Cao@microsoft.com

Jilong Xue

微软研究院

中国，北京

jxue@microsoft.com

Fan Yang

微软研究院

中国，北京

fanyang@microsoft.com

Mao Yang

微软研究院

中国，北京

maoyang@microsoft.com

# 摘要 (Abstract)

大语言模型（LLM）推理正变得资源密集，促使人们转向低比特模型权重以减少内存占用并提高效率。此类低比特LLM需要混合精度矩阵乘法（mpGEMM），这是一项重要但尚未被充分探索的操作，涉及低精度权重与高精度激活值的乘法。现成的硬件并不原生支持此操作，导致了基于反量化（dequantization）的间接实现，从而效率低下。

在本文中，我们研究了用于mpGEMM的查找表（LUT）方法，发现传统的LUT实现未能实现预期的收益。为了释放基于LUT的mpGEMM的全部潜力，我们提出了 **LUT TENSOR CORE**，这是一种面向低比特LLM推理的软硬件协同设计。LUT TENSOR CORE通过以下方面区别于传统的LUT设计：1）基于软件的优化，以最小化表格预计算开销，并通过权重重解释（weight reinterpretation）减少表格存储；2）一种基于LUT的Tensor Core硬件设计，采用细长的分块形状（elongated tiling shape）以最大化表格复用，并采用位串行设计以支持mpGEMM中多样的精度组合；3）面向基于LUT的mpGEMM的新指令集和编译优化。与现有的纯软件LUT实现相比，LUT Tensor Core显著优于前者，并且与之前最先进的基于LUT的加速器相比，计算密度和能效提高了 $1.44\times$。

# CCS 概念 (CCS Concepts)

- 计算机系统组织 $\rightarrow$ 神经网络；架构； - 硬件 $\rightarrow$ 算术和数据通路电路。

# 关键词 (Keywords)

低比特LLM，软硬件协同设计，LUT，加速器

This work is licensed under a Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International License.

ISCA '25, Tokyo, Japan

© 2025 Copyright held by the owner/author(s).

ACM ISBN 979-8-4007-1261-6/25/06

https://doi.org/10.1145/3695053.3731057

# ACM 参考文献格式 (ACM Reference Format):

Zhiwen Mo, Lei Wang, Jianyu Wei, Zhichen Zeng, Shijie Cao, Lingxiao Ma, Naifeng Jing, Ting Cao, Jilong Xue, Fan Yang, and Mao Yang. 2025. LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference. In Proceedings of the 52nd Annual International Symposium on Computer Architecture (ISCA '25), June 21-25, 2025, Tokyo, Japan. ACM, New York, NY, USA, 15 pages. https://doi.org/10.1145/3695053.3731057

# 1 引言 (Introduction)

大语言模型（LLM）的出现为各种AI应用带来了变革性的机遇 [1, 3, 28, 65]。然而，LLM的部署需要大量的硬件资源 [21, 54, 55]。为了降低推理成本，低比特LLM已成为一种有前景的方法 [10, 15, 31, 40]。在各种解决方案中，权重量化，即以低精度权重和高精度激活值量化LLM，因其在保持模型精度的同时降低了内存和计算成本而变得特别具有吸引力 [39, 75, 81]。虽然4比特权重量化已变得普及 [12, 32, 64]，但学术界和工业界都在积极探索向2比特甚至1比特发展的进步，以进一步提高效率 [4, 14, 29, 42, 44, 49, 68]。

权重量化将LLM推理的关键计算模式从传统的通用矩阵乘法（GEMM）转变为混合精度GEMM（mpGEMM），其中一个输入矩阵为低精度（例如，INT4/2/1 权重），而另一个保持高精度（例如，FP16/8，INT8 激活值）。目前，现成的硬件并不原生支持混合精度操作。因此，大多数低比特LLM推理系统不得不利用基于反量化的方法进行mpGEMM [16, 39, 51, 69]。反量化将低比特表示上采样以匹配硬件支持的GEMM。这种额外的操作在大批量场景中可能成为性能瓶颈。

查找表（LUT）是另一种流行的低比特计算方法，非常适合mpGEMM [26, 38, 45, 53, 71]。通过预计算低精度权重和高精度激活值之间的乘法结果，基于LUT的方法消除了对反量化的需求，并用简单的查表操作替换了复杂的运算。在实践中，LUT是基于分块（tile）实现的。对于mpGEMM的每个小块，专门为块内的激活值预计算一个查找表，并在权重矩阵列之间复用，从而在保持效率的同时显著减少存储开销。

尽管潜力巨大，但基于LUT的mpGEMM在软件和硬件实现中仍面临显著的性能差距和挑战。在软件方面，LUT内核面临指令支持有限和内存访问效率低下的问题，导致其性能低于GPU上基于反量化的内核，如图4所示。在硬件方面，传统的LUT设计缺乏针对mpGEMM的优化，往往达不到预期的性能提升。这是由于关键挑战的存在，如高昂的表格预计算和存储开销、对多样化位宽组合的支持有限、次优分块形状导致的低效，以及缺乏专用指令集和编译支持；详见第2.3节。

LUT TENSOR CORE通过整体的软硬件协同设计解决了这些挑战。通过采用基于软件的方法优化硬件不友好的任务（如表格预计算和存储管理），LUT TENSOR CORE减少了硬件的工作负载，简化了设计并提高了其紧凑性和效率。具体而言：

**软件优化 (§ 3.1)。** 为了分摊预计算查找表的开销，我们观察到传统设计在多个单元间进行了冗余的预计算。LUT TENSOR CORE 将预计算拆分为一个独立的算子，从而避免了冗余，并将其与前一个算子融合以进一步减少内存访问。为了减少存储开销，LUT TENSOR Core 通过将 $\{0,1\}$ 重解释为 $\{-1,1\}$ 来揭示并利用mpGEMM查找表固有的对称性，将表格大小减少一半。LUT TENSOR Core 还通过应用表格量化技术减小了表格宽度并支持各种激活位宽。

**硬件定制 (§ 3.2)。** LUT TENSOR CORE 定制了基于LUT的Tensor Core设计。上述软件优化通过将电路任务卸载到软件，简化了硬件设计，将广播和多路复用器的需求减少了一半。同时，LUT TENSOR CORE 结合了灵活的类位串行（bit-serial-like）电路，以适应混合精度操作的各种组合。此外，LUT TENSOR CORE 对基于LUT的Tensor Core的形状进行了设计空间探索（DSE），并确定了一种细长的分块形状，该形状能够实现更高效的表格复用。

**新指令和编译支持 (§ 3.3)。** LUT TENSOR Core 将传统的矩阵乘累加（MMA）指令集扩展为基于LUT的矩阵乘累加（LMMA）指令集，其中包含指定操作数类型和形状的基本元数据。通过扩展，LUT TENSOR Core 利用LMMA中提供的形状信息，使用基于分块的深度学习编译器 [5, 62, 84] 重新编译LLM工作负载，为新硬件生成高效的内核。

我们的基于LUT的Tensor Core与传统Tensor Core相比，功耗和面积减少了 $4 \times$ 到 $6 \times$。为了验证mpGEMM的性能提升，我们将基于LUT的Tensor Core设计和指令集成到了Accelsim [30]（一个GPU硬件模拟器）中。结果表明，我们的基于LUT的Tensor Core仅占传统Tensor Core面积的 $16\%$，却实现了更高的mpGEMM性能。与最先进的（SOTA）LUT软件实现 [53] 相比，LUT TENSOR CORE 在通用矩阵向量乘法（GEMV）中实现了高达 $1.42 \times$ 的加速，在GEMM中实现了 $72.2 \times$ 的加速。与SOTA LUT加速器 [38] 相比，LUT TENSOR CORE 实现了 $1.44 \times$ 更高的计算密度和能效，这得益于软硬件协同设计。我们的代码可在 https://github.com/microsoft/T-MAC/tree/LUTTensorCore_ISCA25 获取。

我们的贡献总结如下：

- 我们提出了 LUT TENSOR CORE，一种面向基于LUT的mpGEMM的软硬件协同设计，以提升低比特LLM的推理效率。
- 实验表明，所提出的基于LUT的Tensor Core实现了 $4 \times$ 到 $6 \times$ 的功耗、性能和面积（PPA）收益。对于像BitNet和量化LLAMA模型这样的低比特LLM，LUT TENSOR CORE 展示了 $2.06 \times$ 到 $5.51 \times$ 的推理加速，且具有相当的面积和精度。
- 除了效率之外，我们的设计还能适应广泛的权重（例如，INT4/2/1）和激活精度（例如，FP16/8，INT8）。此外，LUT TENSOR Core 可以通过扩展的LMMA指令和编译优化集成到现有的推理硬件和软件栈中。

图 1：LLM中的仅解码器Transformer块。主要计算是GEMM操作（或带有权重量化的mpGEMM操作）。

# 2 背景与动机 (Background and Motivation)

# 2.1 LLM推理与低比特量化

近年来，LLM主要依赖于仅解码器（decoder-only）的Transformer架构 [66]，如图1所示。具体而言，LLM由连续的Transformer层构建，其中每个Transformer层包含一个多头注意力块，后跟一个前馈块。在这两个块中，主要计算是GEMM，或带有权重量化的mpGEMM操作。LLM的扩展需要大量的硬件资源 [21, 28]。例如，LLAMA-2-70B [65] 仅模型权重（FP16格式）就消耗140GB内存，远超现代GPU（如NVIDIA A100或H100）的容量。这对LLM部署构成了巨大挑战。

为了降低LLM部署中的推理成本，低比特量化已成为一种流行的方法 [10, 12, 64, 76]。它降低了模型数值表示的精度，从而减少了内存占用和计算时间。在LLM量化中，权重量化优于激活量化 [37, 39]。这是因为模型权重的值是静态的，因此可以离线量化。权重可以被量化为4比特、2比特甚至1比特。对于4比特权重，训练后量化（PTQ）造成的精度损失极小 [12, 64, 76]。最近的研究和实践表明，使用量化感知训练（QAT），2比特权重量化在相同的内存预算下在模型精度上优于4比特 [14, 42, 49]。BitNet进一步表明，从头开始训练具有1.58比特（三值）甚至1比特（二值）权重的模型可以达到与16比特模型相当的精度 [44, 68]。ParetoQ [42] 同时也报告称，考虑到硬件限制，2比特量化在内存减少和加速方面提供了巨大的潜力。

相反，激活值是在运行过程中生成的，具有高方差，表现为动态离群值（outliers）[10, 18, 73]。由于离群值的存在，将激活值量化到8比特以下具有挑战性。针对各种模型和场景，研究者们已经探索了不同的权重和激活位宽组合 [10, 14, 15, 19, 68]，这表明没有通用的解决方案适合所有场景。

图 2：(a) GEMM，(b) 带反量化的间接mpGEMM，(c) 用于低比特LLM推理的直接mpGEMM。

图 3：FP16激活值和INT1权重的朴素LUT-based mpGEMM分块示例。利用预计算的表格，一次查表可以替代4元素向量的点积。

# 2.2 面向低比特LLM的基于LUT的mpGEMM

权重和激活值的不同位宽导致了对混合精度GEMM（mpGEMM）的独特需求，例如INT4/2/1乘以FP16，如图2所示。目前的商用LLM推理硬件，如GPU和TPU，缺乏对mpGEMM的原生支持，而是专注于具有统一输入格式的传统GEMM。基于反量化的mpGEMM通过上采样低精度权重以匹配高精度激活值来弥补这一差距 [50, 69]。然而，这种方法引入了额外的反量化操作，并放弃了低精度计算的效率增益。

基于LUT的mpGEMM是一种日益受到关注的低比特LLM推理方法 [26, 38, 45, 53, 71]。它预先计算高精度激活值和低精度权重之间的点积，然后将其存储在查找表（LUT）中以便快速检索。与其预计算所有可能的高精度和低精度值组合（例如，FP16 $\times$ INT4，这将需要大小为 $(2^{16} \times 2^{4})$ 的表格），基于LUT的mpGEMM以分块的方式组织计算。对于mpGEMM操作的每个小块，即每一小组激活值，专门为这些激活值预计算一个LUT，并在权重列之间复用。这种方法最小化了表格大小，并通过在计算期间动态构建每个分块的LUT来保持效率。图3展示了一个基本示例，其中一个小分块由 $1 \times 4$ 的FP16激活值和 $4 \times N$ 的INT1权重组成。当激活向量长度为4时，查找表大小为16。在这种情况下，长度为4的点积的每个结果都可以通过简单的查表获得。该表可以复用N次，考虑到权重矩阵的大小，这是非常可观的。更大的激活向量或更高比特的权重需要按比例更大的查找表。

图 4：从LLAMA2-70B提取的形状为M0-M3的mpGEMM内核性能。$W_{INT4}A_{FP16}$ 表示INT4权重和FP16激活值。在A100 GPU上，基于LUT的软件内核（LUT-GEMM）性能不如基于反量化的内核（CUTLASS）。

# 2.3 当前基于LUT解决方案的差距

基于LUT的mpGEMM因其在消除反量化和乘法、并通过简单的查表减少加法方面的优势而前景广阔。然而，现有的软件和硬件实现面临着挑战和差距：

**软件LUT内核。** 基于LUT的mpGEMM软件内核经常面临与有限指令支持和低效内存访问相关的挑战。这种限制是双重的：首先，GPU对查表的指令支持有限。最有效的可用指令 `prmt`（置换）具有有限的宽度，无法在单条指令中完成整个查表，从而降低了吞吐量。其次，表格位置显著影响性能。将查找表存储在寄存器文件中会导致数据在线程间的大量复制，这是由于LUT方法的广播特性，在处理大表时会导致寄存器溢出。相反，将表放置在共享内存中可能会由于warp内线程的随机访问而导致bank冲突，严重影响内存带宽。这些问题导致它们在现有的LLM推理硬件（如GPU）上，效果不如基于反量化的内核。图4比较了 [53] 中的基于LUT的mpGEMM内核与CUTLASS [50] 中基于反量化的mpGEMM内核在A100 GPU上的性能。结果表明，基于反量化的内核始终优于基于LUT的内核。值得注意的是，当批量较大时，基于LUT的内核由于表格访问开销而遭受显著的性能下降，表现差几个数量级。“Seg. Error”注释表示在LUT-GEMM [53] 中观察到的分段错误。

**硬件LUT加速器。** 乍一看，定制的LUT硬件因其简单性（仅需寄存器存储表格和多路复用器进行查找）而有望获得效率增益。然而，我们的研究表明，传统的LUT硬件设计未能提供这些增益。图5描绘了一种用于mpGEMM的传统三步LUT硬件解决方案：表格预计算、查表和部分和累加。众多挑战和未探索的设计方面显著影响了整体性能：（1）**表格预计算和存储。** 预计算的表格可能占用过多的存储空间，产生面积和延迟开销，从而削弱效率增益。（2）**位宽灵活性。** 支持各种位宽组合（例如，INT4/2/1 × FP16/FP8/INT8）可能会消耗过多的芯片面积。（3）**LUT分块形状。** 次优的分块增加了存储成本并限制了表格复用机会，从而影响性能。（4）**指令和编译。** 基于LUT的mpGEMM需要新的指令集。然而，针对标准GEMM硬件优化的传统编译栈可能无法有效地映射和调度新指令集，使得与现有软件栈的集成变得复杂。

图 5：三步走的传统LUT硬件。表格预计算和存储引入了沉重的开销。

图 6：LUT TENSOR CORE 工作流程。

# 3 LUT TENSOR CORE 设计

我们介绍了 LUT TENSOR CORE，一种旨在解决上述效率、灵活性和集成挑战（§2.3）的软硬件协同设计。图6提供了 LUT TENSOR CORE 的概览。与传统的基于硬件的LUT解决方案不同（其表格预计算和存储引入了显著的硬件开销），LUT TENSOR CORE 引入了基于软件的优化（§3.1）来优化表格预计算和存储：通过算子融合执行输入激活张量的LUT表格预计算，同时对输入权重张量进行重解释以实现表格存储优化。在硬件方面，基于LUT的Tensor Core微架构（§3.2）为mpGEMM处理提供了高效率，并为不同位宽的数据类型提供了灵活性。为了将 LUT TENSOR CORE 集成到现有的深度学习生态系统中，LUT TENSOR CORE 设计了 LMMA 指令集以暴露基于LUT的Tensor Core用于mpGEMM编程，并实现了一个编译栈来调度端到端的LLM执行（§3.3）。

# 3.1 基于软件的表格优化

如第2节所述，基于LUT的mpGEMM需要额外的表格预计算过程和存储空间来存储预计算结果。朴素地，在 $W\_BIT$ 权重上预计算长度为 $K$ 的激活向量的点积需要 $(2^{W\_BIT})^K$ 个表项。对于每个激活元素，将其与 $W\_BIT$ 权重相乘有 $2^{W\_BIT}$ 种可能的结果，从而为该激活元素构建预计算表。因此，对于长度为 $K$ 的激活向量，预计算表有 $(2^{W\_BIT})^K$ 个表项。图3展示了 $K = 4$，$W\_BIT = 1$ 时具有 $2^4$ 个表项的查找表。

一种常用的优化是位串行（bit-serial）[27]，它将一个 $W$ 位的整数表示为 $W$ 个1位整数，并通过位移在1位整数上执行乘法。这种范式可以在1位上复用预计算表，从而将表格大小减少到 $2^{K}$。尽管如此，即使这种减小的表格大小也带来了显著的硬件开销。LUT TENSOR CORE 提出了数据流图（DFG）变换和算子融合来消除表格预计算开销，以及权重重解释和表格量化来减小表格大小。

**3.1.1 利用DFG变换和算子融合预计算查找表。** 基于LUT的mpGEMM需要预计算高精度激活值和一组低精度权重之间的点积作为表格，以供后续的查表操作使用。传统实现将预计算单元放置在LUT单元附近，为每个LUT单元即时执行表格预计算。这种方法由于冗余引入了显著的硬件成本，因为多个预计算单元通常执行相同的操作。以OPT-175B中的 [4096,12288] $\times$ [12288,12288] GEMM为例，一个朴素的直接预计算单元在数组大小为 $N = 4$ 的基于LUT的Tensor Core之间共享一个表格。在这种设置下，每个表格在整个过程中被不同的LUT单元重复计算（12288/4 = 3072次），施加了巨大的计算负担。

为了解决这种低效问题，我们首先变换DFG，将预计算拆分为一个独立的内核，实现一次性预计算，结果可广播到所有LUT单元。这一修改将预计算开销减少了数百倍，使其可由现有的向量单元（如CUDA核心）管理。为了分摊广播引入的额外内存流量，LUT TENSOR CORE 利用其逐元素（element-wise）计算模式，将预计算算子与前一个算子融合，如图6所示，详见§3.3.2。这种融合减少了内存访问，并将预计算开销降低到几乎为零，如§4.6.1中的评估所示。

**3.1.2 通过权重重解释实现表格对称化。** 预计算长度为 $K$ 的激活向量的 $2^{K}$ 表格大小在表格存储和表格访问方面都引入了显著的成本。为了解决这个问题，我们观察并利用了整数表示的对称性。

假设原始量化的权重表示为：

$$r _ {w} = s _ {w} \left(q _ {w} - z _ {w}\right) \tag {1}$$

其中 $r_w$ 是实数值权重，$s_w$ 是缩放因子，$z_w$ 是偏置，$q_w$ 是 $K$ 位整数表示。

我们的目标是映射 $q_{w}$ 使其围绕零对称，同时保持数学等价性。为了实现这一点，必须调整 $s_{w}$ 和 $z_{w}$。当映射无符号 $q_{w}$ 使其关于零对称时，需要进行以下调整：

$$q _ {w} ^ {\prime} = 2 q _ {w} - \left(2 ^ {K} - 1\right), \quad s _ {w} ^ {\prime} = s _ {w} / 2, \quad z _ {w} ^ {\prime} = 2 z _ {w} + 1 - 2 ^ {K} \tag {2}$$

图 7：将 0,1 重解释为 -1,1 以实现对称性，从而将表格大小减半。

此过程如图7所示，展示了变换4比特无符号整数的示例。通过计算 $s_w'$ 和 $z_w'$，$q_w'$ 从 $\{0, 1, \dots, 14, 15\}$ 映射到 $\{-15, -13, \dots, 13, 15\}$，实现了围绕零的对称。

接下来，点积可以表示为：

$$D P = \Sigma A c t _ {i} s _ {w} \left(q _ {w i} - z _ {w}\right) = \Sigma A c t _ {i} s _ {w} ^ {\prime} \left(q _ {w i} ^ {\prime} - z _ {w} ^ {\prime}\right) \tag {3}$$

其中 $DP$ 是点积，$Act_i$ 是激活值。因此，量化过程保持不变，额外步骤是离线将权重的 $s_w(q_{wi} - z_w)$ 映射到 $s_w'(q_{wi}' - z_w')$。让我们考虑二进制表示 $W_3W_2W_1W_0 = 0.010$ 与变量 $A, B, C, D$ 之间的点积。最初，二进制值 $\{0',1\}$ 被解释为 $\{0,1\}$。假设 $s_w = 2$ 且 $z = 1/2$。计算如下：

$$\begin{array}{l} D P = \sum A c t _ {i} s _ {w} \left(q _ {w i} - z _ {w}\right) \\ = A \cdot 2 \cdot (0 - 0. 5) + B \cdot 2 \cdot (1 - 0. 5) \\ + C \cdot 2 \cdot (1 - 0. 5) + D \cdot 2 \cdot (1 - 0. 5) \\ = - A + B - C - D \\ \end{array}$$

重解释后，二进制值 $\{^{\prime}0^{\prime},1^{\prime}\}$ 被重映射为表示 $\{-1,1\}$，调整后的缩放因子 $s_w' = 1$ 和偏置 $z_w' = 0$。更新后的计算为：

$$\begin{array}{l} D P = \sum A c t _ {i} s _ {w} ^ {\prime} \left(q _ {w i} ^ {\prime} - z _ {w} ^ {\prime}\right) \\ = A \cdot 1 \cdot (- 1 - 0) + B \cdot 1 \cdot (1 - 0) \\ + C \cdot 1 \cdot (- 1 - 0) + D \cdot 1 \cdot (- 1 - 0) \\ = - A + B - C - D \\ \end{array}$$

显然，这两个表达式在数学上保持等价。由于表格条目关于零对称，查找表表现出类似于奇函数的性质。假设索引是一个4位值 $W_{3}W_{2}W_{1}W_{0}$，查找表（LUT）的朴素实现需要 $2^{4} = 16$ 个条目。然而，可以观察到以下类似于奇函数的性质成立：

$$\mathrm {L U T} \left[ W _ {3} W _ {2} W _ {1} W _ {0} \right] = - \mathrm {L U T} [ \sim \left(W _ {3} W _ {2} W _ {1} W _ {0}\right) ] \tag {4}$$

因此，LUT中的条目数可以减少到原始的一半，即 $2^{4 - 1} = 8$，方程变为：

$$\operatorname {L U T} \left[ W _ {3} W _ {2} W _ {1} W _ {0} \right] = \left\{ \begin{array}{l l} - \operatorname {L U T} \left[ \sim \left(W _ {2} W _ {1} W _ {0}\right) \right], & \text {i f} W _ {3} = 1 \\ \operatorname {L U T} \left[ W _ {2} W _ {1} W _ {0} \right], & \text {i f} W _ {3} = 0 \end{array} \right. \tag {5}$$

这里，$\sim$ 表示按位取反操作。因此，给定长度为 $K$ 的激活向量，表格对称化可以将表格长度减少到 $2^{K - 1}$。表格大小不仅影响预计算阶段所需的计算操作，还影响多路复用器的大小。此外，表格中的每个条目还需要广播到 $N$ 个PE（通常为64或128）以进行点积计算。这样的优化显著减少了

图 8：采用位串行的优化LUT单元。

广播开销和MUX选择开销。注意，方程5中的 $W_{3}W_{2}W_{1}W_{0}$ 是静态权重。位级取反可以离线完成以简化设计，如下所示：

$$\mathrm {L U T} \left[ W _ {3} ^ {\prime} W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right] = \left\{ \begin{array}{l l} - \mathrm {L U T} \left[ W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right], & \text {i f} W _ {3} ^ {\prime} = 1 \\ \mathrm {L U T} \left[ W _ {2} ^ {\prime} W _ {1} ^ {\prime} W _ {0} ^ {\prime} \right], & \text {i f} W _ {3} ^ {\prime} = 0 \end{array} \right. \tag {6}$$

这种简化可以消除电路设计中的取反操作，这将在§3.2中介绍。

**3.1.3 表格量化。** 对于FP32或FP16等高精度激活值，我们采用表格量化技术将预计算的表格元素转换为较低的统一精度，如INT8。这种方法通过支持多种激活精度提供了灵活性，并通过减小表格大小提高了效率。

与传统的激活量化相比，表格量化允许在表格预计算阶段进行更动态、更细粒度的量化。例如，对于大小为4的激活元素组，我们对包含8个预计算点积的每个表格进行量化。我们在§ 4.6.2中讨论的实证实验表明，INT8表格量化在简化硬件设计的同时保持了高精度，从而验证了我们方法的有效性。

# 3.2 基于LUT的Tensor Core微架构

**3.2.1 采用位串行的简化LUT单元设计。** 通过利用基于软件的预计算融合和权重重解释，减少了定制每个单独LUT单元的硬件成本。图8展示了我们的LUT单元设计。与直接的设计相比，存储LUT所需的寄存器以及与表格广播和多路复用器相关的成本减少了一半。如方程6所示，位级取反电路可以从每个LUT单元中消除，以进一步提高效率。为了支持灵活的权重位宽，我们采用了位串行电路架构 [27, 74]。该设计将权重位宽映射到 W_BIT 个周期，从而能够以串行方式处理各种位宽。

**3.2.2 细长的LUT分块。** $M$、$N$ 和 $K$ 维度的选择对基于LUT的Tensor Core的性能至关重要，因为针对基于MAC的Tensor Core的传统选择可能会导致次优性能。如图9所示，一个MNK分块的LUT阵列包含 $M$ 个表、$N$ 组权重和 $M*N$ 个基于MUX的单元。每个表包含 $M \times 2^{K-1}$ 个条目，每个条目广播到 $N$ 个MUX单元。每组分组二进制权重包含 $K$ 位，必须广播到 $M$ 个MUX单元作为MUX的选择信号。总表格大小由以下公式给出：

$$\text {T o t a l} = M \times 2 ^ {K - 1} \times \text {L U T} _ {\text {B I T}} \tag {7}$$

图 9：基于LUT的Tensor Core的细长 $MNK$ 分块。基于LUT的Tensor Core需要较大的 $N$（例如，64/128）以最大化表格复用，以及适当大小的 $K$（例如，4）以实现具有成本效益的表格大小。

分组二进制权重的大小由以下公式给出：

$$\text {G r o u p e d B i n a r y W e i g h t s S i z e} = K \times N \times \mathrm {W} _ {\text {B I T}} \tag {8}$$

其中 LUT_BIT 是LUT条目的位宽，W_BIT 是权重的位宽。

基于LUT的Tensor Core受益于细长的分块形状。当 $K$ 很大时，表项数量呈指数增长，而 $N$ 决定了有多少MUX单元可以复用每个表项。最佳配置需要平衡的 $K$、较大的 $N$ 和较小的 $M$，这与传统GPU Tensor Core不同。此外，分块形状影响I/O流量，更像正方形的分块配置可减少数据移动开销。在§4.2.2中，我们探索了 $MNK$ 分块的设计空间，确认细长的分块形状能产生更高的效率。

# 3.3 指令与编译

为了将 LUT TENSOR CORE 集成到现有的GPU架构和生态系统中，我们引入了一套新的指令集，并开发了基于分块的DNN编译器 [5, 62, 84] 的编译栈。

3.3.1 基于LUT的MMA指令。 为了能够使用基于LUT的Tensor Core进行编程，我们定义了一组LMMA指令，作为GPU中MMA指令集的扩展。

Imma.{M}{N}{K}. $\{A_{\mathrm{dtype}}\} \{W_{\mathrm{dtype}}\} \{Accum_{\mathrm{dtype}}\} \{O_{\mathrm{dtype}}\}$ 上述公式显示了LMMA指令的格式，类似于MMA。具体而言，$M,N$ 和 $K$ 指示基于LUT的Tensor Core的形状。$A_{dtype},W_{dtype},Accum_{dtype}$ 和 $O_{dtype}$ 分别指示输入、累加和输出的数据类型。与MMA指令类似，每条LMMA指令被调度到一个线程束（warp）执行。每个warp计算公式 $O_{dtype}[M,N] = A_{dtype}[M,K]\times$ $W_{dtype}[N,K] + Accum_{dtype}[M,N]$

3.3.2 编译支持与优化。 我们在TVM [5]、

图 10：LUT-mpGEMM的编译。整体数据流类似于cutlass [50]。细长的分块用于更好的数据复用。

Roller [84] 和 Welder [62] 之上实现了LUT-mpGEMM内核生成和基于LUT Tensor Core的端到端LLM编译。具体来说，编译栈包含以下关键方面。图10显示了在LLAMA模型上的编译示例：

- **DFG变换。** 给定以DFG表示的模型，我们将混合精度GEMM算子变换为预计算算子和LUT-mpGEMM算子。此变换在Welder [62] 中作为图优化pass实现。
- **算子融合。** 算子融合是一种广泛使用的编译器技术，通过减少内存流量和运行时开销来优化端到端模型的执行。我们复用Welder进行算子融合，通过注册具有所需基于分块表示的预计算和LUT-mpGEMM算子。如图10所示，逐元素的预计算算子与前一个逐元素算子融合。
- **LUT-mpGEMM调度。** 调度LUT-mpGEMM算子需要仔细考虑内存层次结构中的分块以获得最佳性能。传统的GEMM分块策略 [5, 82, 84] 假设激活和权重具有相同的数据类型。然而，mpGEMM对激活和权重使用不同的数据类型，影响内存事务。为了解决这个问题，我们按内存大小而不是形状来表示分块，并在Roller的rTile [84] 接口中注册LMMA指令形状和分块计算，以调度最佳配置。
- **代码生成。** 有了最终的调度计划，使用TVM执行代码生成。具体而言，LMMA指令在TVM中注册为intrinsic（内建函数），TVM可以遵循调度生成带有LMMA指令的内核代码。

# 4 评估 (Evaluation)

在本节中，我们评估 LUT TENSOR CORE 以验证其在加速低比特LLM推理方面的效率。首先，我们通过详细的PPA基准测试（$\S 4.2$）评估我们设计的硬件效率收益。然后，进行内核级实验以说明mpGEMM的加速效果（$\S 4.3$）。接下来，我们在常用的LLM上执行端到端推理评估，以展示实际的性能改进（$\S 4.4$）。最后，我们将

图 11：基于LUT的点积单元沿K轴的设计空间探索。$\mathbf{K} = 4$ 通常是最佳的。

图 12：MAC/ADD/基于LUT的DP4实现的PPA比较。我们的基于LUT的DP4单元具有计算密度和功耗优势。

LUT TENSOR CORE 与以前的基于LUT的工作进行比较（§4.5），并评估我们软件优化的有效性，重点关注表格预计算融合和表格量化（§4.6）。

图 13：MAC、ADD和基于LUT的DP4单元在 $W_{\mathrm{INTX}} \times A_{\mathrm{FP16}}$ 中随权重位宽变化的面积比较。传统的LUT实现不具备面积优势。

# 4.1 实验设置和方法

**4.1.1 硬件PPA基准测试。** 我们将基于LUT的Tensor Core与两个基线进行比较：基于乘累加（MAC）的Tensor Core和基于加法（ADD）的Tensor Core。MAC代表了当前GPU中的典型设计，需要反量化来支持mpGEMM。ADD采用了 [27] 中提出的位串行计算来支持mpGEMM，其中每一位权重需要一次加法。我们使用Verilog实现了基于LUT的Tensor Core和基线，并使用Synopsys的Design Compiler [63] 和TSMC 28nm工艺库综合电路并生成PPA数据。我们应用DC的针对1GHz的中等努力级别，以确保所有设计的公平比较。

**4.1.2 内核级评估。** 对于mpGEMM内核级评估，我们使用NVIDIA A100 GPU作为基线，并采用Accel-Sim [30]，一个开源的最先进的模拟器。修改Accel-Sim中的配置和跟踪文件使我们能够模拟原始A100和配备LUT TENSOR Core的A100。

**4.1.3 模型端到端评估和分析。** 为了将我们的评估扩展到真实的LLM，我们利用了四个广泛使用的开源LLM：LLAMA-2 [65]、OPT [80]、BLOOM [36] 和 BitNet [68]。由于Accel-Sim因为大跟踪文件的模拟速度慢而无法进行端到端LLM实验，我们开发了一个基于分块的模拟器来支持端到端推理评估，详见§4.4。

# 4.2 硬件PPA基准测试

**4.2.1 点积单元微基准测试。** 在本实验中，我们将 $M$ 和 $N$ 固定为1，并改变 $K$（即 $K$ 元素向量的点积单元）以探索其对计算密度的影响。较大的 $K$ 可能导致查找表条目的指数增长，而较小的 $K$ 会导致 $1 / K$ 的计算仍由加法器执行。如图11所示，我们发现INT操作在 $K = 4$ 时密度达到峰值，而浮点操作在 $K = 5$ 时表现最佳，但在 $K = 4$ 时也表现良好。因此，我们在所有后续基于LUT的设计中采用 $K = 4$。

我们对使用MAC、ADD和基于LUT方法的点积实现进行了基准测试，涵盖各种数据格式。这包括使用MAC的统一精度，如 $W_{\mathrm{FP16}}A_{\mathrm{FP16}}$，以及使用ADD和LUT方法的混合精度，如 $W_{\mathrm{INT1}}A_{\mathrm{FP8}}$。如图12所示，基于LUT的方法在 $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 下达到61.55 TFLOPs/mm²，超过了仅达到3.39 TFLOPs/mm²（$W_{\mathrm{FP16}}A_{\mathrm{FP16}}$）的传统MAC实现。能效显示出类似的趋势，基于LUT的方法比其他方法实现了更高的效率。

此外，我们在MAC/ADD/基于LUT的实现中对 $W_{\mathrm{INTX}} \times A_{\mathrm{FP16}}$ DP4单元进行了权重位扩展实验。实验配置为Tensor Core的N维度设置为4，以匹配A100的配置。如图13所示，当权重大于2比特时，传统的基于LUT的实现与MAC基线相比不具有面积优势。主要的面积效率瓶颈是表格预计算和存储开销。基于ADD的实现也仅在1比特和2比特情况下超过MAC基线。通过软硬件协同设计，LUT TENSOR Core在高达6位的权重位宽下优于所有基线，并且与传统LUT实现相比提供了更好的面积效率。

**4.2.2 Tensor Core基准测试。** 我们将评估扩展到Tensor Core级别，结合设计空间探索以确定最佳的 $MNK$ 配置。为了匹配具有 $M, N, K = 8, 4, 16$ 的A100 INT8 Tensor Core，我们将阵列大小设置为 $M \times N \times K = 512$。我们的实验涉及各种激活数据类型，包括 $A_{\mathrm{FP16}}$、$A_{\mathrm{INT16}}$、$A_{\mathrm{FP8}}$ 和 $A_{\mathrm{INT8}}$，以及多种权重位宽，如 $W_{\mathrm{INT1}}$、$W_{\mathrm{INT2}}$ 和 $W_{\mathrm{INT4}}$。我们将基于LUT的方法的性能与基于MAC和ADD的方法进行了比较。

如图14所示，我们扫描了不同的 $M, N, K$ 配置以探索设计空间并确保所有方法之间的公平比较。Y轴标记为“面积”，X轴标记为“功耗”。虚线表示每个设计方法在所有数据点中最小 面积×功耗 点所在的轮廓。我们的结果表明，在具有不同激活数据格式和权重位宽的12组实验中，基于LUT的方法实现了最小的面积和最低的功耗，除了 $W_{\mathrm{INT8}}A_{\mathrm{INT4}}$ 的情况。值得注意的是，在1比特权重下，基于LUT的方法与基于MAC的Tensor Core设计相比，功耗和面积减少了 $4\times -6\times$。我们确定基于LUT的Tensor Core的最佳MNK配置为 $M2N64K4$。这是因为激活是高位的，而权重是低位的。考虑到整体位宽，M维度计算为 $2\times 16 = 32$ 位，而N维度计算为 $64\times 1 = 64$ 位。整体位配置仍然近似于正方形阵列。

# 4.3 mpGEMM内核级评估

我们采用SOTA GPU模拟器Accel-Sim来验证LUT TENSOR CORE在mpGEMM操作上的效率及其与现有GPU架构的兼容性。mpGEMM形状提取自LLAMA2-13B，其中 $M = 2048$，$N = 27648$，$K = 5120$。mpGEMM的数据流设计为类cutlass且输出驻留（output-stationary），分块形状针对有效的数据复用进行了优化。例如，$W_{\mathrm{INT1}}A_{\mathrm{INT8}}$ 分块的一个很好的候选配置是将线程块（Thread Block）分块设置为 [128, 512, 32]，Warp分块设置为 [64, 256, 32]。

如图15所示，基于LUT的Tensor Core在mpGEMM操作中优于传统的基于MAC的Tensor Core。每个子图中最左边的两个条形分别代表A100的理想峰值性能和使用cuBLAS测量的性能。其余条形代表基于LUT的结果：理想峰值性能、模拟性能以及增加寄存器容量后的模拟性能。寄存器容量调整解决了由寄存器不足引起的瓶颈，这限制了大的分块并将性能与内存限制挂钩。例如，在 $W_{\mathrm{INT1}}A_{\mathrm{FP16}}$ 下，基于LUT的方法提供了略高的mpGEMM性能，同时仅使用了基于MAC的Tensor Core $14.3\%$ 的面积。

# 4.4 模型端到端评估

虽然Accel-Sim提供了详细的架构仿真，但其模拟速度大约慢了五百万倍，将A100 GPU上的十秒任务变成了长达579天的模拟周期，并生成超过79TB的跟踪文件。

为了克服这些障碍，我们开发了一个端到端模拟器，旨在以分块粒度进行快速准确的仿真。我们的关键见解是，高度优化、几乎没有停顿的大型GPU内核的行为可以被视为加速器，特别是在LLM场景中。这一观点得到了NVIDIA在NVAS [67] 中的发现的支持，该发现建议在哲学上将GPU模拟视为“动态交互的roofline组件”，而不是“逐周期的推进”。因此，我们采用已建立的加速器建模框架（如Timeloop [52]、Maestro [34] 和 Tileflow [83]）中的分析方法，开发了一个基于分块的GPU模拟器。该工具

图 14：不同 LUT-/ADD-/MAC-based Tensor Core 实现的mpGEMM PPA。

图 15：$A_{\mathrm{FP16}}$ 和 $A_{\mathrm{INT8}}$ Tensor Core设计的Accel-Sim运行时间和面积。符号 $\times$ 表示相对于 $1 \times$ 基线的Tensor Core阵列大小，其中 $1 \times$ 对应于NVIDIA A100中的 $M \times N \times K = 512$ 阵列大小。

有助于对数据流、内存带宽、计算资源和算子融合进行详细准确的评估。我们计划在未来的工作中开源此模拟器。

**4.4.1 模拟器精度评估。** 在图16中，我们在A100和RTX 3090 GPU上使用OPT-175B、BLOOM-176B和LLAMA2-70B的各种配置对单层进行了验证。我们的模拟器相对于真实GPU性能的平均绝对百分比误差仅为 $5.21\%$，而在模拟速度上显著快于Accel-Sim。

图 16：端到端模拟器精度评估。

图 17：LLM上的端到端模拟结果（A100和3090）。R：真实GPU，M：建模，DRM：双倍寄存器建模。

图 18：LUT TENSOR CORE 与基于LUT的软件工作 LUT-GEMM [53] 在GEMM和GEMV上的比较。

**4.4.2 端到端推理模拟结果。** 图17展示了OPT、BLOOM和LLAMA模型的基准测试结果。我们的实验表明，LUT TENSOR CORE 实现了高达 $8.2 \times$ 的端到端加速，同时占用的面积小于传统的 $W_{\mathrm{FP16}} A_{\mathrm{FP16}}$ Tensor Core。值得注意的是，即使在 $8 \times$ 设置下，LUT TENSOR CORE 的面积也仅为传统 $W_{\mathrm{FP16}} A_{\mathrm{FP16}}$ MAC-based Tensor Core 的 $38.3\%$。

**4.4.3 总体比较。** 如表1所示，配备 LUT + BitNet 的A100在推理速度上提供了高达 $5.51 \times$ 的加速，同时仅利用了原始Tensor Core面积的 $38.3\%$。这导致计算密度增加了高达 $20.9 \times$，能效提高了 $11.2 \times$，这是由量化的LUT表格和通过软硬件协同设计高度优化的LUT电路实现的。与H100的原始 $W_{FP8}A_{FP8}$ Tensor Core相比，LUT Tensor Core可以实现高达 $2.02 \times$ 的面积效率提升。

# 4.5 与先前工作的比较

**4.5.1 基于LUT的软件。** LUT-GEMM [53] 和 T-MAC [71] 分别是以前针对GPU和CPU的SOTA基于LUT的软件解决方案。由于T-MAC是为CPU设计的，我们使用LUTGEMM在GPU上进行更相关的比较。LUT TENSOR CORE 的配置仅使用了传统FP16 Tensor Core $57.2\%$ 的面积。图18展示了LUT TENSOR CORE和LUT-GEMM相对于A100上 $W_{FP16}A_{FP16}$ cuBLAS的比较加速比。LUT-GEMM仅在GEMV情况下提高了性能，但在GEMM中比cuBLAS慢几十倍。与基于软件的LUT-GEMM相比，LUT TENSOR CORE 实现了高达 $1.42\times$ 更快的GEMV和 $72.2\times$ 更快的GEMM。

**4.5.2 基于LUT的硬件。** UNPU [38] 是用于DNN工作负载的SOTA基于LUT的硬件加速器。由于没有公开代码，我们根据其论文重新实现了UNPU设计，并应用优化以确保公平比较。我们在Tensor Core级别对UNPU和LUT TENSOR CORE进行了DSE。以Tensor Core配置为 $M\times N\times K = 512$ 下的 $W_{\mathrm{INT8}}A_{\mathrm{INT2}}$ 为例，消融研究评估了每项优化的影响。表2显示，针对多位权重的权重重解释和对称化将计算强度和功率效率提高了 $30\%$。额外的优化，包括离线权重重解释、取反电路消除、DFG变换和内核融合，使LUT TENSOR CORE 在这些指标上比UNPU提高了 $1.44\times$。

**4.5.3 量化DNN加速器。** 以前的工作，如Ant [19]、FIGNA [25] 和 Mokey [78]，主要设计带有针对专用量化精度（例如，int8×int8或int4×fp16）的MAC的PE。虽然对特定数据类型有效，但这些设计缺乏适应不同精度要求的灵活性。它们要么在转换为较低精度格式时牺牲模型精度，要么在转换为较高精度格式时错失效率机会。相比之下，我们采用了一种基于LUT的方法，通过不同的LMMA指令支持1-4位INT权重和FP/INT 16/8激活，涵盖了大多数低比特LLM用例。表3将LUT TENSOR CORE与其他加速器进行了比较。

# 4.6 软件优化分析

**4.6.1 表格预计算融合分析。** 表4展示了将预计算与DNN编译器Welder [62] 结合的影响，该编译器通过优化算子融合来增强推理性能。此评估是在批量预取和解码配置下的OPT-175B、BLOOM-176B和LLAMA2-70B模型的单层上进行的。最初，CUDA核心上的预计算导致了 $16.47\%$ 和 $24.41\%$ 的平均开销。然而，通过将预计算视为Welder搜索空间内的独立算子，开销减少到 $2.62\%$ 和 $2.52\%$，使其在整体执行时间中可以忽略不计。

**4.6.2 表格量化分析。** 为了评估表格量化的影响，我们在具有2比特量化权重的LLAMA2-7B模型 [65] 上进行了比较实验。第一行数据代表原始的 $W_{FP16}A_{FP16}$ LLAMA2-7B模型，第二项对应于BitNet-b1.58论文 [44] 中报告的LLAMA-3B模型。接下来的2比特模型源自BitDistiller [14]，这是一个用于增强超低比特LLM的开源QAT框架。原始配置包括INT2权重和FP16激活。在BitDistiller开源代码的基础上，我们进一步实现了基于LUT mpGEMM的INT8表格量化。评估指标与BitDistiller一致，包括WikiText-2数据集 [46] 上的困惑度（perplexity），MMLU [20] 上的5-shot准确率，以及多个任务上的zero-shot准确率 [2, 7, 48, 59, 79]。该实证研究的结果总结在表5中。第二数据行中的“N/A”表示 [44] 中未报告MMLU准确率。虽然2比特权重量化的性能不如原始 $W_{FP16}A_{FP16}$ LLAMA2-7B模型，但它仍然优于 $W_{FP16}A_{FP16}$ LLAMA-3B模型。值得注意的是，INT8表格量化并没有损害模型精度，显示出困惑度的微不足道的下降和

表 1：总体比较。

<table><tr><td>硬件配置</td><td>模型</td><td>BS1

SEQ2048

延迟</td><td>BS1024

SEQ1

延迟</td><td>峰值

性能</td><td>每SM TC

面积</td><td>TC计算

密度</td><td>TC能量

效率</td></tr><tr><td>A100† FP16 TC.</td><td>LLAMA 3B

(WFP16AFP16)</td><td>49.7%</td><td>106.71ms</td><td>41.15ms</td><td>312 TFLOPs</td><td>0.975mm²</td><td>2.96 TFLOPs/mm²</td></tr><tr><td>A100† INT8 TC</td><td>BitNet b1.58 3B

(WINT2AINT8)</td><td>49.4%</td><td>67.06ms</td><td>21.70ms</td><td>624 TOPs</td><td>0.312mm²</td><td>17.73 TOPs/mm²</td></tr><tr><td>A100†-LUT-4X*</td><td>BitNet b1.58 3B

(WINT2AINT8)</td><td>49.4%</td><td>42.49ms</td><td>11.41ms</td><td>1248 TOPs</td><td>0.187mm²</td><td>61.84 TOPs/mm²</td></tr><tr><td>A100†-LUT-8X*</td><td>BitNet b1.58 3B

(WINT2AINT8)</td><td>49.4%</td><td>38.02ms</td><td>7.47ms</td><td>2496 TOPs</td><td>0.373mm²</td><td>61.95 TOPs/mm²</td></tr><tr><td>H100† FP8 TC</td><td>BitNet b1.58 3B

(WFP8AFP8)</td><td>-</td><td>38.20ms</td><td>12.30ms</td><td>1525 TFLOPs</td><td>0.918mm²</td><td>12.59TFLOPs/mm²</td></tr><tr><td>H100†-LUT-4X*</td><td>BitNet b1.58 3B

(WINT2AFP8)</td><td>-</td><td>28.70ms</td><td>9.90ms</td><td>1525 TFLOPs</td><td>0.488mm²</td><td>23.69TFLOPs/mm²</td></tr><tr><td>H100†-LUT-8X*</td><td>BitNet b1.58 3B

(WINT2AFP8)</td><td>-</td><td>23.48ms</td><td>5.97ms</td><td>3049 TFLOPs</td><td>0.909mm²</td><td>25.40TFLOPs/mm²</td></tr></table>

由于缺乏关于A100/H100 Tensor Core及其7/4nm工艺的公开数据，$\dagger$ 表示数据归一化到28nm和1.41GHz，并尽我们所能进行了优化以进行公平比较。-LUT* 表示配备LUT TENSOR CORE并采用双倍寄存器建模的GPU。$\times$ 表示A100 FP16 Tensor Core阵列大小的倍数。TC. 指 Tensor Core。未报告 $A_{FP8}$ 的模型精度，因为BitNet是从头开始以 $A_{INT8}$ 格式训练的。先前的工作 [33, 47, 81] 表明，$A_{FP8}$ 在精度方面通常优于 $A_{INT8}$。

表 2：LUT TENSOR CORE 与 UNPU [38] 的比较：${W}_{\mathrm{{INT}}2}{A}_{\mathrm{{INT}}8}$ Tensor Core 案例。

<table><tr><td>配置</td><td>面积 (mm2)</td><td>归一化计算强度</td><td>功耗 (mW)</td><td>归一化能效</td></tr><tr><td>UNPU (DSE Enabled)</td><td>17,271.71</td><td>1×</td><td>23.39</td><td>1×</td></tr><tr><td>+ 权重重解释</td><td>13,116.60</td><td>1.317×</td><td>17.98</td><td>1.301×</td></tr><tr><td>+ 取反电路消除</td><td>12,780.05</td><td>1.351×</td><td>17.37</td><td>1.347×</td></tr><tr><td>+ DFG变换 + 内核融合 =LUT TENSOR CORE (本文提出)</td><td>11,991.29</td><td>1.440×</td><td>16.22</td><td>1.442×</td></tr></table>

表 3：LUT TENSOR CORE 与量化模型加速器的比较。

<table><tr><td></td><td>UNPU[38]</td><td>Ant[19]</td><td>Mokey[78]</td><td>FIGNA[25]</td><td>LUT TENSOR CORE</td></tr><tr><td>Act. 格式</td><td>INT16</td><td>flint4</td><td>FP16/32, INT4</td><td>FP16/32, BF16</td><td>FP/INT8, FP/INT16</td></tr><tr><td>Wgt. 格式</td><td>INT1~INT16</td><td>flint4</td><td>INT3/4</td><td>INT4/8</td><td>INT1~INT4</td></tr><tr><td>计算引擎</td><td>LUT</td><td>flint-flint MAC</td><td>Multi Counter</td><td>预对齐 INT MAC</td><td>LUT</td></tr><tr><td>工艺</td><td>65nm</td><td>28nm</td><td>65nm</td><td>28nm</td><td>28nm</td></tr><tr><td>PE 能效</td><td>27TOPs/W @0.9V (WINT1AINT16)</td><td>N/A</td><td>N/A</td><td>2.19× FP16-FP16 (WINT4AFP16)</td><td>63.78TOPs/W @0.9V DC (WINT1AINT8)</td></tr><tr><td>编译器栈</td><td>X</td><td>X</td><td>X</td><td>X</td><td>✓</td></tr><tr><td>评估模型</td><td>VGG-16, AlexNet</td><td>ResNet, BERT</td><td>BERT, Ro/DeBERTa</td><td>BERT, BLOOM, OPT</td><td>LLAMA, BitNet, BLOOM, OPT</td></tr></table>

任务准确率的轻微增加，这可能归因于量化的正则化效应。

# 5 讨论与局限性 (Discussion and Limitations)

**低比特训练与微调。** 目前，LUT TENSOR CORE 仅适用于低比特LLM的推理加速。最近的趋势表明，人们对LLM的低比特训练和微调越来越感兴趣 [11, 72]。虽然 LUT TENSOR CORE 的mpGEMM方法适用于低比特训练的前向传递，但训练过程的复杂性和稳定性仍然要求在后向传递中进行更高精度的计算。这涉及梯度和优化器状态等张量和计算，它们尚不完全兼容低比特格式。此外，训练效率受限于广泛的因素，如内存效率和通信效率，而不仅仅是GEMM性能。因此，优化低比特训练过程需要更全面的策略，可能需要能够拥抱更低精度的新训练算法

表 4：分离的表格预计算与融合的表格预计算的比较。通过算子融合，表格预计算开销可忽略不计。

<table><tr><td>模型</td><td>配置</td><td>Welder</td><td>Welder +预计算</td><td>Welder +融合预计算</td></tr><tr><td>OPT-175B</td><td>BS1SEQ2048</td><td>32.38 ms</td><td>38.77 ms</td><td>33.63 ms</td></tr><tr><td>OPT-175B</td><td>BS1024SEQ1</td><td>14.99 ms</td><td>17.43 ms</td><td>15.50 ms</td></tr><tr><td>BLOOM-176B</td><td>BS1SEQ4096</td><td>107.11 ms</td><td>129.85 ms</td><td>108.38 ms</td></tr><tr><td>BLOOM-176B</td><td>BS1024SEQ1</td><td>20.99 ms</td><td>26.05 ms</td><td>21.31 ms</td></tr><tr><td>LLAMA2-70B</td><td>BS1SEQ4096</td><td>34.68 ms</td><td>37.60 ms</td><td>35.65 ms</td></tr><tr><td>LLAMA2-70B</td><td>BS1024SEQ1</td><td>11.45 ms</td><td>15.21 ms</td><td>11.75 ms</td></tr></table>

表 5：LLAMA模型的表格量化分析。

<table><tr><td rowspan="2"># 模型配置</td><td rowspan="2">WikiText2 PPL ↓</td><td rowspan="2">MMLU 5s ↑</td><td colspan="6">Zero-shot 准确率 ↑</td></tr><tr><td>HS</td><td>BQ</td><td>OQ</td><td>PQ</td><td>WGe</td><td>平均</td></tr><tr><td>LLAMA2-7B WFP16AFP16 [65]</td><td>5.47</td><td>45.3</td><td>57.1</td><td>77.9</td><td>31.4</td><td>78.0</td><td>69.1</td><td>62.7</td></tr><tr><td>LLAMA-3B WFP16AFP16 [44]</td><td>10.04</td><td>N/A</td><td>43.3</td><td>61.8</td><td>24.6</td><td>72.1</td><td>58.2</td><td>49.7</td></tr><tr><td>LLAMA2-7B WINT2AFP16 [14]</td><td>7.68</td><td>30.5</td><td>49.2</td><td>70.2</td><td>25.8</td><td>73.8</td><td>63.1</td><td>56.4</td></tr><tr><td>LLAMA2-7B WINT2ALUT_INT8 [14]</td><td>7.69</td><td>30.61</td><td>49.2</td><td>70.0</td><td>26.2</td><td>73.7</td><td>63.5</td><td>56.5</td></tr></table>

和硬件创新，以支持训练工作流的复杂需求。我们将这些挑战确定为未来扩展 LUT TENSOR CORE 用于训练的潜在方向。

**长上下文注意力与KV缓存量化。** 解决长上下文问题是LLM能力的一个重要前沿 [13, 56]。在长上下文场景中，注意力机制往往成为计算瓶颈。当前的研究和实践表明，在预填充（prefilling）阶段，将注意力计算量化为FP8并不会显著损害模型精度 [60]。然而，超低比特精度对模型精度的影响仍大多未被探索。在解码阶段，几项研究表明，将KV缓存量化为4比特甚至2比特对模型性能的影响可以忽略不计 [22, 41]。鉴于Q矩阵保持高精度，该计算与mpGEMM一致。探索用于长上下文场景的 LUT TENSOR CORE 是未来研究的一个有前景的方向。

**更多数据灵活性与非整数权重。** 我们认为基于LUT的方法本质上适合灵活的精度组合，因为它用查表替换了主要的点积操作。目前，LUT TENSOR CORE 支持 $W_{\mathrm{INT}}A_{\mathrm{FP}}$ 和 $W_{\mathrm{INT}}A_{\mathrm{INT}}$ 组合。为了将其扩展到 $W_{\mathrm{FP}}$，我们的初步策略涉及将尾数和符号位像 $W_{\mathrm{INT}}$ 一样处理，将它们用作表索引。另一方面，指数位被视为移位器的输入。LUT方法也适应非整数权重格式。例如，在三值权重的情况下，LUT方法可以将三个三值权重打包成5比特，而基于ADD/MAC的方法需要6比特来表示相同的信息。

**支持mpGEMM的新兴趋势。** 新兴的GPU如B100 [8] 在Tensor Cores中原生支持混合精度GEMM [9, 50]。Blackwell引入了窄精度格式，如FP4、FP6、FP8及其变体NVFP4、MXFP4、MXFP6和MXFP8。它启用了一系列混合精度GEMM，包括 $A_{FP4,FP6,FP8} \times W_{FP4,FP6,FP8}$ 和 $A_{MXF4,MXF6,MXF8} \times W_{MXF4,MXF6,MXF8}$ 的组合，同时提供与 $W_{FP8}A_{FP8}$ Tensor Cores 相同的吞吐量。LUT Tensor Core 通过位串行方法支持这些操作，并在不同格式之间实现可扩展的性能。随着来自主要

图 19：传统 $W_{FP16}A_{FP16}$ Tensor Core 和来自 LUT Tensor Core 的 $W_{INT1}A_{FP16}$ 的Roofline分析。

厂商如NVIDIA的原生支持的出现，mpGEMM可能会成为一种关键且广泛采用的计算模式。

**LUT TENSOR CORE 的Roofline分析。** 图19展示了在A100内存系统上，传统 $W_{FP16}A_{FP16}$ Tensor Core 和基于LUT的 $W_{INT1}A_{FP16}$ Tensor Core 的Roofline图。X轴表示基于主存流量的运算强度（operational intensity）。来自 LUT TENSOR CORE 的 $W_{INT1}A_{FP16}$ Tensor Core 占用的面积仅为 $W_{FP16}A_{FP16}$ Tensor Core 的 $58.4\%$，但提供了 $4\times$ 的理论FLOPs。虽然原始 $W_{FP16}A_{FP16}$ 是计算受限的，但朴素的基于LUT的实现是内存受限的。通过软硬件协同优化工作——重新解释权重以将表格大小减半并减少激活内存流量，采用细长的分块以获得更好的数据复用，以及swizzling线程块以提高L2命中率——LUT TENSOR CORE 提高了运算强度，并将优化点推近了“脊点（ridge point）”。

# 6 相关工作 (Related work)

**低比特DNN加速器。** 随着LLM规模的增长，对低比特量化技术的需求日益增加，以减少模型大小和计算需求。硬件加速器已被开发出来，以有效地支持量化模型的更低位宽数据类型。NVIDIA的GPU架构反映了这一趋势，逐步纳入了更低精度的格式。从Fermi架构支持FP32和FP64开始，后续架构逐步包含了更低位宽的格式，如Pascal中的FP16，Turing中的INT4和INT8，以及Ampere中的BF16。在LLM时代，Hopper引入了FP8 [47]，Blackwell进步到了FP4 [57]。除GPU外，最近的研究提出了专门针对低比特量化DNN的定制加速器 [19, 35, 43, 58, 77, 78]。虽然这些进展展示了显著的进步，但它们主要集中在GEMM操作上，其中两个输入（权重和激活）共享相同的数据类型和位宽。FIGNA [25] 定制了一个 $W_{INT4}A_{FP16}$ 算术单元，以增强低比特LLM推理。LUT TENSOR CORE 利用基于LUT的计算范式提高了mpGEMM的效率，并提供了支持多种精度组合的灵活性，而无需复杂的硬件重新设计。

**稀疏DNN加速器。** 除了低比特量化，稀疏性是减小模型大小和加速DNN推理的另一种流行策略。稀疏性利用DNN权重矩阵或激活中固有的零值元素，在计算和存储中省略它们以提高效率。随着NVIDIA A100 GPU的出现，引入了稀疏Tensor Core，通过促进2:4结构化稀疏性 [6] 提供对稀疏性的原生支持。除了商用GPU，对定制稀疏DNN加速器的兴趣日益增长。这些设计旨在不同程度地利用稀疏性，通常采用剪枝（pruning）、零跳过（zero-skipping）和稀疏矩阵格式等技术来优化存储和计算 [17, 23, 24, 61, 70, 74, 85]。稀疏性在低比特LLM中也很普遍。当与量化结合时，稀疏性有可能产生更显著的效率增益。然而，有效地整合量化和稀疏性在保持模型精度和设计高效微架构方面构成了重大挑战。将稀疏性纳入 LUT TENSOR CORE 代表了一个有前景的研究方向，我们留待未来探索。

# 7 结论 (Conclusion)

本文提出了 LUT TENSOR CORE，一种基于LUT计算范式的软硬件协同设计，旨在为低比特LLM加速实现高效的混合精度GEMM操作。LUT TENSOR CORE 增强了性能，为各种精度组合提供了广泛的灵活性，并与现有的加速器架构和软件生态系统无缝集成。