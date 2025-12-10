# FIGLUT: An Energy-Efficient Accelerator Design for FP-INT GEMM Using Look-Up Tables

# FIGLUT：一种面向 FP-INT GEMM 的节能查找表加速器设计

Gunho Park $^{1,2*}$ , Hyeokjun Kwon $^{1*}$ , Jiwoo Kim $^{1}$ , Jeongin Bae $^{2}$ , Baeseong Park $^{2}$ , Dongsoo Lee $^{2}$ , and Youngjoo Lee $^{1}$

$^{1}$ Pohang University of Science and Technology (POSTECH),  $^{2}$ NAVER Cloud

{gunho PARK3} @navercorp.com, {kwon36hj, youngjoo.lee} @postech.ac.kr

 **摘要** ——仅权重量化（weight-only quantization）已经成为应对大型语言模型（LLM）部署难题的一种有前景的解决方案。然而，这类方法需要执行 FP-INT 运算，在通用 GPU 等硬件上实现较为困难。本文提出了一种高效的、基于查找表（LUT）的 GEMM 加速器架构 FIGLUT。FIGLUT 不再执行传统算术运算，而是根据权重模式从 LUT 中读取预先计算好的值，从而显著降低计算复杂度。我们同时提出了一种新型 LUT 设计，用于克服传统存储架构的限制。为了进一步提升基于 LUT 的运算效率，我们提出了一种半尺寸 LUT，并配合专用的解码与多路复用单元。FIGLUT 能在单一固定硬件配置下高效支持不同比特精度和多种量化方法。在相同 3 比特权重量化精度下，FIGLUT 相比最新加速器设计实现了  $59%$  更高的 TOPS/W 和  $20%$  更低的困惑度（perplexity）。在目标困惑度相同的条件下，通过执行 2.4 比特运算，FIGLUT 的 TOPS/W 提升达  $98%$ 。

---

# I. INTRODUCTION

# I. 引言

预训练大型语言模型（LLM）在各类语言理解和生成任务中展现出了卓越的性能。这些模型大多基于 Transformer 架构 [1], [3], [11], [29], [31]，其性能相对于参数规模呈现出可预期的幂律标度规律。随着这一规律的推进，模型规模被推至前所未有的高度，大幅拓展了自然语言处理的能力边界。然而，模型规模的大幅增长也带来了显著挑战，尤其是在内存占用方面 [26], [35]。例如，一个拥有 1750 亿参数的模型大约需要 350GB 的内存，这远远超过当前最先进 GPU 所提供的 80GB DRAM 容量 [28]。此外，内存带宽也成为关键瓶颈，导致严重的内存访问限制。该瓶颈会显著影响这些超大模型的部署性能与能效，因为数据传输速率往往跟不上 LLM 的计算需求。这种算力与带宽间的错配凸显了：迫切需要更高效的方法，将如此庞大的模型部署在硬件加速器上。

缓解 LLM 内存相关挑战的一条极具潜力的途径，是参数量化 [7], [15], [32]。在 LLM 场景中，人们已经逐渐转向 **仅权重量化（weight-only quantization）** ——即只将权重量化到子 4 比特精度，而保持激活为浮点（FP）格式 [4], [8]。这一转变的动机在于：在 LLM 中，权重的内存占用远大于激活，同时激活中存在的离群值（outliers）也进一步说明仅权重量化的有效性。尽管仅权重量化带来诸多优点，却必须依赖非标准的 FP-INT 运算，因此实现上仍存在持续的挑战。

近期研究提出了多种仅权重量化的高效 kernel，使得在 GPU 上可以获得实际的加速效果 [10], [25], [28], [36]。尽管 [10], [25] 中提出的 kernel 相比 cuBLAS——一套包含 GEMM 在内、用于加速高性能计算（HPC）应用的 GPU 库——表现出更快的速度，但这些改进在很大程度上源于权重压缩之后更高效的数据搬运。而在计算阶段中，被压缩的权重会被反量化回 FP 格式，导致实际算术运算仍然通过 FP-FP 单元执行，从而带来效率损失。NVIDIA 最新在 CUTLASS [21] 中发布的 FP-INT GEMM kernel，同样依赖底层的 FP 运算。为解决这些低效问题，一些研究提出了 LUT-based GEMM kernel [28], [36]，试图在不进行反量化的情况下直接执行 FP-INT GEMM 运算。然而，将 LUT 存储在 GPU 的共享内存中常常会导致存储体冲突（bank conflict）：多个线程同时访问同一个共享内存 bank，从而引发低效。

为了解决上述问题，最新的一些工作提出了专门面向仅权重量化模型、可以高效执行 FP-INT 运算的新型硬件架构 [20]。这些新兴解决方案试图在理论上仅权重量化的优势与现有硬件上的实际部署之间架起桥梁。例如，iFPU [22] 提出了一种高效的比特串行（bit-serial）架构加速器，使用二进制编码量化（Binary-Coding Quantization, BCQ）格式。在 BCQ 中，二值权重与输入激活的内积被替换为对输入激活的加或减。iFPU 引入了一种预对齐（pre-alignment）技术，将输入激活的指数与 FP 值对齐。因此，预对齐后的尾数部分可以由整数算术单元处理，从而高效地执行 FP-INT 运算。类似地，FIGNA [16] 提出了一种采用预对齐技术的 FP-INT4 乘法方案，同时缓解了 iFPU 所采用比特串行结构固有的低效。通过预对齐，FIGNA 将 FP-INT 乘法替换为对齐尾数与量化权重之间的整数乘法，从而提升仅权重量化模型的计算效率。然而，FIGNA 受限于固定精度（例如 4 比特权重量化），因此对于低于 4 比特精度的模型，或采用混合精度量化方法的模型，其效率会受到限制。

在本文中，我们提出了一种新的加速器架构 FIGLUT，通过 LUT-based GEMM 来降低比特串行加速器的计算复杂度。FIGLUT 使用权重模式作为键，利用查找表替代算术运算，大幅提升计算效率。我们引入了一种专用算子——读累加单元（read-accumulate，RAC），用来替换传统硬件加速器中的乘加单元（MAC）。FIGLUT 能够高效支持仅权重量化模型的 FP-INT 运算，并采用比特串行结构，在单一硬件框架下处理不同精度的计算。此外，通过使用 BCQ 作为权重表示格式，FIGLUT 不仅可以对当前性能最优的 BCQ 模型进行加速，也可以支持常见的均匀量化模型。因此，FIGLUT 能在一套硬件平台上，以高效方式同时支持不同比特精度和不同量化方法，为仅权重量化模型的部署带来显著进步。

本文的主要贡献如下：

* 我们提出了一种基于 LUT 的 FP-INT GEMM 方法，通过 LUT 读操作替换算术运算，实现能效更高的 GEMM 运算。
* 我们设计了一种新型专用 LUT 架构，使并行处理过程中能够无存储体冲突地同时访问 LUT。
* 我们提出了 FIGLUT，这一创新的 LUT-based FP-INT GEMM 加速器，能够在单一硬件平台上高效支持多种量化方法及不同精度，并充分利用 LUT-based 运算。

---

# II. BACKGROUND

# II. 背景

# A. Methods and Systems for Weight-Only Quantization

# A. 仅权重量化的方法与系统

量化是一种通过将高比特精度的浮点数（FP）转换为低比特精度整数（INT）来降低计算复杂度和内存占用的有力手段。然而，将量化应用于 LLM 时面临显著挑战，尤其是在尝试同时将权重与激活动态地量化到低比特精度时。相比权重，激活更难以量化，这一方面源于其存在离群值，另一方面在于其分布具有较强的动态性，从而难以被准确建模 [9], [34]。这些因素促使人们需要专门的技术来有效处理 LLM 中的激活量化问题。

为解决 LLM 推理中内存带宽受限以及 DRAM 容量有限等问题，研究者广泛探索了仅权重量化方法。这类方法通过保留激活为 FP 格式来维持精度。由于简单的“舍入到最近”（round-to-nearest, RTN）量化会造成明显精度下降，已有大量工作提出技术来最小化量化误差。OPTQ [10] 提出了一种均匀量化方法，利用来自校准数据集的近似二阶信息，实现了亚 4 比特量化且精度损失可以忽略。AWQ [25] 提出了一种仅权重量化方法，通过基于激活分布选择关键权重来最小化量化误差。近期，一些基于 BCQ 的非均匀量化技术也展示了优异表现。ShiftAddLLM [36] 通过对权重进行后训练的按位移位与加法重参数化，并采用非均匀 BCQ，将 LLM 构建为“无乘法”模型，从而取得最先进结果。此外，ShiftAddLLM 还基于各层敏感度采用混合精度量化，在精度与效率之间取得更好的折中。

随着仅权重量化方法的高效性得到验证，各类硬件加速器被提出以支持激活与权重之间的 FP-INT 运算。表 I 比较了不同硬件加速器的特性。商用 GPU 缺乏可以直接处理 FP 激活与 INT 权重的 FP-INT 算术单元。因此需要反量化过程，将 INT 权重转换回 FP 格式，以适配现有的 FP-FP 算术单元。[10] 和 [25] 中提出的 FP-INT GEMM kernel，得益于权重压缩带来的带宽利用效率提升，相比 cuBLAS 的 FP-FP GEMM 具有更好的延迟表现，这一点在 LLM 这类内存带宽受限的应用中尤为明显。然而，由于 GPU 不具备原生 FP-INT 运算单元，最终仍需通过 FP-FP 运算完成计算，削弱了权重量化本应带来的优势。

为解决 GPU 中的低效问题并高效处理 FP-INT GEMM 运算，已有若干加速器架构被提出。iFPU [22] 提出了一种高效处理 BCQ 格式权重与 FP 激活之间计算的方法：其通过根据最大指数值对输入激活的尾数进行预对齐，使 FP-INT 运算可以被尾数与二值权重之间的 INT-INT 加法所取代，从而降低硬件复杂度。尽管这种比特串行方式可以支持混合精度量化，但其计算复杂度与量化权重的比特精度  $q$  成正比。

FIGNA [16] 通过提出一种 FP-INT 算术单元来缓解比特串行架构的计算开销，该单元执行预对齐尾数与均匀量化权重之间的 INT-INT 乘法。尽管这种方法有效减少了计算量，但其受限于固定精度，只适用于均匀量化模型，无法支持子 4 比特精度以及基于 BCQ 的高级非均匀量化方法。因此，需要一种新的比特串行硬件架构，在最小化计算开销的同时，能够支持更广泛的精度和量化方式，包括采用 BCQ 的非均匀量化。

---

**TABLE I COMPARISON OF DIFFERENT HARDWARE ACCELERATORS**

**表 I 不同硬件加速器的对比**

---

# B. Binary Coding Quantization

# B. 二进制编码量化（BCQ）

Binary Coding Quantization（BCQ）[33] 是一种将实值权重  $w \in \mathbb{R}$  转换为  $q$ 比特精度表示的量化技术。其思想是将  $w$  表示为一组二值权重  ${b_i} *{i=1}^q$  与对应缩放因子  ${\alpha_i}* {i=1}^q$ 的线性组合，其中  $b_i \in {-1,1}$  [36]。二值权重和缩放因子通过如下目标函数进行优化，以最小化量化误差：

$$
\underset {\alpha_ {i}, b _ {i}} {\arg \min } \left| w - \sum_ {i} \alpha_ {i} b _ {i} \right| ^ {2}. \tag {1}
$$

由于该目标不存在解析解，通常需要采用启发式方法来近似求解最优的缩放因子与二值矩阵。

BCQ 既可以用于激活量化，也可以用于权重量化，但当应用于激活量化时会带来更高的计算复杂度 [17]。为了在保持精度的前提下专注于仅权重量化，本文仅在权重量化背景下讨论 BCQ。因此，给定一个 FP 激活向量  $\mathbf{x} \in \mathbb{R}^{n \times 1}$ ，其与经二值矩阵  $\mathbf{B}_i \in {-1,1}^{m \times n}$  和缩放因子  $\alpha_i \in \mathbb{R}^{m \times 1}$  量化后的权重矩阵相乘，可以表示为：

$$
\mathbf {y} = \sum_ {i = 1} ^ {q} \left(\alpha_ {i} \circ \left(\mathbf {B} _ {i} \cdot \mathbf {x}\right)\right) \tag {2}
$$

其中符号  $\circ$  表示 Hadamard 逐元素乘积。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/2f98b870edb1534bce6211c63fd60692f647640a079c0cecd0adeade84334fdc.jpg)

(a) Non-uniform quantization

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/9e865af57ff4a48bd1078477cedc511acd028ac5605a594c0b86e102c4524431.jpg)

(b) Uniform quantization

Fig. 1. Extension of binary-coding quantization to support both non-uniform and uniform quantization formats, achieved by including a offset term  $(q = 3)$ .

图 1. 通过引入偏移项扩展二进制编码量化，使其支持非均匀量化与均匀量化格式示意（ $q = 3$ ）。

传统 BCQ 作为一种非均匀量化方法，与大多数采用均匀量化的最新模型并不兼容。[28] 在 BCQ 中引入偏移项  $z$ ，从而增强其表示能力：

$$
w = \sum_ {i = 1} ^ {q} \left(\alpha_ {i} \cdot b _ {i}\right) + z \tag {3}
$$

已有研究表明，通过适当调整缩放因子与偏移项，均匀量化可以在这一框架中得到有效表示 [28]。如图 1 所示，传统 BCQ 使用多组不同的缩放因子，而偏移项的引入则使其能够方便地表示均匀量化数值。

---

# C. Bank Conflict

# C. 存储体冲突（Bank Conflict）

为了高效执行并行运算，需要在同一时间访问多份数据。为实现这种并行访问，许多并行处理器采用**存储体分组（memory banking）** 技术，将内存划分为多个独立的 bank，使多个线程能够并发访问。然而，当单个时钟周期内所需访问的数据恰好落在同一 bank 中时，就会出现**存储体冲突（bank conflict）** [12]。这一问题在 GPU 中尤为突出，因为 GPU 使用共享内存空间来实现线程之间的高效并行操作。当不同线程同时访问同一个 bank 时，就会发生存储体冲突。这会阻碍理论峰值性能的实现，并造成算力浪费：原本可以并行执行的操作被迫串行化。

此外，当基于 LUT 的方法部署在现有硬件平台上时，存储体冲突也会成为重要瓶颈。以 LUT-GEMM [28] 为例，它将 LUT 存储在 GPU 的共享内存中，并使用多个线程执行相关操作。在 LUT 构建阶段，由于每个线程被设计为并行访问不同的 bank，因此可以避免冲突。而在 LUT 读取阶段，由于权重模式具有随机性，往往会频繁触发 bank 冲突，从而导致性能下降。图 2 展示了多个线程访问存储体的示例。在最坏情况下，同一周期内所有本应并行的访问都被串行化，带来巨大的性能开销。因此，要使基于 LUT 的运算获得最佳性能，必须尽量避免存储体冲突，这就需要一种新的 LUT 架构，使多个并行算子能够在无冲突的情况下同时从 LUT 读取不同的值。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/b299325b96297766553d7f7d7425d9da9ed563649eff4917a91ca8507f5bd38a.jpg)

(a) Ideal case

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/8cdcaf678a5d13da581996ae49f42f06f4a6dd57a1a6358453243e9ab0c4b9b4.jpg)

(b) Worst case

Fig. 2. Comparison of bank conflicts during shared memory access.

图 2. 共享内存访问时存储体冲突情况的对比示意。

---

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/585af3c3b4c717b15f8ddf6dbf94b16b8fbc2751dcc0aa8bc70978bd9cded2e5.jpg)

Fig. 3. Illustration of look-up table based FP-INT GEMM.

图 3. 基于查找表的 FP-INT GEMM 示意图。

**TABLE II EXAMPLE OF LOOK-UP TABLE WHEN  $\mu = 3$**

**表 II  当  $\mu = 3$  时查找表示例**

# III. LOOK-UP TABLE-BASED FP-INT GEMM ACCELERATOR

# III. 基于查找表的 FP-INT GEMM 加速器

在本节中，我们介绍 FIGLUT，这是一种将计算过程替换为从查找表（LUT）中读取数据，从而提升 FP-INT GEMM 运算效率的新型计算方法与加速器架构。为了解决现有基于 LUT 的方法中阻碍并行处理的 LUT 存储体冲突（bank conflict）问题，我们提出了一种专用的 LUT 结构。该结构旨在消除存储体冲突，使多个算子能够高效并行访问 LUT 数据。我们还分析了在共享 LUT 的情况下扇出（fan-out）增加对功耗的影响，并据此提出一种最优的处理单元（Processing Element，PE）结构，以最大化能效。得益于这些设计，FIGLUT 相较于传统计算单元在能量效率方面具有显著优势。

---

# A. Look-up Table based FP-INT GEMM

# A. 基于查找表的 FP-INT GEMM

查找表广泛用于通过查表替代直接计算，以提升计算效率，尤其是在结果取值来自一个预先限定集合的情况下。该技术可以扩展到涉及二值权重矩阵  $\mathbf{B} \in {-1, +1}^{M \times N}$  的深度学习运算，此时输出激活由输入激活向量  $\mathbf{x} \in \mathbb{R}^{N \times 1}$  的加减运算得到。因此，输出激活被限制为输入激活的各种线性组合，从而可以将所有可能的输出结果预先计算并存储于 LUT 中。这样一来，实际的算术运算就被从 LUT 中读取数据的操作所替代。

为构建 LUT，我们引入一个超参数  $\mu$ ，用于指定参与构成查找键的二值权重数量。需要注意的是，随着  $\mu$  的增大，一次 LUT 读取能够替代的运算次数也随之增加；但与此同时，用于存储 LUT 的存储空间则呈指数级增长，因此选择合适的  $\mu$  值至关重要。考虑如下二值矩阵  $\mathbf{B} \in {-1, +1}^{4 \times 6}$  和输入激活向量  $\mathbf{x} \in \mathbb{R}^{6 \times 1}$ ：

$$
\mathbf {B} = \left[ \begin{array}{r r r r r r} + 1 & - 1 & - 1 & - 1 & - 1 & + 1 \ + 1 & - 1 & - 1 & + 1 & + 1 & - 1 \ - 1 & + 1 & - 1 & - 1 & - 1 & + 1 \ + 1 & - 1 & - 1 & - 1 & - 1 & + 1 \end{array} \right], \tag {4}
$$

$$
\mathbf {x} = \left[ \begin{array}{l l l l l l} x _ {1} & x _ {2} & x _ {3} & x _ {4} & x _ {5} & x _ {6} \end{array} \right] ^ {\top}. \tag {5}
$$

例如，在  $\mu = 3$  时计算  $\mathbf{B}\mathbf{x}$  的过程中，可以观察到诸如  $(x_{1} - x_{2} - x_{3})$  和  $(-x_{4} - x_{5} + x_{6})$  之类的运算会被多次重复。随着模型规模增大、矩阵维度增长，这类冗余运算的次数会显著增加。如表 II 所示，通过预先计算所有可能的输入组合并将结果存入 LUT，可以避免重复计算，使单次 LUT 读取即可替代多次加法运算。如果将  $\mu$  增加到 6，则会出现如  $(x_{1} - x_{2} - x_{3} - x_{4} - x_{5} + x_{6})$  这样的重复计算，同样可以通过 LUT 高效处理。尽管增大  $\mu$  能带来更多计算上的收益，但存储 LUT 所需的内存空间会呈指数级增长。

图 3 给出了在  $\mu = 2$  时计算  $\mathbf{B}\mathbf{x}$  的 LUT-based FP-INT GEMM 的整体流程示意。正如前文所述，为了减少冗余计算并高效执行 GEMM，我们基于输入激活  $\mathbf{x}$  的  $\mu$  个元素构建 LUT，并针对不同输入动态生成 LUT。每个 LUT 包含  $2^{\mu}$  个值，使用  $\mu$  位二值权重作为键来索引对应的 LUT 元素。在查询阶段，

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/e5cf0e3e6022b01e91ec98b84eb3dee3776f741f9e1b5ab464d2a745dbd5dad0.jpg)
Fig. 4. Overall MPU architecture of FIGLUT

图 4. FIGLUT 的整体 MPU 架构

共有  $k$  个读累加单元（RAC）同时访问该共享 LUT，以读取各自对应的部分和。在该示例中， $k = 4$ ，表示有四个 RAC 可以并行访问同一个 LUT。支撑这一并发访问能力的专用 LUT 结构会在后续小节中详细介绍。最后，从 LUT 中读取的值会被累加器收集。当对所有二值矩阵  $\mathbf{B}_i$  完成这一步骤后，即可得到最终的输出激活。需要注意的是，前面的示例为了简化说明，输入激活假设为向量（对应 GEMV 运算），而实际上该方法同样适用于 GEMM：通过为每个 batch 生成独立的 LUT，即可高效执行矩阵乘法运算。

---

# B. Overall FIGLUT Architecture

# B. FIGLUT 的整体架构

图 4 展示了所提出的基于 LUT 的 FP-INT GEMM 加速器 FIGLUT 的整体架构。本小节将详细介绍其中的矩阵处理单元（Matrix Processing Unit，MPU），这是该架构的核心贡献部分。整个系统架构的完整说明将在 III-F 小节中给出。FIGLUT 采用与 Google TPU [18] 类似的二维脉动阵列结构。输入数据通过脉动数据流从输入存储器中按顺序读取，并被送入 LUT 生成器以构建 LUT。每个 LUT 生成器接收  $\mu$  个输入激活，并在线生成所有可能的输出部分和；这些部分和随后被写入各个 PE 内部的 LUT，作为 LUT 的表项。为实现有效的数据复用，每一行 PE 中产生的 LUT 值会被传递到下一行的 LUT 中继续使用。每个 PE 内部包含  $k$  个 RAC，由于 LUT 采用了专为 LUT 运算设计的特殊结构，这  $k$  个 RAC 可以同时并行访问同一个 LUT。我们会对  $k$  的取值进行分析，在考虑多个 RAC 同时访问 LUT 所导致的扇出（fan-out）开销后，确定最优的  $k$  值，这部分内容将会在后续章节详细展开。

FIGLUT 采用权重常驻（weight-stationary）数据流来优化 GEMM 运算。在每个 PE 内部，RAC 包含一个寄存器，用于存储一个  $\mu$ 位的权重模式，该模式作为查找键访问 LUT 中的数据。各个 PE 生成的部分和会沿着列方向逐步累积，并传递到右侧相邻列的 PE 中，这一迭代累加过程会在各列 PE 间持续进行，直到最终列的 PE 得到最终部分和。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/4d5cc0eae1095be6751f97ea2a958acea6d56573e3c0c7515dd6aae22792cc0e.jpg)
(a) FP-INT

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/81f1477260b716c63f7574c2ca8ef9daad5f028eb0799a73652dc293c00fe3ac.jpg)
(b) FP-BCQ

Fig. 5. Illustration of the input and weight tile fetching sequence in (a) a systolic array accelerator for INT weight and (b) FIGLUT for BCQ weight. The arrows indicate the order in which the tiles are processed.

图 5. (a) 使用 INT 权重的脉动阵列加速器与 (b) 面向 BCQ 权重的 FIGLUT 中输入与权重 tile 的提取顺序示意。箭头表示各 tile 的处理顺序。

该累加过程沿着每一列的 PE 进行，直到在最后一列的 PE 中得到最终的部分和。最终部分和会与缩放因子  $\alpha_i$  相乘，并写入累加器缓冲区（accumulator buffer）。这一流程会对每一个权重 bit 平面重复执行一次；在处理完所有 bit 平面之后，这些累加结果会与偏移量（offset）相加，最终写入输出缓冲区。

图 5 展示了输入与权重 tile 的提取顺序。对于使用 INT 权重的脉动阵列加速器（例如 FIGNA [16]），每个权重对应单通道多比特表示。相比之下，在 FP-BCQ 加速器（例如 FIGLUT 和 iFPU [22]）中，权重由  $q$  个二值 bit 平面构成。在权重常驻数据流中，无论是 FP-INT 还是 FP-BCQ 加速器，一个权重 tile 被加载一次后会被重复使用，而输入 tile 则按顺序（图中标注 (1)）依次加载。然而，对于 FP-BCQ 加速器来说，流程会有所不同：在处理完当前 bit 平面的同一权重 tile 后，不是加载下一个空间位置的权重 tile，而是加载下一 bit 平面的权重 tile 并进行处理（如图 5(b) 中的 (2) 所示）。在完成所有二值权重 bit 平面的处理后，再转到下一个权重 tile，其整体流程与 FP-INT 加速器类似。

---

# C. Conflict-free LUT Structure

# C. 无冲突 LUT 结构

要高效执行基于 LUT 的 FP-INT GEMM 运算，需要一种专门设计的 LUT 结构，因为传统存储结构难以同时满足以下需求：
首先，必须尽量降低从 LUT 读取数据的功耗。由于 FIGLUT 是用查表操作替代算术运算，只有当读取成本显著低于直接计算时，整体效率才能得到提升。其次，该结构需要支持大量的并行读写操作。考虑到激活值具有高度动态性，LUT 需要能够按需在线重构，即根据输入激活的变化对 LUT 内容进行快速更新，从而要求其支持大规模并行写入。从数据读取的角度看，多个 RAC 会并行工作，每个 RAC 使用不同的权重模式作为键来访问 LUT 中的数据，

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/726abe237d9e7cb4e990f8a0bd4021aa2e7ee57d2538ed1507f4173f84059fe3.jpg)
Fig. 6. Relative Power Consumption of RFLUT and FFLUT Compared to FP Adder Baseline Across Different  $\mu$  Values

图 6. 不同  $\mu$  取值下，RFLUT 与 FFLUT 相对于 FP 加法器基线的归一化功耗比较

因此，LUT 架构必须在支持高度并行访问的同时避免存储体冲突（bank conflict）。

为了确保从 LUT 中读取预计算结果所消耗的功率低于使用 FP 加法器的情况，我们首先测量了单个 FP 加法器的功耗作为基线。随后，将其与基于 LUT 的实现所消耗的功率进行对比。图 6 展示了在保证等效吞吐率的前提下，不同  $\mu$  值下 FP 加法器与多种 LUT 实现之间的功耗比较。针对目标硬件约束，我们实现了两种 LUT 结构：(1) 传统的基于寄存器文件的 LUT（Register File based LUT，RFLUT），以及 (2) 所提出的基于触发器的 LUT（Flip-Flop based LUT，FFLUT）。

RFLUT 使用类似于 CPU 寄存器堆的寄存器文件结构实现，其读地址由权重模式直接构成。为了进行公平比较，我们使用  $28\mathrm{nm}$  工艺的存储编译器生成 RFLUT 宏块，以便能够针对低功耗 LUT-based GEMM 运算进行定制设计。生成的寄存器文件作为 IP 宏集成进整体硬件设计中。包括 FFLUT 在内的所有 LUT 结构的功耗分析，均基于在 ICC2 中完成布图布线（P&R）后的物理版图结果进行评估。由于在该环境下  $\mu = 2$  的 RFLUT 尺寸过小，编译器无法生成对应的宏块，因此未包含在测量中。总体而言，RFLUT 的读取功耗普遍高于 FP 加法器。尽管  $\mu = 4$  的 RFLUT 相比  $\mu = 8$  时单次读取的功耗更低，但它需要执行两倍数量的读取操作，导致总功耗更高。因此，RFLUT 并不适合用于本文所采用的 LUT-based 方法，有必要引入新型 LUT 结构。

为克服传统 RFLUT 的局限性，我们提出了一种基于触发器的查找表结构——FFLUT，其架构如图 7 所示。FFLUT 由一组通过多路复用器访问的触发器（flip-flops）构成。在 LUT 生成阶段，一旦某个触发器被写入并使能，它就会持续输出该值，直到被重置。这种设计使得处理单元可以通过将相应的权重键送入多路复用器，来选择并读取对应触发器中的值。与受读/写端口数量限制的 RFLUT 不同，FFLUT 允许多个单元同时访问不同的键，而不会产生存储体冲突。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/1859ca92832dceaec085d92c65528612544fa1511db2a4285586fa16fc972bdf.jpg)
Fig. 7. Architecture of the Flip-Flop based Look-Up Table (FFLUT)

图 7. 基于触发器的查找表（FFLUT）架构

这种并发访问能力是通过在 FFLUT 前端配置专用多路复用器实现的，不同单元可在同一时刻访问不同的键。由此，多个单元能够高效共享一个 LUT，大幅降低 LUT-based 运算的额外开销，并显著提升整体能效。

为了进一步优化 FFLUT 的数据读取过程，我们引入了一种专用算子——RAC（Read-Accumulate）单元，用以替代传统的 MAC（Multiply-Accumulate）单元。正如图 4 所示，FIGLUT 架构中的每个 PE 包含一个 LUT 和多个 RAC 单元。与先进行乘法再对部分和累加的 MAC 不同，RAC 直接使用权重模式作为键从 LUT 中取值，并将读取到的数据累加到部分和中。因此，RAC 能够更高效地执行基于 LUT 的运算，从而提升 FIGLUT 整体架构的效率。

为了实现高能效的 GEMM 运算，我们在架构设计中引入了一个优化搜索过程，使多个 RAC 单元可以共享同一个 LUT。该优化主要包括两个步骤：
第一，确定 LUT 的最优  $\mu$  值，在计算量减少带来的收益与 LUT 的功耗之间取得平衡；
第二，在该  $\mu$  取值下，通过评估扇出对功耗的影响，确定每个 LUT 最佳的 RAC 共享数量。

为此，我们引入变量  $k$  表示单个 PE 中 LUT 的扇出，即共享同一个 LUT 的 RAC 单元数量。图 8 展示了在  $\mu = 2$  与  $\mu = 4$  时，随着  $k$  变化 FIGLUT 的功耗情况。相对功耗的基线由使用 FP 加法器完成相同运算所得的结果设定。由于  $\mu = 8$  时 LUT 的面积和功耗过高（如图 6 所示），因此不纳入考虑范围。随着  $k$  的增加，由于多个 RAC 共享同一个 LUT，LUT 的总数量减少，从而使得 LUT 部分的功耗逐渐下降。需要注意的是，完成同一工作负载所需的 RAC 总数仅由  $\mu$  决定，与  $k$  无关，因此总的读累加操作次数不会因共享方式不同而发生变化。然而，当  $\mu$  较小时，为保持相同吞吐量，需要更多的 RAC 单元，因此在  $\mu = 2$  时 RAC 的功耗高于  $\mu = 4$ 。当 LUT 不共享（ $k = 1$ ）时， $\mu = 4$  所需 LUT 的尺寸更大，导致其总功耗高于  $\mu = 2$ 。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/0afaea91d3f5f12eef2cfc3f64a885fa8ffa33f5b22528f9d9c8735e16d1be2c.jpg)
Fig. 8. Relative power comparison of baseline,  $\mu = 2$  and  $\mu = 4$  for various the number of RACs per LUT configuration.

图 8. 不同每 LUT RAC 数配置下，基线与  $\mu = 2$ 、 $\mu = 4$  的相对功耗比较

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/e36c7d2dd57d177874e6bae680fb4c0794df196e6925426b68d9da0665a1095e.jpg)
Fig. 9.  $P_{RAC}$  and  $P_{PE}$  analysis for various RAC numbers. Normalized by  $k = 1$  results.

图 9. 不同 RAC 数量下  $P_{RAC}$  与  $P_{PE}$  的功耗分析，相对值以  $k = 1$  归一化

因此，在不共享 LUT（ $k = 1$ ）的情况下，由于  $\mu = 4$  的 LUT 尺寸更大，其总功耗会高于  $\mu = 2$  的情形。然而，随着  $k$  逐渐增大，多个 RAC 共享同一个 LUT，使得 LUT 总数量减少，从而整体功耗下降。基于这些观察结果，并结合架构中可采用足够大的  $k$  这一事实，本文最终选择在 FIGLUT 中使用  $\mu = 4$  的配置。

在设计能够在多个 RAC 之间共享单个 LUT 的高能效加速器时，如何控制扇出是一个关键挑战。在数字电路中，扇出是指一个信号驱动多个电路单元，此过程会增加功耗并带来信号延迟。在 FIGLUT 的场景中，当多个 RAC 同时访问 LUT 时，为保证 LUT 输出能够稳定地驱动所有 RAC，扇出效应会导致额外的驱动功耗。虽然通过在多个 RAC 间共享 LUT 可以减少 LUT 的总数量，从而降低整体 LUT 面积与静态功耗，但过大的扇出会提高单个 LUT 的动态功耗。因此，需要在“减少 LUT 数量”与“控制扇出引起的额外功耗”之间小心权衡。有效解决这一问题是降低 LUT-based GEMM 总功耗并维持高能效的关键。

图 9 展示了在不同 LUT 扇出配置下，单个 PE 的总功耗  $(P_{PE})$  以及每个 RAC 的平均功耗  $(P_{RAC})$  的变化情况。为评估扇出对功耗的影响，所有实验结果均基于完成布图布线后的 PE 物理版图（包括 LUT 与 RAC）的功耗分析得到。

**TABLE III COMPARISON OF RELATIVE POWER CONSUMPTION OF LUT AND OTHER COMPONENTS**
**表 III LUT 与其他组件相对功耗比较**

<table><tr><td></td><td>LUT</td><td>MUX</td><td>Decoder</td><td>MUX+Decoder</td></tr><tr><td>FFLUT</td><td>1.000</td><td>0.003</td><td>0.000</td><td>0.003</td></tr><tr><td>hFFLUT</td><td>0.494</td><td>0.002</td><td>0.003</td><td>0.005</td></tr></table>

在布图布线完成后，我们对包含 LUT 与 RAC 在内的 PE 进行功耗分析。每个 RAC 的平均功耗  $P_{RAC}$  通过  $P_{PE} / k$  计算得到。随着  $k$  增加，单个 PE 中 RAC 的数量增加，导致  $P_{PE}$  总体上升。起初，增大  $k$  会降低  $P_{RAC}$ ，因为共享 LUT 的功耗被摊分到更多的 RAC 上。然而，当  $k$  继续增大时，由扇出带来的额外驱动功耗开始占主导，使  $P_{RAC}$  重新上升。基于这一观察，我们在架构中选择  $k = 32$  作为最优配置。也就是说，每个 PE 中包含一个共享 LUT 和 32 个 RAC 单元。

---

# D. Optimizing LUT using Vertical Symmetry

# D. 利用垂直对称性优化 LUT

在基于 LUT 的 FP-INT GEMM 中，与传统 GEMM 相比最主要的额外开销来自 LUT 本身。因此，降低 LUT 的功耗对提升 FIGLUT 的整体效率至关重要。FFLUT 是由一组触发器构成的，其功耗基本上随 LUT 规模线性增长。为此，我们提出了一种半尺寸 FFLUT（half-FFLUT，hFFLUT）技术，将 LUT 的大小压缩为原来的一半。

如表 II 所示，LUT 的取值具有**垂直对称性**。换句话说，由于 LUT 中每个表项都是由  $\mu$  个 FP 值通过加减组合得到，因此对于任意一种组合，总存在另一种组合，其结果只需对前者取符号反转即可得到。利用这一性质，我们仅存储表的“上半部分”，然后在查询时根据键对最终 LUT 值进行解码。图 10 以  $\mu = 3$  为例展示了 hFFLUT 以及相应解码逻辑的模块框图。在该结构中，键的最高位（MSB）用作选择信号，用于选出供 hFFLUT 访问的  $(\mu - 1)$ 位键；从 hFFLUT 读出值后，再根据 MSB 对其符号进行翻转，从而得到完整的 LUT 输出值。

尽管 hFFLUT 有效地减小了 LUT 的规模，但它也引入了额外的解码开销。表 III 对比了 hFFLUT 的解码电路（包括多路复用器）与 FFLUT 前端多路复用器在硬件复杂度和功耗方面的差异。可以看到，hFFLUT 的解码电路功耗略高于 FFLUT 所需的多路复用器。然而，由于 LUT 本身在整体功耗中占主导地位，解码部分的额外开销相对而言可以忽略不计。综合来看，所提出的 hFFLUT 技术能够在系统层面将 LUT 的功耗近乎减半，从而显著提升 FIGLUT 的能量效率。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/81faf167ced554f3f2d084392f72ba5c7e2aa746eab982aaa5f25613c7b7afd4.jpg)
(a) FFLUT

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/ef3772a0c1c482d819c2204fe787564fbd49ae1e2d715d889359b124d0847611.jpg)
(b) hFFLUT

---

# E. Efficient LUT Generator

# E. 高效 LUT 生成器

为了根据不断变化的输入激活动态计算 LUT 元素，需要一个高效的表项生成器。我们实现了一个两级树形结构的硬件模块：第一步对输入的符号进行调整（加/减），第二步并行累加，从而一次性生成多个 LUT 值。得益于 hFFLUT 仅需存储一半表项的特性，该生成器只需预计算所有模式的一半即可。正如图 11 所示，对于给定的高位模式（图中黄色区域），底层低位模式会重复出现，这使我们能够减少针对高位模式的加法操作；重复使用的低位部分（绿色区域）只需计算一次，之后与不同的高位结果并行组合。通过将高 2 位模式与低 2 位模式相结合，便可生成 hFFLUT 所需的全部表项。与直接实现的 LUT 生成器相比，这种设计可将加法器数量与总加法操作次数减少约  $42%$ 。在  $\mu = 4$  且假设所有 LUT 元素都被使用的情况下，我们的 LUT 生成器总共需要 14 次加法即可计算出完整的 LUT；而若不使用 LUT、直接对每个结果进行累加，则每个结果需要  $\mu - 1 = 3$  次加法。因此，当  $k > 4$  时，在生成相同数量的结果时，所提出的 LUT 生成器的加法操作次数少于具有  $k$  个 RAC 的直接硬件实现。随着  $k$  的进一步增大，该方法在加法次数上的节省会更加显著，从而实现更高效的 LUT 生成。

---

# F. System overview

# F. 系统概览

图 12 展示了整个系统的总体结构，其中包含 FIGLUT 模块。作为一个硬件加速器，FIGLUT 在设计上具有较强的灵活性，并不依赖某种特定的系统配置，因此可以方便地集成到各种不同的系统架构中，而不仅限于本文给出的示例系统。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/92b68155d58d664f455c7f6613e73961dd589b338206e5aa2de467b9bcd9dffc.jpg)
Fig. 11. Required generating pattern and LUT generator module for  $\mu = 4$ .

图 11.  $\mu = 4$  时所需的生成模式与 LUT 生成器模块

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/b05c04e888abb27b54e26f64ea9d23e0a5ba0ce4a1d81e51817fc844c5589316.jpg)
Fig. 10. Block diagram of the (a) FFLUT and (b) hFFLUT with proposed decoder.
Fig. 12. System architecture of FIGLUT.

图 10. (a) FFLUT 与 (b) hFFLUT 及其解码结构框图
图 12. FIGLUT 的系统架构

该系统在主机与 FIGLUT 之间采用共享内存结构，而非各自独立的存储空间。主机与 FIGLUT 之间的数据传输通过 AXI 总线完成。在该架构下，无需显式地将计算结果从 FIGLUT 拷贝回主机内存；主机可以直接访问共享内存以读取最终结果，从而减少主机与加速器间的数据搬运开销，提升整体系统效率。

FIGLUT 内部包含用于存储计算数据的片上缓冲区、执行矩阵运算的 MPU，以及用于处理非 GEMM 运算的向量处理单元（Vector Processing Unit，VPU）。为最大化效率，FIGLUT 采用基于 tile 的 GEMM 计算方式：每个 tile 所需的数据会通过双缓冲技术预先加载并保存在片上缓冲区中。MPU 负责执行基于 LUT 的 GEMM 运算，并将中间计算结果以部分和的形式存储在 Psum 缓冲区中以供后续使用。最终的 GEMM 结果由 VPU 进行进一步处理，并写入统一的输出缓冲区。该设计有效减少了对片外存储器的访问次数，从而进一步提升了系统整体效率。

# IV. 评估

## A. 精度评估

为评估所提出 FIGLUT 的数值精度，我们在 NVIDIA GPU 与 FIGLUT 引擎上分别进行推理精度测试。为了进一步优化引擎，我们将 FP-INT GEMM 中使用的预对齐（pre-alignment）技术 [16], [22] 应用于 FIGLUT，从而得到两种变体：

* **FIGLUT-F**：不使用预对齐技术；
* **FIGLUT-I**：使用预对齐技术。

我们在 WikiText-2 数据集 [27] 上，对基于 Transformer 的 OPT 模型族 [39] 进行语言建模任务评估，并采用困惑度（perplexity）作为指标。所有模型均使用 FP16 激活与 4 比特权重，权重通过简单的均匀量化（round-to-nearest, RTN）得到。需要注意的是，为保持累积结果的数值精度，我们在累加阶段采用 FP32 [6], [13]。表 IV 给出了各类 GEMM 引擎相对于 NVIDIA GPU 的困惑度结果对比。尽管 FP 运算顺序的变化可能导致数值结果存在细微差异，但依托于高精度的 FP32 累加，FIGLUT-F 相比 NVIDIA GPU 几乎没有精度损失。采用预对齐方法的 FIGLUT-I 也展现了与既有研究 [16], [22] 一致的可比数值精度。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/0478649c78b3cd0da24dad670450ed3dc2c2bb61a417002a1810122bdc944cc5.jpg)
Fig. 13. 针对 Q4 与 Q8 语言模型的各类硬件引擎 TOPS/mm$^2$，结果相对 FPE 归一化。

**TABLE IV 使用不同 GEMM 引擎时 OPT 模型族的困惑度结果**

<table>
<tr><td>OPT</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>30B</td></tr>
<tr><td>GPU</td><td>55.24</td><td>67.95</td><td>35.46</td><td>24.13</td><td>20.93</td><td>19.17</td></tr>
<tr><td>FIGLUT-F</td><td>55.24</td><td>67.95</td><td>35.46</td><td>24.13</td><td>20.93</td><td>19.17</td></tr>
<tr><td>FIGLUT-I</td><td>55.24</td><td>67.95</td><td>35.46</td><td>24.13</td><td>20.89</td><td>19.17</td></tr>
</table>

---

## B. 硬件评估

### a) 配置设置

我们评估了五种硬件引擎：FIGLUT-F、FIGLUT-I、FPE、iFPU [22] 与 FIGNA [16]。其中，FPE 作为基线硬件引擎，其 PE 首先将 INT 权重量化值反量化为与激活相同精度的 FP 值，然后执行 FP 乘法与 FP 累加。对于 iFPU 与 FIGNA，两者均在执行 MAC 运算前执行预对齐过程，将 FP 输入激活转换为具有统一指数的整数尾数。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/b495b01e2be7cf22740c4f88a8eef901bb55ce283afdebc5b014bd2c5809c31b.jpg)
Fig. 14. 不同输入格式下 MPU 的面积分解，结果相对 FPE 归一化。

为了与这些引擎进行对比，我们实现了两种 FIGLUT 硬件版本，对应不同的数据处理格式：

* **FIGLUT-F**：将 FP 输入直接送入 LUT 元素，并采用 FP RAC；
* **FIGLUT-I**：在预对齐后执行整数算术运算。

所有硬件引擎均被设计为处理相同比特精度和字长的输入，并以相同格式输出结果。值得注意的是，iFPU 与 FIGLUT 支持 BCQ 权重量化格式，而 FPE 与 FIGNA 使用 INT 权重量化格式。为便于表示各硬件所使用的不同权重格式，我们按权重量化比特宽度进行记号，例如：Q4 同时代表 4 比特 BCQ 与 4 比特 INT 权重。

为保证公平性，所有引擎的 MPU 与 VPU 均被设计为提供相同吞吐率。具体而言，在 Q4 权重输入的情形下，FPE 与 FIGNA 采用  $64 \times 64$  的 PE 阵列；iFPU 在处理 1 比特权重时采用  $64 \times 64 \times 4$  阵列；FIGLUT 则采用  $2 \times 16 \times 4$  阵列，结合  $\mu = 4$ 与  $k = 32$  的设定，其总计算单元数量与 iFPU 相同。对于比 Q4 更低精度的权重，我们使用相同的硬件配置；对于 Q8 权重，则在 FPE 与 FIGNA 的基础上扩展权重量化比特精度至 8 比特，以获得相应结果。为了充分利用 iFPU 与 FIGLUT 的比特串行（bit-serial）特性，这两类硬件在配置时均被调整为在 Q4 下具有相同吞吐率，从而影响其它权重精度下矩阵乘运算所需的周期数。

所有硬件引擎均使用 Synopsys Design Compiler，在  $28\mathrm{nm}$  CMOS 工艺下针对  $100\mathrm{MHz}$  目标频率进行综合。基础 FP 和整数模块（包括 FPE 中的整数到 FP 转换器）均基于 Synopsys DesignWare 组件实现。所有输入、权重与输出缓冲区均采用  $28\mathrm{nm}$  工艺的 SRAM 实现，片外 DRAM 的能耗由 CACTI 仿真器 [5] 估算。来自片外 DRAM 的数据在主机控制器的管理下直接传输至 FIGLUT 加速器的 SRAM 缓冲区，随后被 MPU 与 VPU 用于计算。除面积评估外，所有评估结果均将内存数据传输的功耗与延迟计入能耗统计中（面积评估未计入 DRAM 面积，因为其未被显式给出）。

### b) 面积评估

图 14 展示了不同引擎类型下，在多种输入精度配置下的 MPU 归一化面积分解。面积主要分为两部分：执行计算的算术逻辑（arithmetic logic）与用于暂存数值的触发器（flip-flops）。结果表明，相比其他整数运算引擎，FPE 与 FIGLUT-F 的算术逻辑部分面积更大，但 FIGLUT-F 由于执行的是 FP 加法而非 FP 乘法，其面积仍小于 FPE。在 Q8 场景中，FIGNA 的算术逻辑面积相较 FPE 增长更为明显，这是因为 FIGNA 的计算单元面积随权重量化比特精度扩展，而 FPE 仅在反量化模块上产生额外开销，FP 运算单元的输入位宽并未改变。

在对比 FIGLUT-I 与 FIGNA 时，可以发现尽管 FIGLUT-I 额外引入了 LUT 生成器，但由于通过 LUT-based 运算减少了实际算术单元数量，其整体算术逻辑面积与 FIGNA 相近。同时，引入 LUT-based 运算也降低了相较其他硬件架构的触发器总面积。相较于传统的  $64 \times 64$  或  $64 \times 64 \times 4$  MPU 配置，FIGLUT 使用更小规模的  $2 \times 16 \times 4$  MPU，从而有效减少了脉动阵列运算所需的流水级数。此外，在权重常驻数据流下，传统硬件可能需要多达 63 级的输入缓冲，而 FIGLUT 最多只需要 15 级输入缓冲，在效率与资源利用方面都有明显优势。

图 13 展示了针对从 OPT-125M 到 OPT-30B 不同规模 LLM 的面积效率结果。对于非比特串行架构的硬件设计，其整体趋势与图 14 中面积分解的变化呈反比，并在各个网络规模与输入比特宽度配置上保持一致。例如，如前所述，FIGNA 在 Q8 下的 PE 面积相较 Q4 较大，导致其归一化面积效率下降。而对于比特串行架构，由于 TOPS/mm² 还隐含反映了完成计算所需时间的因素，在相同硬件配置下，当权重量化比特宽度增加时，比特串行硬件需要约两倍的时钟周期，从而在 Q8 阶段表现出更明显的性能劣化。对于 FP32，由于尾数位宽增加，FIGNA 与 FIGLUT-I 之间的差距缩小，进而在 FP32-Q8 场景中出现性能优势反转。尽管如此，在当前主流的 sub-4-bit 权重仅量化场景下，所提出的引擎在面积效率上最高可达现有最优方案的  $1.5 \times$ 。需要注意的是，非 GEMM 运算会随模型规模带来轻微偏差，但受限于 LLM 中 GEMM 运算占据主导，这部分影响可以忽略。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/64212ea73367bca1a41aa3e207b658083c0b518ad487b8e2c39a01e81a2c611f.jpg)
Fig. 15. 不同比特精度下 FP-INT GEMM 硬件引擎的归一化能耗分解。

### c) 能耗评估

图 15 展示了不同输入数据精度组合下的能耗分解，相对 FPE 的能耗进行归一化。为评估并比较在不同比特精度下（包括 Q4、Q8 以及 sub-Q4 精度的 Q1、Q2、Q3）的能效，我们基于 OPT-6.7B 模型进行仿真。如前所述，所有系统均被配置为具有相同的峰值性能，以保证公平比较。

在 Q8 运算中，比特串行硬件无论比特精度如何变化，其底层架构保持不变，而 FPE 与 FIGNA 等固定精度架构则需要扩展以支持 8 比特权重。对于 sub-4-bit 权重精度，比特串行硬件的运算次数会随着比特宽度降低而成比例减少，而固定比特宽度硬件则需要以填充到 4 比特的形式处理 sub-4-bit 数据。这使得 iFPU 与 FIGLUT 等比特串行硬件能够更快完成计算，随着权重量化精度下降有效降低能耗。

在 Q4 精度下，计算单元（包括 MPU 与 VPU）能耗的变化趋势与图 14 中的面积分布类似。然而，由于 iFPU 采用了比 FPE 更多的触发器，尽管在面积上更具优势，但在功耗上却高于 FPE，从而呈现出不同的能耗分布特性。对于 FIGLUT-I，由于计算单元面积随输入激活的尾数位宽扩展，其从 BF16 过渡到 FP32 时，PE 面积会相应增大。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/968dc4e27a67074c01575c742d860a03666bf6103f2af1f2d1d491b832b7c556.jpg)
Fig. 16. 针对 sub-4-bit OPT 语言模型，各硬件引擎的 TOPS/W，相对 FPE 归一化。

图 16 展示了在不同规模 LLM 上执行 sub-4-bit 权重运算时各硬件引擎的能效（TOPS/W）。为突出比特串行硬件的表现差异，我们给出了 Q2、Q3 与 Q4 三种精度下的能效结果。由于 FIGLUT-I 在整数运算下的能效优于 FIGLUT-F，本节主要针对 FIGLUT-I 进行分析。如预期，降低权重量化比特宽度可以减少比特串行硬件完成同一任务所需的周期数，从而提升能效。FIGLUT 在所有权重量化比特宽度下均取得最高 TOPS/W，其中 FIGLUT-Q2 相较其他配置展现出尤为显著的性能收益。该结果表明，在内存带宽受限的计算场景（例如 LLM 推理）中，尽可能降低权重量化比特宽度并结合比特串行运算，能够显著提升计算速度，并获得远高于传统硬件的能效表现。

图 17 展示了在 OPT-6.7B LLM 上，FIGLUT 在不同混合精度配置下的 TOPS/W 与困惑度结果。我们将这些结果与基线 FIGNA 在 2、3、4 比特精度下采用 OPTQ 量化方法 [10] 的性能进行对比；而 FIGLUT（包括其混合精度结果）则采用 ShiftAddLLM [36] 进行评估。需要注意的是，最近的权重仅量化技术 [7], [10], [14], [19], [24], [25], [36], [38] 已经使得 sub-4-bit 模型的性能接近未量化 FP16 基线（图中虚线所示）。在相同 4 比特量化设置下，FIGLUT 的能效比 FIGNA 高  $1.2 \times$ ，并略微降低了困惑度。在 Q3 精度下，FIGLUT 的能效是 FIGNA 的  $1.6 \times$ ，同时取得更低的困惑度。此外，在相近能效水平下，FIGLUT-Q2.4 的计算效率相较 FIGNA-Q3 提升了  $1.98 \times$ ，同时实现了  $20%$  的模型尺寸压缩。在 2 比特权重量化下，FIGLUT 支持非均匀 BCQ 格式，相比 FIGNA 在困惑度表现上更加稳定，能效最高可提升  $2.4\times$ 。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-07/4629908f-0f93-48ef-a57f-d826a4d0ffbc/4ed68ae97e637ef82b28eac37ebf7709175e3870b849c1c5611c7c20060ba9b1.jpg)
Fig. 17. 基于 FIGLUT 的混合精度 OPT-6.7B 推理：TOPS/W 与困惑度结果。

**TABLE V 硬件加速器设计对比**

<table>
<tr><td>Hardware</td><td>Format (Act.-Weight)</td><td>Throughput (TOPS)</td><td>Power (W)</td><td>Energy Effi. (TOPS/W)</td></tr>
<tr><td>A100</td><td>FP16-FP16</td><td>40.27*</td><td>192</td><td>0.21*</td></tr>
<tr><td>A100</td><td>FP16-Q4**</td><td>1.85</td><td>208</td><td>0.01</td></tr>
<tr><td>H100</td><td>FP16-FP16</td><td>62.08*</td><td>279</td><td>0.22*</td></tr>
<tr><td>iFPU</td><td>FP16-Q4</td><td>0.14</td><td>0.67</td><td>0.21</td></tr>
<tr><td>FIGNA</td><td>FP16-Q4</td><td>0.14</td><td>0.41</td><td>0.33</td></tr>
<tr><td>FIGLUT</td><td>FP16-Q4</td><td>0.14</td><td>0.29</td><td>0.47</td></tr>
</table>

* 对 GPU 而言单位为 TFLOPS。
  ** 使用 LUT-GEMM kernel [28]。

表 V 展示了不同硬件加速器的能效比较。由于 GEMM 运算在 LLM 推理工作负载中占据主导 [28]，我们主要关注 GEMM 的性能表现。为公平起见，我们额外纳入了当前主流商用加速器 A100 与 H100 GPU。由于 GPU 的具体架构细节并未公开，我们基于实测结果进行评估：吞吐率由计算任务规模与实际测得的延迟共同确定，功耗则通过 nvidia-smi 工具获取 [2], [28], [30]。评估目标模型为 OPT-6.7B，批大小（batch size）统一设为 32，以贴近实际 LLM 使用场景。为保证横向对比公平性，我们在所有测试中统一采用 4 比特权重量化格式。需要指出的是，由于 GPU 不具备 FP-INT 运算单元，因此仍需依赖 FP-FP 单元并通过反量化进行计算。尽管 LUT-GEMM [28] 提出了高效的 FP-INT 内核，但其仅支持 batch size 为 1 的场景。为了统一比较基准，我们将所有 FP-Q4 加速器调至相同最大性能（TOPS）。

H100 在能效方面优于 A100，主要归功于其更加先进的制程工艺与更高的内存带宽，这对于内存带宽受限的 LLM 工作负载尤为重要。需要注意的是，GPU 实测 TFLOPS 远低于其理论峰值，主要是由于 batch size 较小。在实际 LLM 推理系统中，由于 KV cache 大小限制以及面向用户请求的批处理调度策略优化 [23], [37]，生成阶段的 batch size 很少超过 32。LUT-GEMM 的性能下降更为明显，因为它仅支持 batch size 为 1，并依赖 CUDA 核心而非 Tensor Core。iFPU 虽然专门针对 FP-INT 运算单元进行设计以提升效率，但受限于比特串行的固有特性，其性能仍不及 H100。FIGNA 在此基础上进一步缓解了这些问题，从而获得更高的能效；最终，FIGLUT 通过 LUT-based GEMM 从根本上规避比特串行运算中的大量冗余计算，使能效进一步提升。考虑到 FIGLUT 采用的是  $28\mathrm{nm}$  工艺，而 A100 与 H100 分别使用 7nm 与 4nm 的先进工艺，如果在相同制程下进行对比，FIGLUT 的能效优势会更加显著。

---

# V. 结论

**总结。**
本文提出了 FIGLUT，一种针对权重仅量化模型的 FP-INT 加速器，通过 LUT-based 运算显著提升比特串行加速器的效率。我们引入了基于触发器的查找表结构（FFLUT），并以读-累加单元（RAC）替代传统 MAC 单元以获得更优性能。同时，我们通过对 LUT 规模与 RAC 配置进行分析，提出了利用对称性将 LUT 规模减半的高能效设计。实验结果表明，在 sub-4-bit 权重精度下，FIGLUT 在 TOPS/W 指标上明显优于现有最先进 FP-INT 加速器，在显著提升能效的同时，仍能保持良好的系统集成灵活性。

**局限性。**
如图 15 所示，随着权重量化比特精度的增加，FIGLUT 的性能收益逐渐减弱，这是比特串行架构固有的局限。若要在更高比特精度下获得更高性能，需要进一步增大  $\mu$  值以充分利用 LUT-based GEMM 带来的计算优势，但这会显著增大 LUT 及其生成器的面积与功耗开销。尽管如此，表 VI 表明，当前最先进的权重仅量化技术已经能够在保持接近 FP16 基线精度的前提下，将权重量化精度降至 sub-4-bit。我们预计，随着权重仅量化方法的进一步发展，在更低比特精度下，FIGLUT 的优势将更加显著。

**TABLE VI 权重仅量化场景下的困惑度比较**

<table>
<tr><td>OPT</td><td>350M</td><td>1.3B</td><td>2.7B</td><td>6.7B</td><td>13B</td><td>30B</td></tr>
<tr><td>FP16</td><td>22.00</td><td>14.62</td><td>12.47</td><td>10.86</td><td>10.13</td><td>9.56</td></tr>
<tr><td>BCQ4 [36]</td><td>22.59</td><td>15.11</td><td>12.73</td><td>11.08</td><td>10.33</td><td>9.70</td></tr>
<tr><td>BCQ3 [36]</td><td>28.72</td><td>19.69</td><td>15.28</td><td>11.80</td><td>10.70</td><td>9.89</td></tr>
</table>
