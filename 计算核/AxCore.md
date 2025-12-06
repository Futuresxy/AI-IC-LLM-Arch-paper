# AxCore：面向 LLM 推理的量化感知近似 GEMM 单元

Jiaxiang Zou*
香港科技大学（广州）
中国广州
[jiaxiangzou@std.uestc.edu.cn](mailto:jiaxiangzou@std.uestc.edu.cn)

Yonghao Chen*
香港科技大学（广州）
中国广州
[ychen433@connect.hkust-gz.edu.cn](mailto:ychen433@connect.hkust-gz.edu.cn)

Xingyu Chen
香港科技大学（广州）
中国广州
[xchen740@connect.hkust-gz.edu.cn](mailto:xchen740@connect.hkust-gz.edu.cn)

Chenxi Xu
香港科技大学（广州）
中国广州
[cxu930@connect.hkust-gz.edu.cn](mailto:cxu930@connect.hkust-gz.edu.cn)

Xinyu Chen†
香港科技大学（广州）
中国广州
[xinyuchen@hkust-gz.edu.cn](mailto:xinyuchen@hkust-gz.edu.cn)

# 摘要

大语言模型（LLM）已经成为现代自然语言处理的基础，但其巨大的计算与内存需求给高效推理带来了主要障碍。基于 Transformer 的 LLM 严重依赖浮点通用矩阵乘（FP-GEMM），该操作同时主导计算与带宽开销。本文提出 AxCore：一种量化感知的近似 GEMM 单元，将**仅权重量化**与**浮点乘法近似（FPMA）**相结合，以实现高效且准确的 LLM 推理。与传统 GEMM 单元不同，AxCore **完全消除了乘法器**，在一种新型脉动阵列中用**低比特整数加法**替代乘法。AxCore 具有以下关键创新：（1）一种基于 FPMA 的混合精度处理单元（PE），支持对压缩权重与高精度激活进行直接计算；（2）一种轻量级精度保持策略，包括次正规数处理、误差补偿与格式感知量化；（3）一组脉动阵列优化，包括共享的校正与归一化逻辑。在开源 LLM 上的评估表明，AxCore 的计算密度相对传统 FP GEMM 单元最高可提升至  $6.3 \times 12.5 \times$ 。与最先进的 INT4 加速器 FIGLUT 和 FIGNA 相比，AxCore 的计算密度分别提升  $53%$  与  $70%$ ，同时困惑度更低。AxCore 已开源： [https://github.com/CLab-HKUST-GZ/micro58-axcore。](https://github.com/CLab-HKUST-GZ/micro58-axcore。)

# CCS 概念分类

* 计算机系统组织  $\rightarrow$  脉动阵列。

*两位作者对本研究贡献相同。 $\dagger$ 通讯作者。

允许在不收费的情况下，为个人或课堂使用制作本作品全部或部分的电子版或纸质版复制件，前提是复制件不得用于盈利或商业目的，并且复制件需在首页保留本声明与完整引用。非作者（多个）所拥有版权的本作品组件必须得到尊重。允许在注明出处的情况下进行摘要。其他复制、再出版、发布到服务器或分发到列表等行为，需要事先获得特定许可和/或支付费用。权限申请请联系 [permissions@acm.org](mailto:permissions@acm.org)。

MICRO '25，韩国首尔

© 2025 版权归所有者/作者持有。出版权许可给 ACM。ACM ISBN 979-8-4007-1573-0/25/10

[https://doi.org/10.1145/3725843.3756094](https://doi.org/10.1145/3725843.3756094)

# 关键词

大语言模型，近似计算，仅权重量化，硬件加速器

# ACM 参考格式：

Jiaxiang Zou, Yonghao Chen, Xingyu Chen, Chenxi Xu, and Xinyu Chen. 2025. AxCore: A Quantization-Aware Approximate GEMM Unit for LLM Inference. In 58th IEEE/ACM International Symposium on Microarchitecture (MICRO '25), October 18-22, 2025, Seoul, Republic of Korea. ACM, New York, NY, USA, 15 pages. [https://doi.org/10.1145/3725843.3756094](https://doi.org/10.1145/3725843.3756094)

# 1 引言

大语言模型（LLM）已经革新了自然语言处理任务，例如语言理解、翻译与生成 [10, 42, 48, 53]。这些模型由多层堆叠的 Transformer 层构成，参数规模从数十亿到数千亿不等，从而带来显著的内存与计算需求。例如，拥有 1750 亿参数的 GPT-3 在 FP16 表示下需要约 350GB 内存 [5]，远超 GPU 等标准硬件加速器的容量 [7]。LLM 的核心计算瓶颈源自 Transformer 架构，其中通用矩阵乘（GEMM）操作同时主导算术吞吐与内存带宽。这些 GEMM kernel 通常采用浮点运算实现（如 FP16 或 BF16），硬件代价高，阻碍了高效推理。

量化通过使用低精度数据类型表示高精度浮点值，已成为解决上述挑战的关键技术。尤其是**仅权重量化**：将模型权重压缩为低比特格式（如 INT4 或 FP4），同时保留较高精度的激活（如 FP16），已被广泛用于 LLM 推理 [12, 15, 26, 29, 30, 45]。该方法有效的原因在于权重占用的内存显著多于激活；而激活是动态且依赖输入的，难以在不牺牲准确性的情况下进行量化 [29, 30, 51]。然而，这类量化需要混合精度 GEMM（mpGEMM）单元：专用硬件直接处理高精度激活与量化权重，从而消除传统 GEMM 单元所需的显式反量化步骤，并提升吞吐与带宽效率 [22, 40, 43, 46, 49, 50, 54]。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/0f1fc5b642594064c85a03e97db7519ade94e27fa9e713778d8d0c278daaacec.jpg)
Figure 1: AxCore achieves significantly higher compute density and comparable or better perplexity compared to conventional FP GEMM cores (FPC) and state-of-the-art INT4-based accelerator FIGNA [22].

与此同时，使用整数加法的浮点乘法近似（FPMA）正日益受到高效模型推理的关注 [24, 33, 36]。Mitchell 的对数近似 [35] 表明浮点数可在对数数制下解释；Gustafsson 等人 [20] 从理论上证明浮点乘法可由整数加法替代。这一洞见揭示：GEMM 单元中昂贵的浮点乘法器可以被更简单的整数加法器替换，为 LLM 所需的大规模密集计算带来显著资源节省。尽管前景可观，现有 FPMA 方法通常局限于统一精度场景，并且在深层 LLM 上会出现精度损失，尤其在低比特量化条件下，次正规值频繁出现且误差累积并不简单。

本文提出 AxCore：一种面向 LLM 推理的量化感知近似 mpGEMM 单元。AxCore 将低比特量化与 FPMA 融合，实现高效、无乘法器的混合精度矩阵乘，同时保持端到端模型精度。其核心创新包括：

* **混合精度 FPMA 处理单元（PE）**：AxCore 扩展 FPMA，使其支持高精度激活与低比特量化权重之间的直接 mpGEMM，从而降低数据通路位宽与 PE 复杂度。该设计可并行支持多种浮点格式，使其能灵活适配不同量化配置的推理。
* **轻量级精度保持软硬件协同设计**：为缓解 FPMA 的近似误差，AxCore 引入轻量级软件-硬件协同方案，包括：（1）在线次正规数转换以保证低比特格式下的正确性；（2）在线常数型误差补偿；（3）自适应的格式感知离线量化。
* **优化的脉动阵列架构**：AxCore 采用高效的脉动阵列，通过在 PE 间共享误差校正逻辑与结果归一化逻辑来降低硬件资源消耗。

大量评估表明 AxCore 在硬件效率与精度方面均表现优异。如图 1 所示，在 W4A16 设置下，AxCore 的计算密度最高可比传统 FP GEMM 核高  $6.7 \times$ ，并比基于 INT4 的设计 FIGNA [22] 高  $1.7 \times$ 。尽管采用近似设计，AxCore 仍保持与现有方案竞争甚至更优的困惑度。

例如在 OPT-30B 模型上，AxCore 的困惑度为 9.78，优于 FPC（9.82）与 FIGNA（9.95）。AxCore 通过弥合近似计算与 mpGEMM 运算之间的鸿沟，实现了高效的 LLM 推理。

# 2 背景

# 2.1 LLM 推理中的 GEMM

大语言模型（LLM）通常由多个堆叠的 Transformer 解码器块构成 [5, 42, 48, 53]，每个块包含掩码自注意力与线性变换层。注意力机制提供了 Transformer 的推理能力，而线性层（包括前馈网络与注意力投影）在 LLM 推理中占据主要计算负载，并贡献了模型参数的大部分 [5, 10, 42, 48, 53]。这些线性层高度依赖通用矩阵乘（GEMM），因为其核心计算是密集线性变换：通过大型权重矩阵将高维输入激活映射到输出激活。

图 2 展示了在不同序列长度下，OPT-175B 与 LLaMA-3.1-405B 中注意力机制与线性层的相对操作数占比。可以看到，尽管在 LLM 推理中注意力计算占比会随序列长度增加而增长，但在线性层中 GEMM 操作在实际序列长度（1 万–2 万 token）下仍然主导计算负载（69%–99%）[41]。值得注意的是，在预取（prefetch）阶段，注意力中也主要使用 GEMM [22]，这意味着 GEMM 的真实计算占比可能更高。该趋势表明：在大规模场景下优化线性层 GEMM 仍是提升整体 LLM 推理效率的关键。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/de775ae84b3406af99c97ecd7328869a4f0a31b9c82ef7a678d92d48f64ecd1d.jpg)
Figure 2: Relative proportion of operations (OPs) in attention mechanism and linear layers of OPT-175B and LLaMA-3.1-405B across various sequence lengths with a batch size of 32.

# 2.2 仅权重量化

为了降低内存与计算开销，量化技术被广泛用于 LLM 推理。量化将高精度权重映射为紧凑的低比特格式（如 INT4 或 FP4），显著缩小模型规模并降低算术位宽。尤其是**仅权重量化**在 LLM 推理中非常实用：模型权重通常远比激活占用更多内存，因此对权重量化能有效降低内存占用与带宽需求。相对而言，激活是动态且与输入相关的，对其进行激进量化往往会造成显著精度损失 [27]。因此，仅权重量化（低比特权重 + 高精度激活，如 FP16 或 BF16）已成为学术界与工业界的标准做法，并广泛应用于现代 AI 硬件加速器（如 GPU 与 TPU）[15, 18, 28, 29, 39]。

量化通常通过缩放因子  $s$  将权重  $w$  映射为低比特表示  $w_{q}$ ：

$$
s = \frac {w _ {\operatorname* {m a x}}}{F _ {\operatorname* {m a x}}}, \quad w _ {q} = \operatorname {c l a m p} \left(\operatorname {r o u n d} \left(\frac {w}{s}\right), - F _ {\operatorname * {m a x}}, F _ {\operatorname * {m a x}}\right), \tag {1}
$$

其中  $F_{\mathrm{max}}$  是目标格式可表示的最大值（例如 INT4 中为 7）。round 表示整数量化中的舍入或浮点量化中的映射 [32, 55]。clamp 将量化权重限制在可表示范围  $[F_{\mathrm{min}}, F_{\mathrm{max}}]$  内。

为在激进低比特量化下保持精度，常采用**分组量化**[15, 29, 55]。该策略将权重张量划分为更小的组（例如 32 或 128 个元素），并为每组分配独立的缩放因子（通常为 FP16）。这种细粒度方法能更好拟合每组内部的局部分布，从而减少量化误差。

# 2.3 量化感知 GEMM

在 LLM 推理中，GEMM 操作涉及大矩阵的权重与激活相乘。在不量化时，GEMM 使用全精度操作数执行（例如 FP16 × FP16）[5, 29, 42, 48, 53]，如图 3a 所示。在仅权重量化下，有两种常见 GEMM 执行策略 [15, 29, 49, 50]：**间接 GEMM**（图 3b），先将量化权重反量化（即乘以缩放因子）恢复浮点值后再执行 GEMM；以及**直接混合精度 GEMM（mpGEMM）**（图 3c），直接在低比特权重与 FP 激活之间执行 GEMM，并仅在累加输出后再做反量化。直接 mpGEMM 更具硬件效率，因为它使 GEMM 单元的数据通路更轻量，并避免了逐权重反量化的开销 [22, 40, 43, 46, 49, 50, 54]。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/fb338cbd844480a9cb27d71b02ed1efa77747e79937beaef8af914444f5cf33a.jpg)
(a) Standard GEMM (b) Indirect GEMM (c) Direct mpGEMM
Figure 3: Comparison of GEMM computation modes, with 4-bit quantized weights and 16-bit activations.

为降低 mpGEMM 单元的硬件复杂度，许多硬件加速器因其简单性而采用均匀量化格式（如 INT4 或 INT8）[22, 40, 54]。例如 FIGNA [22] 采用 INT4 量化并设计了 INT4-FP16 mpGEMM 单元，面积节省最高可达 4 倍。然而，均匀格式对数值进行均匀分布的编码方式，与 LLM 权重近似高斯分布的特性并不契合 [19]。相对地，非均匀格式（如 FP4、FP8）在零附近分配更多表示能力，具有更高精度潜力。本文表明，基于 FP 的量化不仅精度更好，也能通过我们提出的 AxCore 架构实现更高效的硬件实现。

# 2.4 用整数加法实现浮点乘法近似（FPMA）

浮点（FP）乘法在许多应用中是基础运算，但由于复杂性，其硬件面积成本较高。根据 IEEE 754 标准 [1]，一个规范化浮点数  $x$  表示为：

$$
x = (- 1) ^ {S _ {x}} \cdot 2 ^ {E _ {x} - B} \cdot (1 + M _ {x}), \quad 0 \leq M _ {x} <   1, \tag {2}
$$

其中  $S_{x}$  为符号位， $E_{x}$  为使用  $N_{E}$  位表示的指数（即  $N_{E}$  表示指数域位宽）， $B = 2^{N_{E} - 1} - 1$  为偏置（bias）。 $M_{x}$  为尾数，使用  $N_{M}$  位编码，并隐含一个前导 1。FP 乘法需要分别处理符号、指数与尾数，然后再归一化与舍入，因此硬件成本显著。

为解决这一低效问题，浮点乘法近似（FPMA）[6, 20, 31] 用更简单的整数加法替代昂贵乘法。基于 Mitchell 的对数近似 [35]，FPMA 将浮点数  $x$  在对数域中近似为：

$$
\log_ {2} (| x |) = E _ {x} - B + \log_ {2} \left(1 + M _ {x}\right) \approx E _ {x} - B + M _ {x} \tag {3}
$$

通过对对数项线性化，可将浮点乘法  $r = x \cdot y$  近似为：

$$
\log_ {2} (| r |) = \log_ {2} (| x \cdot y |) \approx \left(E _ {x} + M _ {x}\right) + \left(E _ {y} + M _ {y}\right) - 2 B \tag {4}
$$

由于乘积  $r$  也可表示为  $\log_2(|r|) \approx E_r + M_r - B$ ，因此近似乘法可通过下式实现：

$$
R = X + Y - B \tag {5}
$$

其中  $X = E_{x} + M_{x}, Y = E_{y} + M_{y}$ ， $R = E_{r} + M_{r}$ 为结果的二进制近似表示。所有运算均为整数加法，从而消除复杂乘法器。并且结果  $R$  已是按指数与尾数字段顺序排列的标准浮点值，无需额外重转换。

尽管 FPMA 具有显著的硬件效率，它也会引入近似误差，来源于线性化近似  $\log_2(1 + M) \approx M$ 。这会在精度敏感的模型（如 LLM）中导致精度下降。此外，FPMA 假设输入为规范化浮点数，不能直接用于次正规数（subnormal）。次正规数表示非常小的数值，其尾数没有隐含前导 1。

# 3 将 FPMA 应用于量化 LLM 推理 3.1 挑战

尽管 FPMA 通过用轻量的整数加法器替代 FP 乘法器而具有显著提升硬件效率的潜力，但在量化 LLM 推理中采用 FPMA 仍然面临挑战。

挑战 1：对基于 FPMA 的 mpGEMM 的硬件支持。现代 LLM 常用仅权重量化，从而产生 mpGEMM 运算，例如 FP16 激活乘以 FP4 或 INT4 权重。FPMA 虽可用于间接 GEMM（图 3b），在反量化后对全精度乘法进行近似，但这会抵消量化带来的效率收益。为了充分利用低比特表示，FPMA 必须支持直接 mpGEMM（图 3c），即权重在计算过程中保持压缩格式。

然而，传统 FPMA 方法仅支持统一精度（例如  $\mathrm{FP16} \times \mathrm{FP16}$ ），将其扩展到混合精度 FPMA（mpFPMA）需要对处理单元（PE）与数据通路进行根本性重设计。难点在于对齐不同格式的操作数、处理偏置不匹配，并在基于整数的近似中维持足够精度。此外，设计能够处理这种混合精度整数近似计算、同时保持最小数据通路开销与最大复用率的高效 mpGEMM 单元并不简单。若缺乏精心的体系结构创新，mpFPMA 会遭遇过宽的数据通路、低效的格式对齐，以及每个 PE 中冗余的校正逻辑，从而限制可扩展性与硬件效率。

# 挑战 2：在最小成本下保持精度。

FPMA 依赖近似  $\log_2(1 + M) \approx M$ ，这会引入系统性误差，并在深层模型中跨层累积。图 4 展示了在不同 OPT 模型规模上应用 FPMA 的困惑度退化。尽管 FP4 相较全精度 FP16 仅引入适中精度损失，但引入 FPMA 会导致困惑度明显升高。注意困惑度差异  $>1%$ 通常被认为是显著的。这表明：若没有适当的误差补偿，FPMA 会带来不可接受的性能退化。尽管既有工作 [31, 33] 引入了误差补偿技术（如位串行校正或附加偏置项）以缓解 FPMA 数值误差，但这些方法通常针对同精度 FPMA（如 FP16  $\times$  FP16），无法推广到混合精度 mpFPMA（例如 FP16  $\times$  FP4）。

此外，将权重量化为低比特 FP 格式（如 FP4）会产生更高比例的次正规数，因为指数位更少。这是由于低比特格式中的次正规数能覆盖相对较大的数值范围；但其尾数没有隐含前导 1，会导致 FPMA 在数学上不再正确，从而产生显著误差。如图 4 所示，若 mpFPMA 不正确处理次正规数（naive mpFPMA），困惑度会进一步恶化——而这在已有工作中往往被忽视。

# 3.2 我们的解决方案 - AxCore

为应对上述挑战，我们提出 AxCore：一种量化感知的近似 mpGEMM 单元，将 FPMA 与低比特量化紧密融合，以实现高效且准确的 LLM 推理。

# 特性 1：通过基于 mpFPMA 的脉动阵列单元实现高效 mpGEMM 执行。

为解决 FPMA 支持混合精度 GEMM 的挑战，AxCore 引入由优化的 mpFPMA PEs 组成的脉动阵列，可直接对压缩的低比特权重与高精度激活进行计算。为降低数据通路位宽与 PE 复杂度，AxCore 采用**校正前移（correction advancing）**：在 PE 外部预计算校正项并在行内共享。归一化被推迟到共享单元以减少每个 PE 的逻辑，

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/a4bc9d1c074a394a0affafc7f95166fe316554b8c16c8e69721f66a1554c4f95.jpg)
Figure 4: Perplexity comparison across different OPT model sizes using various computation methods. Activations are in FP16. FPMA and mpFPMA without subnormal handling result in significant accuracy loss.

同时，基于 FPMA 的反量化消除了后 GEMM 乘法器的需求。

# 特性 2：轻量级精度保持机制。

为应对 FPMA 固有的误差，AxCore 采用面向混合精度场景专门设计的轻量级补偿机制，包括预计算的偏置与校正项，能够在不同操作数组合下稳定输出。关键的是，每个 PE 集成了次正规数转换逻辑，可检测并将次正规值转换为最近的规范化表示，从而避免因尾数形式不正确导致的精度退化。此外，AxCore 采用自适应的格式感知量化：为每个权重分组动态选择最合适的 FP4 编码（例如 E1M2、E2M1、E3M0）。这种细粒度的适配进一步增强了在不同层、不同分布下的量化保真度。以上特性共同使 AxCore 在保持 LLM 推理所需精度的同时实现高硬件效率。

# 4 面向 LLM 的精度保持 mpFPMA

# 4.1 将 FPMA 扩展到 mpFPMA

为在量化 LLM 推理中实现高效的混合精度矩阵乘（mpGEMM），我们扩展浮点乘法近似（FPMA），使其支持不同精度的操作数。尽管近似公式在结构上与传统 FPMA 相似，但位宽、定点对齐与偏置校正必须被仔细重新设计。为便于说明，记  $r = a \times w_{q}$  为 FP16 激活  $a$  与 FP4 低比特量化权重  $w_{q}$  的乘法。

在 mpFPMA 中，首先将操作数对齐到公共定点表示以确保加法正确。由于 FP4 的尾数位少于 FP16，我们对 FP4 操作数的尾数进行左移（即补零）以匹配 FP16 的分辨率。对齐后的值表示为：

$$
\operatorname {A l i g n} \left(w _ {q}\right) = w _ {q} \ll \left(\text {M a n t i s s a} _ {\mathrm {F P} 1 6} - \text {M a n t i s s a} _ {\mathrm {F P} 4}\right) \tag {6}
$$

这可确保两操作数的小数点位置一致。然而，由于指数偏置不同（例如 FP16 为 15，而 FP4 E2M1 为 1），需要一个格式感知的偏置校正项  $B_{1}$ ：

$$
B _ {1} = B _ {\mathrm {a}} + B _ {w _ {q}} - B _ {\mathrm {r}} \tag {7}
$$

其中  $B_{\mathrm{a}}$ 、 $B_{w_q}$ 、 $B_{\mathrm{r}}$ 分别为激活、量化权重与结果的指数偏置。对于激活与结果均为 FP16 的典型配置，该式可简化为  $B_{1} = B_{w_q}$ 。将对齐与偏置校正结合后，混合精度乘积的近似结果  $R$  为：

$$
R = A + \operatorname {A l i g n} \left(W _ {q}\right) - B _ {1} \tag {8}
$$

举例而言，将编码为 “0_01_1”（表示 1.5）的 FP4（E2M1）权重与一个值为 2 的 FP16 激活相乘，对齐后的 FP4 变为 “0_00001_1000000000”，而偏置校正值  $B_{1}$  对应 1。二者相加再减去偏置得到 3，从而准确近似  $1.5 \times 2$ 。

为提升 mpFPMA 的数值保真度，尤其是在量化噪声与近似误差存在时，我们引入常数补偿项  $C_1$ （细节见第 4.3 节）。最终 mpFPMA 表达式为：

$$
R = A + \operatorname {A l i g n} \left(W _ {q}\right) - B _ {1} + C _ {1} \tag {9}
$$

其中  $R, A$ 与  $W_{q}$  分别为结果、激活与权重的二进制近似表示。该形式使 AxCore 仅用整数加法即可高效且准确地近似混合精度乘法。

# 4.2 在 mpFPMA 中处理次正规数

随着量化 LLM 推理中浮点位宽进一步缩小，特别是 FP4 等格式，次正规值的处理变得愈发关键。

4.2.1 次正规数的问题。在浮点格式中，次正规值用于表示非常接近零、比最小规范化指数所能编码的数更小的值。这些值有助于保持渐进下溢，并在零附近提供更细的分辨率。次正规数的指数为 0 且尾数没有隐含前导 1：

$$
x _ {\text {s u b}} = (- 1) ^ {S} \cdot 2 ^ {1 - B} \cdot M \tag {10}
$$

其中符号位为  $S$ ，指数偏置为  $B$ ，尾数为  $M$ 。与公式 2 的规范化浮点数相比，它去掉了尾数中的 “1+”，并将指数范围下移。因此，由于 FPMA 依赖近似  $\log_2(1 + M) \approx M$ ，对于次正规数该近似在数学上不再成立，进而导致显著不准确。

与 FP16 或 FP32 等高精度格式相比，低比特浮点格式（如 FP4、FP8）更容易遇到高比例的次正规值，这是因为指数位数显著减少。例如 FP4 通常只有 2 个指数位，使可表示的指数值仅有 4 个。结果是规范化数的范围极窄，许多小幅值落在该范围之外并被编码为次正规值。在高精度格式中，次正规数通常是极小幅值的罕见边界情况（例如 FP32 中常低于  $10^{-38}$ ），但在 FP4 等低比特格式里，次正规数可表示相对较大的数（最高甚至可到 0.5），因此更为常见，尤其是在小值频繁出现的量化权重中。因此，次正规数在低比特格式中不再是边缘情况，必须在 FPMA 中被谨慎处理。

Table 1: Subnormal Number Conversion Table

<table><tr><td colspan="5">M1</td></tr><tr><td>subnormal</td><td>value</td><td></td><td>normal or 0</td><td>value</td></tr><tr><td>(0).0</td><td>0</td><td>→</td><td>return 0</td><td>0</td></tr><tr><td>(0).1</td><td>0.5</td><td>→</td><td>(1).0</td><td>0.5</td></tr><tr><td colspan="5">M2</td></tr><tr><td>subnormal</td><td>value</td><td></td><td>normal or 0</td><td>value</td></tr><tr><td>(0).00</td><td>0</td><td>→</td><td>return 0</td><td>0</td></tr><tr><td>(0).01</td><td>0.25</td><td>→</td><td>(1).00 / return 0</td><td>0.5↑ / 0↓</td></tr><tr><td>(0).10</td><td>0.5</td><td>→</td><td>(1).00</td><td>0.5</td></tr><tr><td>(0).11</td><td>0.75</td><td>→</td><td>(1).10</td><td>0.75</td></tr><tr><td colspan="5">M3</td></tr><tr><td>subnormal</td><td>value</td><td></td><td>normal or 0</td><td>value</td></tr><tr><td>(0).000</td><td>0</td><td>→</td><td>return 0</td><td>0</td></tr><tr><td>(0).001</td><td>0.125</td><td>→</td><td>return 0</td><td>0</td></tr><tr><td>(0).010</td><td>0.25</td><td>→</td><td>(1).000 /return 0</td><td>0.5↑ / 0↓</td></tr><tr><td>(0).011</td><td>0.375</td><td>→</td><td>(1).000</td><td>0.5</td></tr><tr><td>(0).100</td><td>0.5</td><td>→</td><td>(1).000</td><td>0.5</td></tr><tr><td>(0).101</td><td>0.625</td><td>→</td><td>(1).010</td><td>0.625</td></tr><tr><td>(0).110</td><td>0.75</td><td>→</td><td>(1).100</td><td>0.75</td></tr><tr><td>(0).111</td><td>0.875</td><td>→</td><td>(1).110</td><td>0.875</td></tr></table>

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/2e77160b078673a00aa47098b95f1456165518d39793d600c3f9c5060dfbfff3.jpg)
Figure 5: The value represented in subnormal encoding "011" and its equivalent normal encoding "010" in E1M2.

4.2.2 次正规数转换（SNC）。尽管现有工作大多忽略 FPMA 对次正规值的支持，我们通过提出一种轻量级的次正规数转换（SNC）方法来解决该问题。我们观察到：次正规与规范化编码可表示数值上非常接近的值。例如如图 5 所示，在 FP4（E1M2）中，尾数 “11” 的次正规值表示  $(-1)^{S} \cdot 2^{(1-B)} \cdot (0 + \frac{1}{2} + \frac{1}{4}) = (-1)^{S} \cdot 2^{(1-B)} \cdot 0.75$ （由公式 10）。若将其映射到规范化编码 “10”，根据公式 2 得到  $(-1)^{S} \cdot 2^{(0-B)} \cdot (1 + \frac{1}{2}) = (-1)^{S} \cdot 2^{(1-B)} \cdot 0.75$ ，与原次正规形式的数值等价。因此，如果把次正规输入转换到数值最近的规范化值，就能保持 FPMA 在次正规输入下的数学一致性。

次正规与规范化的转换关系如表 1 所示。由于次正规转换主要影响尾数，表 1 按尾数位宽组织映射规则，涵盖 M1、M2、M3 三种典型情形。每种情形中，第一列列出次正规编码（隐含前导 0），第三列给出数值上最近的规范化编码（隐含前导 1），第二与第四列分别给出对应十进制值。

运行时，AxCore 识别次正规编码，并根据该预定义映射将其替换为最接近的有效规范化值。对于无法精确映射到规范化表示的次正规值，选择最近的规范化值，表中用下划线标记。若简单地固定方向舍入（例如总向上或总向下），会引入系统性偏差并在矩阵乘中累积。为缓解该问题，AxCore 对这类次正规值采用在向上  $(\uparrow)$ 与向下  $(\downarrow)$ 之间的随机选择策略，使舍入决策在大规模运算中均衡交替，从而平衡累积误差。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/2a5a475fd6424e97eee7fa12c57ea14b336aba5936e80fd4b42df0534f58c830.jpg)
(a) Before compensation
Figure 6: Square error distribution of mpFPMA. The x-axis represents activation (FP16) mantissa  $M_{a}$ , and the y-axis represents weight mantissa  $M_{w}$  including E3M0, E1M2 and E2M1.

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/78196d140c337acb723df6df93b8af4ece951ad955f37b9694ac51eab83dbb10.jpg)
(b) After compensation

# 4.3 mpFPMA 的误差补偿

FPMA 中的近似（即  $\log_2(1 + M) \approx M$ ）会引入可预测的数值误差，该误差主要依赖输入操作数的尾数取值。

4.3.1 误差分布分析。为理解并解决该问题，我们通过比较近似结果与精确乘法结果来分析误差。图 6a 展示了不同位宽 mpFPMA 配置下的平方误差分布。结果表明误差分布高度不均匀，并随尾数空间变化，使得难以建立误差与尾数之间的简单算术关系来直接补偿 [20]。

目前尚无专为 mpFPMA 设计的有效误差缓解技术，直接扩展已有 FPMA 补偿策略通常代价很高。以往工作 [31] 为 FP8（如 E4M3）采用细粒度补偿：为每对  $M_{a} - M_{w}$  组合分配独立补偿值以降低误差。然而当激活精度从 E4M3 提升到 E5M10（如 FP16）时，所需片上存储量会使该方法不现实。

4.3.2 基于均值的常数补偿。为克服上述限制，我们提出一种基于常数的均值补偿策略，引入一个预计算的单一校正值  $C_1$  来降低 mpFPMA 的累积近似误差。该方法利用如下观察：在所有可能尾数组合上，近似误差的平均值可为 LLM 提供足够的误差校正。图 6b 展示了应用所提补偿后的误差分布。为量化 mpFPMA 引入的近似误差，我们定义逐元素误差为  $\varepsilon(m_a, m_w)$ ，表示在每个尾数组合下精确乘积与近似乘积的差异。为得到单一校正值，我们对所有有效尾数组合的期望误差取平均，得到与格式相关的补偿常数  $C_1$ ：

$$
C _ {1} = \frac {1}{2 ^ {N _ {M _ {w}}} \cdot 2 ^ {N _ {M _ {a}}}} \sum_ {m _ {a}, m _ {w}} \varepsilon \left(m _ {a}, m _ {w}\right) \tag {11}
$$

其中  $N_{M_w}$  为权重尾数位宽， $m_a, m_w$ 分别表示激活与权重可表示的尾数取值集合。

因此，只要给定输入数据格式的尾数位数，就可以通过一次性预计算得到误差补偿值，并以几乎可忽略的开销对所有模型或所有层通用。

该方法具有多项实用优势：补偿值可预先计算，没有运行时开销；额外逻辑极少；并可自然推广到多种 FP 格式组合，例如 FP16 × FP4、BF16 × FP4、FP16 × FP8 等。我们的实验表明：每种格式组合仅使用一个常数即可显著恢复精度，使该方法在量化 LLM 推理中既高效又有效。

# 4.4 自适应格式感知量化

为在激进量化下保持精度，我们提出一种格式感知方法，在统一框架内支持多种低比特 FP 格式。该观察源于：随着量化粒度变细，LLM 各层权重的数值分布更加多样 [21]。仅使用单一低比特格式（如 E2M1）在其动态范围或分辨率与局部数据分布不匹配时往往会导致次优量化。我们提出的格式感知量化在两方面具有新颖性：（a）按块（block）工作，可更细粒度地适配局部分布；（b）与 AxCore 的在线混合格式处理共同设计，对每个块分配最优 FP4 格式（例如对稀疏数据用 E3M0、对更均匀数据用 E1M2）。与 NVIDIA 的 FP4 [38] 与 LLM-FP4 框架 [32] 一致，这些格式将所有比特模式都用于编码有效的有限数值。

4.4.1 按块格式选择。不同于在全模型或张量级强制采用固定格式 [19, 47]，我们采用按块自适应策略，为每个权重块选择最优 FP4 格式。在 4-bit 量化中，我们考虑三种代表性的 FP4 格式：E3M0（类 2 的幂编码）、E2M1（标准）、E1M2（更均匀）。三者在动态范围与粒度之间权衡不同，适用性取决于局部权重分布。该格式选择通过离线流程完成：先将权重矩阵划分为大小为  $g \times n$  的块，其中  $g$  表示沿输入通道的权重分组大小， $n$  表示每块的输出通道数；并要求  $n$  和  $g$  均为 GEMM 阵列尺寸的整数倍。每个块包含  $n$  个权重分组。对于每个块，我们在校准数据集 [16] 提供的实际输入激活分布下，评估所有候选格式：对  $n$  个权重分组进行量化并选择使均方误差最小的格式。该目标形式化如下：

$$
D t y p e = \operatorname {a r g m i n} _ {d \in \mathcal {D}} | A \cdot W ^ {d} - A \cdot W | _ {2} ^ {2} \tag {12}
$$

其中  $\mathcal{D}$  为候选 FP4 数据类型集合（E3M0、E2M1、E1M2）， $W^{d}$  表示使用数据类型  $d$  量化并再反量化后的权重张量， $W$  为未量化权重张量， $A$  为激活。该格式选择流程的开销与传统静态量化相当 [28]。

图 7 可视化了 Llama2-7B 中第 0 层与第 29 层注意力输出张量权重分布的差异。第 0 层权重分布呈尖峰，更适合类 2 的幂编码，因此我们的方法选择 FP4 E3M0；而第 29 层分布更宽且更均匀，此时 FP4 E1M2 与 E2M1 更合适。

4.4.2 与 FPMA 的集成。我们也将这种格式感知量化扩展为与 FPMA 无缝集成。不同于传统量化：

$$
w _ {q} = \operatorname {c l a m p} \left(\operatorname {r o u n d} \left(\frac {w}{s}\right)\right) \tag {13}
$$

我们用 FPMA 风格近似重新定义量化与反量化：

$$
w _ {q} = \operatorname {c l a m p} (\operatorname {r o u n d} (w - S + B - C)) \tag {14}
$$

$$
w _ {r} = w _ {q} + S - B + C _ {2} \tag {15}
$$

其中  $S$ 、 $B$ 、 $C$ 、 $C_2$ 为预计算常数，分别表示 FP 数的二进制缩放、偏置与格式相关补偿项。 $C$ 为量化阶段补偿， $C_2$ 为反量化阶段补偿。将式 14 与式 15 结合得到：

$$
w _ {r} \approx w \tag {16}
$$

表明 FPMA 的补偿项  $C$ 与  $C_2$  可相互抵消，从而维持原始数值并保证正确性。

传统浮点量化与重构（即  $w \rightarrow w_{q} \rightarrow w_{r}$ ）因除法与乘法的不精确会引入数值漂移；而基于 FPMA 的量化仅依赖加减法，舍入偏差更小，数值一致性更好。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/5550a99428a48d9afc87a9b34e4b913dcfaaf52ecbfe9df126ee739672e4fa82.jpg)
Figure 8: AxCore systolic spatial array architecture.

# 5 AxCore 架构

# 5.1 概览

在 AxCore 的权重驻留（weight-stationary）数据流中，量化后的低比特权重（如 FP4）被预加载并在每列 PE 内保持不动，而高精度激活（如 FP16）则沿每一行水平传播。一个集中式 PreAdd 单元通过对激活施加校正项来预计算中间值  $T$ ：  $T = A - B_{1} + C_{1}$ ，其中  $A$ 为高精度激活， $B_{1}$ 为指数偏置校正项， $C_{1}$ 为格式相关补偿常数。随后该值沿行传播，以减少 PE 内逻辑的重复。

在每个 PE 内部，AxCore 引入了精心流水化的微架构。输入的低比特权重先经过专用的次正规数转换（SNC）单元，该模块识别次正规值并将其转换为最近的规范化表示。SNC 输出被统一到共享的内部格式（如 S1E3M2），使后续逻辑对格式无关。因此阵列可同时支持多种 FP 格式（如 FP4 的 E3M0、E2M1、E1M2），这对于实现自适应格式感知量化至关重要。对齐后的权重与预先计算的  $T$  在一个轻量整型加法器中相加，用一个简单的 2 输入加法替代传统乘法器。

后处理包含三个阶段：归一化（Normalization），将结果调整为标准浮点格式；AxScale，用基于 FPMA 的加法逻辑取代反量化乘法器以进行高效缩放；以及累加器（Accumulator），将缩放后的部分和与之前存储的值相加。

# 5.2 mpFPMA 处理单元（PE）

5.2.1 概览。如图 9 所示，每个 PE 在逻辑上由两个顺序模块组成：近似乘法（Approx Mult）模块与累加模块。PE 接收两类主要输入：低比特量化权重  $W_{q}$  与来自 PreAdd 单元的预计算中间值  $T$ 。  $T$  由 PreAdd 单元在阵列外部计算并沿行广播到该行所有 PE。进入 PE 后，量化权重  $W_{q}$  首先由 SNC 单元处理，识别次正规值并映射到最近的规范化表示。经 SNC 处理后的权重再进行尾数对齐：由于权重精度通常低于激活，权重尾数将补零以匹配激活的定点域。对齐后的权重随后通过低比特整型加法器与  $T$  相加，完成近似乘法模块的功能并得到乘积  $R = T + \mathrm{Align}(W_{q})$ 。乘积  $R$  进入累加模块，其中 Guard 单元检查激活或权重是否为零；若任一输入为零，Guard 将输出  $R$  强制置零。得到的  $R$  随后送入部分浮点加法器，与垂直方向传播的部分和  $P_{\mathrm{sum}}$  累加。该加法器位宽与激活输入相同，在 PE 内就地完成累加，并将完全归一化与舍入推迟到后处理阶段。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/0e0402ef77e4b8e2bad395fe23286cda01849ac5bfc9e32bebb7becb8fd797f4.jpg)
Figure 9: Architecture of Processing Element in AxCore.

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/d09fcab19047ae11321603e0e9375e72c1b28dba4526c8a296586cf3bf41a2d3.jpg)
(a) SNC
Figure 10: Subnormal Number Conversion (SNC) unit.

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/e1656049170de96a61274b8cf7a0e8b287fccad5cfbe648cb2d8c04fe40b7dd0.jpg)

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/57b6d06ac5d04835be186470b0b98af95398c32c12df1df2066e3cebefb5bb04.jpg)
(b) Input format
(c) Output format

5.2.2 次正规数转换（SNC）模块。SNC 单元结构如图 10a 所示，以 FP4 作为示例权重格式。SNC 接收输入量化权重  $W_{q}$ ，该权重可能采用多种 FP4 子类型之一编码：E1M2、E2M1 或 E3M0，如图 10b 所示。FormatSel 信号选择对应的格式专用解码器（例如 M2 Cvt、M1 Cvt），并将输入路由到相应路径。在每个解码器内部，一个小型逻辑表检测次正规编码（例如 S-0-00）并将其映射到附近的规范化值。由于输入  $W_{q}$  可能同时包含规范化与次正规值，SNC 也提供规范化值的旁路路径。

Zero Flag 单元检测全零输入并向下游 Guard 单元发出信号。除基本的零检测外，它还用于对无法精确映射的次正规值实现随机舍入。由于向下舍入总会得到 0，Zero Flag 被用来控制舍入方向：当某次正规值需要随机舍入时，Zero Flag 由随机比特设置，该比特从激活尾数的最高有效位取样；否则，它由常规零检测确定。该机制使舍入方向交替，从而减少反复近似带来的偏差。

所有转换后的输出都会被转成统一的内部格式：S1E3M2，如图 10b 所示。选择 S1E3M2 的原因是：其可同时支持所有 FP4 子类型，从而在 AxCore 中实现自适应格式感知量化。这样做允许每个权重分组在量化时根据分布特性灵活选择最合适的 FP4 编码（E1M2、E2M1 或 E3M0），同时保持推理阶段硬件逻辑的简洁。

# 5.3 脉动阵列优化

5.3.1 校正前移。在混合精度 FPMA（mpF-PMA）中，近似乘法结果计算为  $R = A + W_{q} - B_{1} + C_{1}$ ，其中  $B_{1}$ 与  $C_{1}$ 是仅由操作数 FP 格式决定的偏置校正与补偿项。由于这些校正值对每个 GEMM 行而言是常数，且与权重  $W_{q}$  的具体数值无关，AxCore 将其从每个 PE 中抽离，放入集中式 PreAdd 模块（图 11b 所示）。PreAdd 对每行仅计算一次共享项  $T = A - B_{1} + C_{1}$ ，并将其流式传输给所有 PE。这样，每个 PE 只需执行轻量的整数加法  $R = T + \mathrm{Align}(W_{q})$ ，显著简化数据通路并减少芯片面积。

图 12 对比了使用与不使用该技术的 mpFPMA PE 设计。在基线设计（图 12a）中，每个 PE 直接接收高精度激活  $A$ （如 FP16）与经 SNC 处理后的权重  $W_{q}$ （如 FP4）。由于 SNC 输出为 S1E3M2 格式并需与 FP16 指数域对齐，计算  $A + W_{q}$  需要 7 位加法器（5 位来自对齐指数，2 位来自尾数）。为应用校正，还需额外一个 15 位加法器计算  $C_{1} - B_{1}$ ，覆盖指数与尾数字段。项  $C_{1} - B_{1}$ 可被视为  $-B_{1}$ 与  $C_{1}$ 的拼接，因为它们作用于不同位域。尽管功能可行，但该双加法器结构导致宽数据通路并在每个 PE 中大量复制逻辑。相反，在使用校正前移后（图 12b），激活  $A$  与  $-B_{1} + C_{1}$  在阵列外通过一个 15 位加法器合并，并得到预计算值  $T$ ，随后沿行传递。每个 PE 只需用一个 7 位加法器将  $T$  与对齐后的  $W_{q}$  相加；对于 E3M2 权重与 FP16 激活（E5M10），该位宽已足够。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/e7bfbf7e320bb5510d7ebc0549177f39c06da9c01ca550af921c10684acb4a8a.jpg)
(a) Partial Add in PEs

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/b781b867ae90624f79fcbc823f01527ec90b704ea54a72b9837c7843ec8dc31c.jpg)
(c) Norm Module
Figure 11: Optimizations for resource sharing across PEs, including advanced correction in PreAdd module and postponed normalization in Norm module.

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/d7c3be4d2dcb589c68b3d6c578449234b21c6a360907d50db05c186855a6e004.jpg)
(a) mpFPMA PE without Correction Advancing.

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/21a12bb20d0db9c5d0d12c659effd3592220a6912fc57d397c9eef8535670167.jpg)
(b) mpFPMA PE with Correction Advancing.
Figure 12: Comparison of different designs of mpFPMA.

5.3.2 归一化后移。传统 GEMM 架构中，浮点加法通常在 PE 内完成归一化以保持精度。但该做法由于需要前导零检测（LZD）、移位、舍入等操作，会引入显著面积与延迟。受 [14, 23] 启发，AxCore 将归一化推迟到 PE 外部的共享 Norm 模块，并保留  $N_{M_a} + 2$ 位尾数（其中  $N_{M_a}$ 是激活尾数位宽），以维持数值精度并提供额外整数位防止溢出。每个 PE 在不归一化的情况下累加部分结果，产生由多个字段（符号、指数、整数、小数）组成的中间和，如图 11a 所示。这些结果被传至 Norm 模块，在其中通过流水化归一化完成最终输出，包括 Abs、LZD、Cmp、Round 等组件，如图 11c 所示。将归一化从每个 PE 中卸载到共享模块，可在  $n \times n$ 阵列中将逻辑重复减少  $n$ 倍，从而提升可扩展性与能效。

5.3.3 基于 FPMA 的反量化。分组量化要求每个输出通道按浮点缩放因子进行缩放。与使用乘法器不同，AxCore 使用 FPMA 来实现反量化，形成 AxScale 模块的基础。累加完成后，量化输出  $O_{q}$  的反量化为：

$$
O = O _ {q} + S - B + C _ {2} \tag {17}
$$

其中  $S$ 为缩放因子的二进制表示， $B$ 为格式相关偏置， $C_2$ 为补偿常数。该设计将反量化逻辑简化为两次整数加法，从而在后处理流水线中实现低成本缩放。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/2698e1c56f1d2aa363fce49379232203e3dc595c5646fce0414351af67811b80.jpg)
Figure 13: Architecture of AxCore-based LLM Accelerator.

# 5.4 基于 AxCore 的 LLM 推理加速器

图 13 展示了基于 AxCore 的 LLM 推理加速器的完整系统架构。与现有加速器 [22, 40] 类似，该设计围绕一个面向量化模型优化的紧耦合 GEMM 流水线组织。加速器核心是 GEMM 单元（AxCore），由二维 Tile 阵列组成，每个 Tile 由多个 mpFPMA PE 构成。为准备计算数据，Weight Buffer 存储量化后的模型权重，Unified Buffer 存放激活与中间数据。体系结构还包含 Vector Unit，用于辅助逐层向量运算；以及控制单元（CTRL），用于编排指令调度与数据流。与片外 DRAM 的数据通信由连接缓冲区的内存接口管理。该模块化且适合脉动阵列的设计，使 AxCore 能在低比特量化条件下高效支持大规模 Transformer 推理。

# 6 评估

# 6.1 实验设置

6.1.1 精度评估设置。我们在两类广泛使用的 LLM 系列（OPT 与 LLaMA2）上评估 AxCore 与基线设计。所有模型采用成熟的仅权重量化方法 [12] 量化为 4-bit，OPT 的分组大小为 128，LLaMA2 的分组大小为 64 [11, 15, 29]。对按块自适应格式量化，使用来自 Pile 数据集 [16] 的小型校准集合以避免过拟合。块大小对 OPT 设置为  $128 \times 64$ ，对 LLaMA2 设置为  $64 \times 64$ 。遵循先前工作 [22, 40]，我们在 WikiText-2 [34] 上用困惑度（PPL）评估模型性能，序列长度为 2048，值越低表示精度越好。此外，在零样本评估中，我们使用四个基准数据集：ARC-e [8]、HellaSwag [52]、PiQA [4] 和 Winogrande [2]，通过 lm-eval-harness 框架 [17] 进行评测。

6.1.2 硬件评估设置。为评估硬件效率，我们用 SpinalHDL [13] 实现 AxCore，并使用 Synopsys Design Compiler 在 28nm 台积电工艺节点上综合生成的 Verilog RTL。所有设计在相同目标频率（1GHz）下综合，并归一化为相同峰值吞吐（以 TOPS 计）。为公平比较，基线与 AxCore 均采用  $64 \times 64$  脉动阵列配置并进行  $4 \times$ 4 的平铺。为探索不同精度设置的性能，我们定义了多种评估场景，涵盖权重类型（INT4、FP4、INT8、FP8）与激活格式（FP16、BF16、FP32）的组合。我们基于开源周期级模拟器 DNNWeaver [44] 开发模拟器进行性能评估。SRAM 模块功耗使用 CACTI [37] 模拟。所有加速器配置采用相同 SRAM 大小。

6.1.3 基线。我们将 AxCore 与四类代表性 GEMM 加速器基线比较：浮点 GEMM 核（FPC）[22]、FPMA、FIGNA [22]、FIGLUT [40] 与 Tender [25]。FPC：在每个 PE 中使用标准浮点融合乘加（FMA）单元，累加器为 FP32，与 FIGNA 与 FIGLUT 的配置一致。FPMA：用原始 FPMA 逻辑替换 FP 乘法器。在 PE 内累加时，对 FP16/BF16 激活使用 FP16/BF16 加法器，对 FP32 激活使用 FP32 加法器。FIGNA：面向仅权重量化 LLM 的最先进 FP-INT 混合精度 GEMM 单元。FIGLUT：面向 LLM 的最先进 LUT 基 FP-INT GEMM 设计。Tender：面向 LLM 的最先进纯 INT 非混合精度 GEMM 设计。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/12a483cb85750b6d807aa35af78d3eeb8393113ba0b9beb4eb2acc8f5847fcee.jpg)
Figure 14: Normalized area breakdown of processing element (PE) under different formats.

# 6.2 面积效率

6.2.1 mpFPMA PE 的面积效率。图 14 给出了在六种数据类型配置下单个 PE 的归一化面积拆分，包括乘法逻辑、加法逻辑、次正规数转换（SNC）及其他组件。FIGLUT 缺乏详细组件数据，因此其面积归入 “Others”。在所有设计中，FPC 因昂贵的浮点单元而面积最大，FPMA 通过近似降低了乘法器面积。AxCore 在所有格式下都实现了最小 PE 面积，这归因于其 mpFPMA 设计完全消除乘法器。与 FIGLUT 相比，在 W4-FP32 情形下 AxCore 最多将 PE 面积降低  $34%$ ，在 W4-FP16 与 W4-BF16 中分别降低  $31%$  与  $22%$ 。与 FIGNA 相比，AxCore 在 4-bit 格式下降低 PE 面积  $32% - 39%$ ，在 8-bit 格式下降低  $43% - 56%$ 。值得注意的是，SNC 在 AxCore 中引入的开销极小，平均仅占 PE 总面积的  $3.5%$ 。

6.2.2 跨 GEMM 设计的面积效率。图 15 给出了不同设计与输入格式下 GEMM 单元的归一化面积拆分。面积拆分分为 PE 阵列（由所有 PE 构成）与 Others（位于激活数据通路上的各类预处理与后处理模块）。AxCore 在所有设置下都保持最低面积，优于 FIGNA 与 FIGLUT。在 4-bit 权重场景中，相比 FIGLUT，AxCore 在 W4-FP16、W4-BF16、W4-FP32 下分别降低总面积  $31%$ 、 $26%$ 、 $34%$ ；相比 FIGNA 分别降低  $37%$ 、 $36%$ 、 $29%$ 。在 8-bit 设置中，AxCore 相对 FIGLUT 平均降低  $25%$ ，相对 FIGNA 平均降低超过  $55%$ 。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/b8e5fb7b6cf5c89602d461c0483adcd8c8f38c67b8b59912b5f3f5ed6fd911c6.jpg)
Figure 15: Normalized area breakdown of the GEMM unit under six input format configurations, decomposed into the PE array and shared modules (Others).

# 6.3 计算密度

图 16 给出了六种输入格式配置下 GEMM 阵列的归一化计算密度（TOPS/mm $^2$ ），聚焦于 PE 阵列并排除最终累加阶段。结果以传统 FP32 设计（FPC）为基准归一化。由于紧凑的 mpFPMA 数据通路、无乘法器设计与集中式校正逻辑，AxCore 在所有格式下均提供最高计算密度。在 W4-FP16 设置中，AxCore 相对 FPC 提升  $6.7\times$ ，显著优于 FIGNA（ $4.0\times$ ）与 FIGLUT（ $4.3\times$ ）。在 W4-FP32 设置中，AxCore 相对 FPC 提升  $12.5\times$ ，并分别以  $1.4\times$  与  $1.5\times$  超过 FIGNA 与 FIGLUT。其他格式也呈现类似趋势：AxCore 在 W4-BF16 达到  $5.3\times$ ，在 W8-FP16 达到  $6.2\times$ 。即使在 W8-FP32 等更高精度配置下，AxCore 仍保持相对 FPC 的  $10\times$  密度提升。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/87ef2157348531d082ab7a6a211d39c1e6b9c7df5bbc856639fe71b0a1c59018.jpg)
Figure 16: Normalized compute density (TOPS/mm²) of the GEMM array across six input format configurations.

# 6.4 能效

图 17 展示了在两个 OPT 模型（13B 与 30B）上、不同输入数据类型条件下，AxCore 与基线加速器的归一化能耗拆分与 TOPS/W。我们在解码阶段测量能耗，batch size 为 32，输出序列长度为 1，与基线 [22, 40] 对齐。所有设计均提供充足带宽。结果表明 AxCore 在所有配置下均具有更优能效：能耗最低且 TOPS/W 最高。FIGNA 与 FIGLUT 在 8-bit 场景下能耗显著增加：FIGNA 的乘法器开销会随计算位宽平方增长，而 FIGLUT 的位串行架构需要更长计算周期，导致能耗上升。平均而言，AxCore 相对 FPC、FPMA、FIGNA 与 FIGLUT 分别实现  $2.2 \times$ 、 $1.5 \times$ 、 $1.1 \times$ 、 $1.3 \times$ 的总能耗降低，并分别实现  $6.4 \times$ 、 $3.1 \times$ 、 $1.4 \times$ 、 $2.0 \times$ 的 TOPS/W 提升。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/41cc20f2be567523bab706f1ee06560149641ca1832b844d8209e346923f9d98.jpg)
Figure 17: Normalized energy of AxCore and baseline accelerator designs across various data formats and model configurations.

# 6.5 精度评估

6.5.1 端到端模型精度。表 2 比较了 AxCore 与基线加速器的困惑度，并给出我们的优化项消融：次正规数转换（SNC）、常数补偿与格式感知量化。FPMA 使用 FP4 的就近舍入量化；FIGNA 使用 GPTQ 量化 [15] 评估；FIGLUT 的结果来自其论文 [40]。所有方法均采用对称量化，OPT 的分组大小为 128，LLaMA 2 的分组大小为 64。由于 FIGNA 与 FIGLUT 不量化注意力层，因此其精度反映线性层量化后的结果。表 2 显示，AxCore 在不同模型规模下均提供有竞争力或更优的困惑度。对 OPT 模型（2.7B 到 30B），AxCore 与现有 4-bit 加速器设计相比持平或更优，并在 OPT-6.7B、OPT-13B 等场景取得最低困惑度。对 LLaMA 2（7B、70B），AxCore 保持接近 FP16 的精度并优于 FIGNA 与 FPMA。
6.5.2 KV cache 量化。除线性层外，注意力机制也是 LLM 推理关键。为支持 AxCore 的端到端推理，我们将 KV cache 量化为 4-bit，并在累加维度上采用分组大小 64。对 OPT 模型：K cache 使用 E1M2，V cache 使用 E3M0；对 LLaMA2：K cache 使用 E2M1，V cache 使用 E3M0。最先进的纯整数加速器 Tender [25] 采用权重-激活量化，并通过分块与重排处理激活与 KV cache 中的离群值。表 2 结果显示 AxCore 在端到端推理精度上优于 Tender。我们还观察到 KV 量化的数据格式选择对精度影响显著，因此对 KV cache 进行数据格式校准是有价值的未来方向。
6.5.3 精度提升拆解。表 2 也强调了 AxCore 各设计特性对精度的贡献。从基础 mpFPMA（仅用 E2M1 格式且无常数补偿与 SNC）开始，困惑度较高（例如 OPT-6.7B 上为 11.83）。加入 SNC（mpFPMA+S）后困惑度降低（11.45），体现次正规数转换的收益。再引入常数补偿（mpFPMA+S+C）进一步提升精度（11.14）。AxCore 结合上述两项并加入格式感知量化，在 4-bit 设计中取得最佳结果（例如 OPT-6.7B 上 11.01，LLaMA 2 7B 上 5.65）。此外，加入 KV cache 量化（AxCore-KV）带来很小精度损失（例如 OPT-6.7B 上 11.18）。

Table 2: Perplexity comparison across OPT and LLaMA 2 models. mpFPMA: base mpFPMA; mpFPMA+S: mpFPMA + SNC; mpFPMA+S+C: mpFPMA + SNC + compensation; AxCore: mpFPMA + SNC + compensation + format-aware quantization; AxCore-KV: AxCore + KV cache quantization.

<table><tr><td rowspan="2">Method</td><td rowspan="2">Bits W/A/KV</td><td colspan="4">OPT (Perplexity↓)</td><td colspan="2">LLaMA 2</td></tr><tr><td>2.7B</td><td>6.7B</td><td>13B</td><td>30B</td><td>7B</td><td>70B</td></tr><tr><td>FP16</td><td>16/16/16</td><td>12.47</td><td>10.86</td><td>10.13</td><td>9.56</td><td>5.47</td><td>3.32</td></tr><tr><td>INT4</td><td>4/16/16</td><td>13.41</td><td>11.28</td><td>10.55</td><td>9.95</td><td>5.78</td><td>3.51</td></tr><tr><td>FP4</td><td>4/16/16</td><td>12.97</td><td>11.10</td><td>10.40</td><td>9.82</td><td>5.70</td><td>3.46</td></tr><tr><td>FPMA</td><td>4/16/16</td><td>13.40</td><td>11.37</td><td>10.56</td><td>9.93</td><td>5.82</td><td>3.53</td></tr><tr><td>mpFPMA</td><td>4/16/16</td><td>13.83</td><td>11.83</td><td>10.80</td><td>9.99</td><td>\</td><td>\</td></tr><tr><td>mpFPMA+S</td><td>4/16/16</td><td>13.24</td><td>11.45</td><td>10.49</td><td>9.86</td><td>\</td><td>\</td></tr><tr><td>mpFPMA+S+C</td><td>4/16/16</td><td>13.12</td><td>11.14</td><td>10.25</td><td>9.74</td><td>\</td><td>\</td></tr><tr><td>FIGNA [22]</td><td>4/16/16</td><td>12.87</td><td>11.04</td><td>10.23</td><td>9.62</td><td>5.69</td><td>3.42</td></tr><tr><td>FIGLUT [40]</td><td>4/16/16</td><td>12.73</td><td>11.08</td><td>10.33</td><td>9.70</td><td>\</td><td>\</td></tr><tr><td>AxCore</td><td>4/16/16</td><td>12.87</td><td>11.01</td><td>10.20</td><td>9.60</td><td>5.65</td><td>3.40</td></tr><tr><td>AxCore-KV</td><td>4/16/4</td><td>\</td><td>11.18</td><td>10.59</td><td>9.79</td><td>5.82</td><td>3.48</td></tr><tr><td>Tender [25]</td><td>8/8/4</td><td>\</td><td>14.51</td><td>13.33</td><td>14.49</td><td>\</td><td>\</td></tr><tr><td>Tender [25]</td><td>4/4/4</td><td>\</td><td>17.09</td><td>21.91</td><td>21.39</td><td>\</td><td>\</td></tr></table>

Table 3: Zero-shot performance on four benchmark datasets. Higher scores indicate better accuracy.

<table><tr><td>Model</td><td>Method</td><td>Arc-e</td><td>Hella.</td><td>Piqa</td><td>Wino.</td><td>Avg.(↑)</td></tr><tr><td rowspan="4">LLaMA2 70B</td><td>FP16</td><td>82.03</td><td>84.13</td><td>82.86</td><td>78.61</td><td>81.91</td></tr><tr><td>INT4</td><td>81.31</td><td>83.37</td><td>82.37</td><td>78.37</td><td>81.36</td></tr><tr><td>FP4</td><td>81.99</td><td>83.50</td><td>82.59</td><td>78.37</td><td>81.61</td></tr><tr><td>AxCore</td><td>82.11</td><td>83.79</td><td>82.59</td><td>78.61</td><td>81.78</td></tr><tr><td rowspan="4">OPT 30B</td><td>FP16</td><td>65.36</td><td>72.31</td><td>78.18</td><td>68.35</td><td>71.05</td></tr><tr><td>INT4</td><td>63.97</td><td>71.43</td><td>78.24</td><td>67.40</td><td>70.26</td></tr><tr><td>FP4</td><td>65.03</td><td>71.63</td><td>77.97</td><td>67.01</td><td>70.41</td></tr><tr><td>AxCore</td><td>64.86</td><td>72.08</td><td>78.07</td><td>68.03</td><td>70.76</td></tr></table>

6.5.4 零样本性能。我们也在四个标准零样本基准数据集（ARC-e [8]、HellaSwag [52]、Piqa [4]、Winogrande [2]）上评估 AxCore，使用 lm-eval-harness 框架 [17]。表 3 总结结果。对 LLaMA2 70B，AxCore 的平均准确率为  $81.78%$ ，与 FP16 基线（ $81.91%$ ）相当，并优于 INT4（ $81.36%$ ）与 FP4（ $81.61%$ ）量化实现。在各单项基准上 AxCore 也保持稳定表现。对 OPT 30B，AxCore 平均准确率为  $70.76%$ ，接近 FP16 基线（ $71.05%$ ）。
6.5.5 数值精度。我们用信噪比（SNR）评估 AxCore 的数值精度，SNR 定义为精确矩阵乘结果功率与近似噪声功率之比（以 dB 计）[9]。更高 SNR 表示更好地保持近似结果的幅值与方向。我们测试了从 128 到 32,768 的 fan-in（LLM 中常见），输入数据为均匀分布。图 18 显示，SNC 在所有测试矩阵规模上都能稳定提升 SNR；结合补偿可进一步提升。随机舍入在几乎无成本下带来正常的精度提升，但对 E2M1 格式无效，因为其次正规值可被精确映射到规范化表示。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/ae4a2e9cce287790c217d64f3887783240633e3497db8928a163ff0fab67f307.jpg)
Figure 18: Signal to Noise Ratio (SNR) analysis of AxCore. mpFPMA: base mpFPMA; S: subnormal number conversion (SNC); C: compensation; SR: stochastic rounding.

# 6.6 与非 mpGEMM 设计的对比

为展示 AxCore 使用高精度激活的混合精度设计优势，我们将其与纯整数加速器 Tender [25] 进行对比。如图 19 所示，AxCore（W4A16KV4）相比 Tender 的 W8A8KV4 与 W4A4KV4，在计算密度与精度上都更优。具体而言，在 FP16 与 BF16 激活下，AxCore 的计算密度分别比 Tender W8A8KV4 高  $1.72 \times$  与  $1.86 \times$ ，并且也超过 Tender 的 W4A4KV4 密度。在精度方面，AxCore 在 OPT 模型上始终保持更低困惑度。例如在 OPT-30B 上，AxCore 的困惑度为 9.79，而 Tender W8A8KV4 为 14.49，Tender W4A4KV4 为 21.39。上述结果表明：AxCore 的仅权重量化结合高精度激活与 FPMA，能在效率与精度之间取得更优权衡。

![](https://cdn-mineru.openxlab.org.cn/result/2025-12-06/3cfb4f62-c984-47e1-aaf6-4235bb1cc126/d565db216edbd79f7dc2cc6072c4b33b6162af9a9b7f3fb14d11f90643d995b3.jpg)
Figure 19: Comparison with integer-based non-mix-precision GEMM accelerator Tender [25].

# 7 结论

本文提出 AxCore：一种量化感知的近似 GEMM 单元，可用于 LLM 推理中的高效混合精度矩阵乘。通过将浮点乘法近似（FPMA）与低比特浮点量化相结合，AxCore 消除了乘法器并显著简化了每个 PE 的逻辑。据我们所知，AxCore 是首个在 LLM 推理中挖掘 FPMA 潜力的体系结构。AxCore 集成了一组轻量但有效的技术：次正规数转换、基于均值的误差补偿，以及自适应格式感知量化。评估结果表明：相对 FP 基线，AxCore 的计算密度最高提升  $12.5 \times$ ，相对 INT4 加速器可节省  $50%$  到  $70%$  的面积，并且困惑度更低。尽管 AxCore 处理标准低比特 FP 格式，将其扩展到自定义数据类型 [19, 21] 或基于块的格式 [9] 仍是有价值的未来方向。

# 致谢

本工作得到国家重点研发计划（No. 2024YFB4504200）与广州-港科大（广州）联合基金（No.2025A03J3568）支持。我们也感谢 AMD 异构加速计算集群（HACC）项目 [3] 提供硬件资源访问。
