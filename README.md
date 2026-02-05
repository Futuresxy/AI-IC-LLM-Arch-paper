<img width="1707" height="63" alt="image" src="https://github.com/user-attachments/assets/05ae4d79-3d66-490d-9e45-df88a70e143a" /><img width="2588" height="497" alt="image" src="https://github.com/user-attachments/assets/3b3c2b46-d3e6-441b-96ea-87a153af1b25" /><img width="1662" height="171" alt="image" src="https://github.com/user-attachments/assets/0c445b23-8774-48bd-b802-32319ec7dab0" /># AI-IC-LLM-Arch-paper
Engram 的训练并非简单的“填空”，而是一种“知识蒸馏”过程：
 阶段一：记忆预训练 (Memory Pre-training)
操作：冻结 Transformer 主干网络（Backbone），只更新 Engram 的 Embedding Table。
数据：使用高知识密度的语料（如 Wikipedia、GitHub 代码库）。
目标：强制模型仅依靠 Engram 提供的 Embedding 来预测下一个 Token。这实际上是强迫模型将“知识性信息”（如 Fact）压缩并写入到 Hash Table 的对应槽位中。
结果：此时的 Embedding Table 变成了一个高度浓缩的“外部大脑”，存储了静态的世界知识。
 
阶段二：联合微调 (Joint Training)
操作：解冻主干网络，训练 Gating（门控） 和 Fusion（融合） 层。
目标：门控层教会模型“如何查字典”。即让模型学会判断：什么时候该信赖 Engram（如回答事实），什么时候该信赖自己的推理（如逻辑分析）。融合层教会模型在扩大查表视野的同时，可以把碎片化的查询结果联系起来变得语义连贯。
这说明：早期稠密计算块的计算强度足以提供一 个足够的时间窗口，以掩盖检索延迟。 

原因分析：每一步的有效通信量与激活的嵌入表槽位数成正比，而不是与整个嵌入表的大小相关。


Engram-27B 路由专家从77变为55个，释放的参数分配一个5.7B的Engram存储（共26.7B参数） 25.7%  FP16:10.6GB
Engram-40B 将Engram模块拓展为18.5B（共39.5B参数）。↑~50%   FP16:34.5GB

如果使用670B+的超大模型，按照25%的比例计算Engram大小，存储表大小是167.5B     
	FP16: 335GB




并发请求：支持 1000 个并发用户  ;  生成速度：每个用户平均每秒生成 50 个 Token
总吞吐量：1000 * 50 = 50,000 Tokens/s
查询次数 (Lookups per Token): 2 Layers × 2 N-Grams × 8 Heads = 32 Lookups
总 IOPS: 50,000 * 32 = 1,600,000  (1.6M)  单次读取粒度: 512 Bytes (包含有效数据 128B + 384B 废弃/放大)
传输带宽 1.6M IOPS*512B ≈ 800 MB/s ≈ 0.8 GB/s
结论：0.8 GB/s 对于 PCIe 4.0 x16 (64 GB/s) 占比约 1.2%）带宽通道非常宽敞。
但是真正的瓶颈：IOPS 与 DDR 的“随机跳跃” ，问题不在于路不够宽，而在于访问可能太碎。
Engram 的哈希特性决定了这 160M IOPS 是完全随机访问 (Random Access)。DDR 工作原理：DDR 内存读取是基于 Row (行) 和 Column (列) 的。Row Miss 代价：如果这 8 个 Head 落在不同的物理 Page (Row) 上，内存控制器必须不断执行 Precharge (关闭旧行) -> Activate (打开新行) -> CAS (读列)。
1. DeepSeek Engram带来的存储系统随机小 I/O 风暴与架构设想 Haohai Mah00472395  加拿大研究院[2012实验室]
  


Gather-Copy-DMA 模式
不能让 GPU 直接去“零拷贝（Zero-Copy）”访问 CPU 内存中的离散地址。
可以采用 “CPU 端 Gather，PCIe 端 Batch” 的策略。
需求：50Tokens/s = 20ms / Token。
Engram I/O 总预估在 150 ~ 200 =(0.2 ms) 左右。
占比: 仅仅占 20ms 时间窗的 1%。


观点一：GPU直接访问CPU的DDR内存不再是“应急”，而是“标配”
以英伟达平台为例，以前大家觉得GB200里那个Grace CPU有点鸡肋，为了买GPU还得搭个CPU。但按DeepSeek的Engram架构：
GPU (Blackwell)：保留HBM，只存那75%的“逻辑推理参数”（MoE专家），专注算力。CPU (Grace)：利用板载的480GB LPDDR5X，存储那25%的“百科全书字典”（Engram Table）。Grace CPU瞬间从“调度员”升级为“外挂索引词典”。这种1个大内存CPU带2个GPU的比例，完美契合了“存算分离”的物理需求。
UB mesh灵衢总线，统一内存架构”含金量上升
观点二：HBM 盲目堆容量的时代可能结束，CPU 内存容量成为新战场如果这种架构普及，HBM的压力会骤减。因为海量的冷知识参数被挪到了CPU侧的LPDDR5X或者DDR5里。未来的趋势：
GPU侧：HBM不需要无限堆容量（192GB可能就够了），更专注于带宽（Bandwidth）。CPU侧：内存容量变得极度重要。未来支持CXL内存扩展或者像Grace这样板载大容量LPDDR的CPU会更吃香。
分级缓存：高频记忆驻留 GPU 显存，中频存 DRAM，低频存 NVMe SSD；通过预取与计算重叠隐藏延迟，搭配专用存储控制器优化哈希冲突，同时采用多队列并行处理 I/O 请求，在控制性能损耗的同时破解风暴难题。
1.【转载+抛转】DeepSeek 新文 Engram 架构将如何影响 AI 计算硬件架构？黄亦腾h00490900  联接芯片产品管理部[半导体业务部]

