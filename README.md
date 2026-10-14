# 邓枭 | AI Infra / ML Systems

面向大模型推理、超大规模向量检索、多模态检索与系统性能优化。现任字节跳动后端研发工程师，南京大学计算机硕士，CCF-B 系统领域国际会议 IEEE ICDCS 2024 第一作者。

## 技术方向

- **AI Infra 与检索系统**：多模态特征生产，十万亿级向量检索，LLM 训练与推理系统原理。
- **分布式系统与数据工程**：HDFS、YARN、Spark、Hive、Flink，以及大规模离线任务的性能与稳定性治理。
- **系统软件与软硬件协同**：Linux 后端与内核机制、并发控制、DMA/I/O、FPGA PHY 和通信信号处理。

## 代表性工作

| 主题 | 核心工作 | 结果与材料 |
|---|---|---|
| 超大规模向量检索 | 在 HDFS、YARN、Spark/Hive 上设计并落地 IVF 检索链路，覆盖聚类训练、分片建库、cell 半径统计和阈值检索剪枝 | 十万亿级数据规模下召回率约 **96.52%**，净算力降至原方案的 **1/8**，端到端耗时降至约 **1/10**；[设计文档](./sys-designs/large_scale_vector_retrieval.md) |
| 有状态业务流程框架 | 借鉴 Linux 内核的稳定入口、operations、调度、等待/唤醒、统一退出和可观测性思想，抽象 Task DAG、异步回调、重试、锁与 diff/writeback | 将复杂申诉流程沉淀为可扩展、可恢复、可观测的通用执行框架；[设计文档](./sys-designs/appeal_system_kernel_style_design.md) |
| Linux/FPGA 软硬件协同系统 | 基于 OpenWiFi 与 Xilinx ZCU-102，贯通 Linux 内核态 Wi-Fi 驱动、DMA 数据通路和 FPGA PHY，实现双缓冲与逐帧信号处理，并在商用网卡和 USRP 平台完成验证 | 展示对内核 I/O、设备驱动、DMA、FPGA 逻辑与通信信号处理的跨层理解和端到端开发能力；第一作者论文发表于 CCF-B 系统领域会议 IEEE ICDCS 2024；[论文](./publications/CSS_Built-in_Channel_State_Scrambling_for_Secure_Wi-Fi_based_Sensing.pdf) |

## LLM Systems 技术专题

以下文档侧重数学原理、数据与参数 ownership、collective 通信、内存生命周期及性能权衡，用于展示对训练与推理系统的系统性理解。

| 主题 | 内容 |
|---|---|
| [LLM 推理引擎](./llm/inference_engine/inference_engine.md) | 请求状态机、prefill/decode 调度、Paged KV Cache、prefix caching、CUDA Graph、张量并行与性能建模 |
| [Speculative Decoding](./llm/speculative_decoding/speculative_decoding.md) | Draft/target 条件分布、接受率、residual distribution、KV Cache 事务语义与服务化权衡 |
| [Context Parallelism](./llm/context_parallel/context_parallel.md) | Ring Attention、分布式 KV Cache、百万 token 精确 prefill、负载均衡与通信复杂度 |
| [Expert Parallelism](./llm/expert_parallelism/expert_parallelism.md) | 稀疏 MoE 路由、dispatch/combine、All-to-All、负载均衡及 prefill/decode 部署差异 |
| [ZeRO 与 FSDP](./llm/fsdp/fsdp.md) | 参数、梯度和 optimizer state 的分片所有权、通信时序、内存模型与 overlap |
| [Pipeline Parallelism](./llm/pipeline_parallelism/pipeline_parallelism.md) | GPipe、microbatch、bubble、跨进程 autograd、1F1B 与多维并行组合 |
| [Megatron-LM](./llm/megatron/megatron.md) | Transformer 层内张量并行、通信边界与多维并行基础 |
| [Byte-Level BPE](./llm/tokenization/byte_level_bpe.md) | UTF-8 字节、可逆子词词表、训练与编码算法、复杂度及可并行性 |

## 研究与早期项目

- **编译原理**：使用 Flex/Bison 完成 C-minus 前端，并基于 LLVM 生成和优化 IR；[代码与说明](./coursework/compiler-principles/)。
- **计算机体系结构**：使用 Verilog 设计支持 57 条 MIPS 指令、CP0、异常与中断的五级流水线 CPU；[项目材料](./coursework/computer-architecture/)。
- **操作系统**：基于 xv6 实现系统调用、调度、多线程、同步、消息传递与写时复制；[实验报告](./coursework/os/xv6-operating-system-lab-report.pdf)。
- **分布式系统**：实现 Raft 的 Leader 选举、日志复制与故障恢复；[项目说明](./coursework/raft/README.md)。

## 仓库导航

| 路径 | 内容 |
|---|---|
| [`sys-designs/`](./sys-designs/) | 生产系统抽象与架构设计 |
| [`llm/`](./llm/) | LLM 训练、推理与基础算法专题 |
| [`publications/`](./publications/) | ICDCS 2024 论文及演示材料 |
| [`coursework/`](./coursework/) | 编译器、CPU、操作系统与分布式系统项目 |
