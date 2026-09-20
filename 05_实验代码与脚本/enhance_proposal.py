#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完善安徽财经大学本科生科研创新基金项目申请书
"""

import zipfile
import xml.etree.ElementTree as ET
import re
from copy import deepcopy

# 论文中的文献综述内容（从main_PI-MoE_final_AE_ready.tex提取）
LITERATURE_REVIEW = """
一、研究背景与问题提出

风电并网规模不断扩大，短期功率预测已不只是统计意义上的误差比较，而会直接影响备用容量安排、弃风控制、平衡成本和调度安全。现有不少研究在清洗后的理想数据集上验证模型，但真实风电场SCADA数据常伴随缺失、重复、异常点、量测漂移和告警混杂等问题，导致模型在实际部署时出现性能塌缩。与此同时，控制器侧部署还受限于批量为1、时延敏感、内存受限等边缘计算条件，难以直接使用参数量大、推理路径固定的深层模型。

二、国内外研究现状

（一）风电功率预测方法研究现状

风电预测研究主要沿三条路线推进：

1. 分解重构与混合建模路线
通过分离趋势、波动和突变提升非平稳序列建模能力。早期研究主要基于统计方法如ARIMA模型，随后演变为结合信号分解技术（如变分模态分解VMD、经验模态分解EMD）与机器学习模型的混合方法。Duan等（2025）提出了针对非平稳信号的分解与元启发式优化框架；Li等（2023）将特征分离融合技术与改进CNN相结合用于综合能源系统多能负荷预测；Xue等（2020）采用VMD分解、KPCA降维与麻雀搜索算法优化LSSVM进行短期风电预测。

2. 基于深度学习的表征学习方法
近年来，基于注意力机制、图结构和长上下文学习的深度模型在时间序列预测领域取得显著进展。Informer（Zhou等, 2021）通过ProbSparse自注意力机制实现长序列高效预测；Autoformer（Wu等, 2021）引入自相关机制捕捉序列周期性；iTransformer（Liu等, 2024）采用"倒置"架构，将注意力机制应用于变量维度而非时间维度；PatchTST（Nie等, 2023）将时间序列分块处理，实现了与NLP领域的跨域迁移。然而，这些模型大多在清洗后的基准数据集上验证，较少在受损SCADA场景下评估。

3. 物理引导与可行性约束方法
功率曲线建模和物理约束引导是提升预测可行性的重要途径。Astolfi等（2015）利用数据挖掘技术分析风电场性能曲线；Martin-Calzada等（2025）采用参数迁移学习优化功率曲线建模中的传感器数据预处理；Liu等（2024）将物理信息嵌入强化学习框架用于极端事件下的概率风电预测。这些方法在训练阶段引入物理约束，但仍依赖于较丰富的辅助状态信息。

（二）SCADA数据质量与异常检测研究

SCADA数据质量问题已引起学术界和工业界的广泛关注。Leahy等（2019）系统分析了风电状态监测与可靠性分析中的数据质量问题；Letzgus（2020）研究了变点检测在正常行为模型鲁棒条件监测中的应用；Renstrom等（2020）采用深度自编码器实现风机系统级异常检测；Maldonado-Correa等（2020）通过系统文献综述总结了SCADA数据在风机状态监测中的应用；Vervlimmeren等（2025）提出了基于自编码器和机群中值滤波的SCADA驱动故障预测方法。

这些研究多聚焦于数据清洗和异常检测本身，而非直接面向受损数据条件下的预测任务。

（三）边缘计算与控制器侧部署研究

边缘智能在能源系统中的应用日益受到重视。Zhou等（2019）提出了边缘智能的概念框架，强调在边缘设备上实现人工智能推理的重要性；Gooi等（2023）综述了边缘智能在智能电网中的应用；Mwangi等（2024）探讨了软件定义工业物联网边缘网络在下一代海上风电场中的应用。这些研究明确了控制器侧推理的约束条件：批处理大小为1、延迟受限、内存受限、通信受限。

（四）预测经济价值与调度耦合研究

预测误差的经济影响是评估预测方法实用价值的关键。Wang等（2011）研究了风电预测不确定性与机组组合问题；Khaloie等（2020）提出了风电-热电-储能协调报价策略；Yan等（2025）提出了风电预测经济价值的数据驱动评估方法；Houben等（2023）研究了不确定性下的多能源系统微网最优调度。这些研究表明，预测质量的评估不能仅依赖统计指标，还需考虑其对下游决策的实际影响。

三、现有研究的不足与本项目的切入点

综上所述，现有研究存在以下三方面不足：

（1）数据侧：多数预测模型在清洗后的数据集上验证，与真实SCADA场景存在差距。风电场SCADA数据常伴随缺失、重复、异常点、量测漂移和告警混杂等问题，现有研究较少系统评估模型在受损数据条件下的性能表现。

（2）部署侧：控制器侧推理受限于批处理大小为1、延迟敏感、内存受限等边缘计算条件，但现有深度预测模型往往参数量大、推理路径固定，难以满足边缘部署需求。

（3）决策侧：预测质量的评估通常独立于下游决策，缺乏统一的预测-调度接口来比较不同预测模型对实际运行的影响。

针对上述不足，本项目拟构建面向受损SCADA场景的边缘鲁棒预测框架，将数据质量控制、边缘部署约束与物理可行性引导统一纳入预测模型设计与评估流程。
"""

# 参考文献列表
REFERENCES = """
主要参考文献

[1] Leahy K, Gallagher C, O'Donovan P, et al. Issues with Data Quality for Wind Turbine Condition Monitoring and Reliability Analyses[J]. Energies, 2019, 12(2): 201.

[2] Letzgus S. Change-point detection in wind turbine SCADA data for robust condition monitoring with normal behaviour models[J]. Wind Energy Science, 2020, 5: 1375-1397.

[3] Renstrom N, Bangalore P, Highcock E. System-wide anomaly detection in wind turbines using deep autoencoders[J]. Renewable Energy, 2020, 157: 647-659.

[4] Maldonado-Correa J, Martin-Martinez S, Artigao E, et al. Using SCADA Data for Wind Turbine Condition Monitoring: A Systematic Literature Review[J]. Energies, 2020, 13(12): 3132.

[5] Vervlimmeren I, Chesterman X, Verstraeten T, et al. Scalable SCADA-driven failure prediction for offshore wind turbines using autoencoder-based NBM and fleet-median filtering[J]. Wind Energy Science, 2025, 10: 2615-2637.

[6] Zhou H, Zhang S, Peng J, et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting[C]//AAAI, 2021: 11106-11115.

[7] Wu H, Xu J, Wang J, et al. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting[C]//NeurIPS, 2021: 22419-22430.

[8] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting[C]//ICLR, 2024.

[9] Nie Y, Nguyen N H, Sinthong P, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers[C]//ICLR, 2023.

[10] Wang S, Wu H, Shi X, et al. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting[C]//ICLR, 2024.

[11] Astolfi D, Castellani F, Garinei A, et al. Data mining techniques for performance analysis of onshore wind farms[J]. Applied Energy, 2015, 148: 220-233.

[12] Martin-Calzada P, Martin Sanchez P, Rodriguez-Sanchez F J, et al. Optimized Sensor Data Preprocessing Using Parameter-Transfer Learning for Wind Turbine Power Curve Modeling[J]. Sensors, 2025, 25(17): 5329.

[13] Liu Y, Wang J, Liu L. Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events[J]. Applied Energy, 2024, 376: 124068.

[14] Zhou Z, Chen X, Li E, et al. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing[J]. Proceedings of the IEEE, 2019, 107(8): 1738-1762.

[15] Gooi H B, Wang Z, Tang Y. Edge Intelligence for Smart Grid: A Survey of Applications, Techniques and Challenges[J]. CSEE Journal of Power and Energy Systems, 2023, 9(5): 1623-1640.

[16] Wang J, Botterud A, Bessa R, et al. Wind power forecasting uncertainty and unit commitment[J]. Applied Energy, 2011, 88(11): 4014-4023.

[17] Khaloie H, Abdollahi A, Shafie-khah M, et al. Coordinated wind-thermal-energy storage offering strategy in energy and spinning reserve markets using a multi-stage model[J]. Applied Energy, 2020, 259: 114168.

[18] Yan J, Li X, Wang H, et al. The economic value of wind power forecasting: a data-driven method and its application in various scenarios[J]. Energy, 2025, 340: 139151.

[19] Houben N, Cosic A, Stadler M, et al. Optimal dispatch of a multi-energy system microgrid under uncertainty: A renewable energy community in Austria[J]. Applied Energy, 2023, 337: 120913.

[20] Verdone A, Panella M, De Santis E, et al. A review of solar and wind energy forecasting: From single-site to multi-site paradigm[J]. Applied Energy, 2025, 392: 126016.
"""

# 详细的研究内容
RESEARCH_CONTENT = """
一、研究目的

本项目的核心研究目的是：在风电场SCADA数据存在缺失、噪声、漂移等质量问题的真实场景下，构建一个满足边缘部署约束（低延迟、低内存占用）且具有物理可行性的短期风电功率预测方法，并通过统一的预测-调度评估接口验证其对实际运行的经济价值。

具体而言，研究目的包括以下四个层面：

（1）数据层面：建立系统化的受损SCADA场景刻画与评估方法，量化不同数据质量问题对预测性能的影响程度。

（2）模型层面：设计轻量化的稀疏混合专家预测架构，通过动态路由机制在保证预测精度的同时满足边缘推理的时延与内存约束。

（3）约束层面：构建训练期可行性引导机制，将风电功率物理边界约束（额定功率、切入/切出风速等）嵌入模型学习过程，抑制不合理输出。

（4）评估层面：建立预测-调度耦合评估框架，除统计精度指标外，从调度成本、弃风量、后备替代频率等角度评价预测方法的实际应用价值。


二、研究意义

（一）理论意义

（1）丰富风电预测理论的适用边界。现有预测理论多基于理想数据假设，本项目将预测理论拓展至受损数据场景，为真实风电场环境下的预测建模提供理论参考。

（2）拓展边缘智能在时间序列预测领域的应用范式。通过稀疏混合专家架构与条件跳跃机制的设计，为资源受限场景下的深度学习模型部署提供新的技术路径。

（3）深化物理信息嵌入机器学习的方法论。本项目提出的可行性引导训练机制，为将领域物理知识融入数据驱动模型提供了一种可操作的实现方式。

（二）实践意义

（1）提升风电场运行管理的精细化水平。研究成果可直接应用于风电场控制器侧的功率预测模块，在数据质量不佳的条件下仍能提供可靠的预测支撑。

（2）降低预测系统的部署与运维成本。轻量化的模型设计使得预测算法可在现有风电场控制设备上运行，无需额外采购高性能计算设备。

（3）服务区域新能源消纳与电网安全。准确的功率预测有助于降低风电并网带来的不确定性，减少备用容量需求和弃风损失，支撑新型电力系统建设。

（4）促进学科交叉人才培养。本项目融合统计学、机器学习、能源系统等多个学科领域，有助于培养具有跨学科背景的复合型人才。


三、主要研究内容

本项目以风电场SCADA时间序列为研究对象，围绕"受损数据输入"与"边缘部署约束"两大核心挑战，开展以下四个方面的研究：

（一）受损SCADA场景刻画与数据预处理研究

系统分析风电场SCADA数据常见的质量问题，包括：
- 数据缺失：通信中断、传感器故障导致的间歇性缺失
- 量测噪声：电磁干扰、湍流感应抖动导致的随机误差
- 重复记录：数据采集系统异常导致的重复条目
- 量测漂移：传感器校准偏移导致的系统性偏差
- 异常扰动：设备故障、限功率运行导致的离群点

构建涵盖上述问题的受损SCADA场景库，设计统一的数据预处理流程（缺失值插补、异常检测、标准化处理），建立数据质量评估指标体系。

（二）基于稀疏混合专家的轻量预测模型研究

设计Ultra-LSNT（Ultra-Lightweight Sparse Network for Time-series）预测架构，核心创新点包括：

1. 多尺度分解嵌入层
采用移动平均分解将输入序列分离为趋势项与季节项，通过并行一维卷积提取多尺度特征，实现不同频率模式的自适应表征。

2. 稀疏Top-K路由机制
引入混合专家（Mixture of Experts, MoE）架构，通过Top-K路由机制为每个时间步选择最相关的K个专家网络（K远小于专家总数N），将计算复杂度从O(N·L·d²)降低至O(K·L·d²)。

3. 跳跃门控条件推理
设计Jump Gate机制，对平稳时段的表征进行条件跳过，进一步减少推理计算量，使模型能够根据输入复杂度动态调整计算资源分配。

4. 负载均衡正则化
引入辅助损失函数促进专家负载均衡，避免路由崩溃问题，提升模型训练稳定性。

（三）可行性引导的训练机制研究

针对风电功率预测中常见的超额定功率、负功率等不合理输出问题，构建FG-MoE（Feasibility-Guided MoE）训练框架：

1. 物理包络建模
基于风机空气动力学功率转换原理，构建由切入风速、额定风速、切出风速、额定功率等参数确定的可行功率包络。

2. 部署一致性代理
采用最后观测风速作为未来风速的代理（因未来风速在推理时不可获取），保证训练阶段使用的可行性约束信息与推理阶段一致。

3. 训练期约束嵌入
在损失函数中引入物理可行性惩罚项，对超出物理包络、违反功率边界、在切出区域非零输出等情况进行惩罚，引导模型学习符合物理规律的预测行为。

4. 与后处理裁剪的对比分析
系统对比训练期约束嵌入（FG-MoE）与推理期后处理裁剪（Clip-only）两种策略在预测精度与可行性之间的权衡关系。

（四）预测-调度耦合评估方法研究

建立统一的预测-调度评估接口，从实际应用价值角度评价预测方法：

1. 统计精度评估
采用R²、RMSE、MAE等指标评估预测精度，并在不同噪声水平（σ_eff ∈ {0.00, 0.15, 0.30, 0.45, 0.60}）下进行鲁棒性切片评估。

2. 边缘性能评估
测量模型的批处理-1推理延迟（CPU/ARM平台）、内存占用、参数量等指标，验证其满足边缘部署约束的能力。

3. 安全合规评估
计算超额定功率发生率、切入/切出区域违规率等安全性指标，评估预测输出的物理可行性。

4. 调度价值评估
构建日前调度筛选模型，以120天滚动窗口评估各预测方法在运营成本、弃风量、后备替代频率等方面的表现，验证预测误差向决策成本的传导效应。


四、研究方案与技术路线

（一）技术路线图

本项目技术路线如下：
1. 数据层：SCADA数据获取 → 质量分析 → 受损场景注入 → 预处理标准化
2. 模型层：多尺度分解 → 稀疏路由 → 条件跳跃 → 预测输出
3. 约束层：物理包络计算 → 可行性惩罚 → 联合优化
4. 评估层：统计指标 → 边缘性能 → 安全合规 → 调度价值

（二）实验设计

1. 数据集
- 主要数据集：新疆某风电场SCADA数据（15分钟采样，约12万条记录）
- 迁移验证：美国风电数据、空气质量数据、GEFCom负荷数据

2. 对比基线
- 统计方法：ARIMA、Persistence
- 机器学习方法：LightGBM、SVR
- 深度学习方法：DLinear、TimeMixer、iTransformer
- 长上下文基线：Chronos、Moirai

3. 主要实验
- 鲁棒性实验：在不同噪声、漂移、量化水平下评估模型性能
- 消融实验：验证稀疏路由、跳跃门控、可行性引导各组件的贡献
- 边缘性能实验：在CPU和ARM平台测量推理延迟与内存占用
- 调度价值实验：在统一调度接口下比较各模型的实际应用价值


五、已完成的研究工作

在本项目申请之前，课题组已开展了相关研究工作，取得了以下初步成果：

（一）模型架构设计与实现

已完成Ultra-LSNT预测模型的设计与实现，该模型融合了多尺度分解、稀疏Top-K路由、跳跃门控等核心组件。模型在PyTorch框架下实现，支持端到端训练与推理。

核心代码结构：
- 分解嵌入层：实现趋势-季节分离与多尺度卷积特征提取
- 稀疏路由模块：实现Top-K专家选择与负载均衡
- 跳跃门控模块：实现条件推理路径跳过
- 可行性约束层：实现物理包络计算与训练期惩罚

（二）实验验证与结果分析

已完成以下主要实验：

1. 鲁棒性验证实验
在受损SCADA场景下，Ultra-LSNT在严重噪声水平（σ_eff=0.60）下仍保持R²=0.7863的正向预测能力，而对比方法DLinear在该条件下R²降至-0.9903。清洁条件下各模型R²：iTransformer 0.9012、Ultra-LSNT 0.9346、DLinear 0.9494。

2. 边缘性能评估
在批处理-1推理条件下，Ultra-LSNT-Lite优化版本的CPU延迟为1.171ms，ARM延迟为5.007ms，内存占用仅1.33MiB，满足边缘部署约束。

3. 可行性引导效果验证
对比三种配置：（1）无约束：清洁R²=0.8727，严重R²=0.8273，超额定发生率0.0550%；（2）后处理裁剪：清洁R²=0.4319，严重R²=0.3120，超额定发生率0%；（3）FG-MoE：清洁R²=0.8747，严重R²=0.8286，超额定发生率0.0092%。结果表明FG-MoE在保持高精度的同时显著提升了可行性。

4. 调度价值评估
在120天滚动调度评估中，Ultra-LSNT相比DLinear基线显著降低了运营成本和弃风量，验证了预测精度向调度价值的有效传导。

（三）论文与代码成果

相关研究成果已整理为学术论文，投稿至Applied Energy期刊。论文系统阐述了受损SCADA场景下的边缘鲁棒预测方法，包括Ultra-LSNT架构设计、FG-MoE训练机制、统一评估协议等内容。

代码实现已开源（GitHub: https://github.com/b1ue13e/Ultra-LSNT），包含数据预处理脚本、模型实现、训练与评估代码，支持研究结果的复现。


六、关键科学问题与创新点

（一）关键科学问题

1. 受损数据条件下的预测稳定性问题
如何在SCADA数据存在缺失、噪声、漂移等质量问题的条件下，保持预测模型的泛化稳定性，避免性能塌缩。

2. 资源约束条件下的模型效率问题
如何在边缘设备有限的计算资源和内存预算下，实现满足实时性要求的预测推理。

3. 物理约束与数据驱动的融合问题
如何将物理可行性约束有效嵌入数据驱动的模型训练过程，在不显著牺牲预测精度的前提下提升输出合理性。

（二）主要创新点

1. 研究视角创新
首次将受损SCADA数据、边缘部署约束、物理可行性控制三类实际约束纳入统一的预测研究框架，填补了现有研究主要基于理想数据假设、较少考虑部署约束的空白。

2. 模型架构创新
提出Ultra-LSNT轻量稀疏预测架构，通过Top-K路由与跳跃门控机制实现计算资源的动态分配，在保持预测精度的同时满足边缘推理的延迟与内存约束。

3. 训练机制创新
提出FG-MoE可行性引导训练机制，采用部署一致性的风速代理，将物理包络约束嵌入训练期优化，避免了后处理裁剪带来的精度损失。

4. 评估方法创新
建立预测-调度耦合评估框架，除统计精度外，从运营成本、弃风量、后备替代频率等角度评价预测方法的实际应用价值。
"""


def extract_docx_content(docx_path):
    """提取docx文件的文本内容"""
    with zipfile.ZipFile(docx_path, 'r') as z:
        xml_content = z.read('word/document.xml')
    
    root = ET.fromstring(xml_content)
    texts = []
    for elem in root.iter():
        if elem.tag.endswith('}t'):
            if elem.text:
                texts.append(elem.text)
    
    return ''.join(texts)


def create_enhanced_proposal():
    """创建完善后的申请书"""
    try:
        from docx import Document
        from docx.shared import Pt, Inches, Cm
        from docx.enum.text import WD_PARAGRAPH_ALIGNMENT, WD_LINE_SPACING
        from docx.oxml.ns import qn
        
        # 读取原始文档
        doc = Document('安徽财经大学本科生科研创新基金项目申请书.docx')
        
        # 查找需要修改的部分并更新
        for para in doc.paragraphs:
            text = para.text.strip()
            
            # 找到"选题依据"部分，在其后添加文献综述
            if '选题依据' in text and len(text) < 10:
                # 找到该段落的索引
                idx = doc.paragraphs.index(para)
                # 清空原有简短内容
                for i in range(idx + 1, min(idx + 50, len(doc.paragraphs))):
                    if doc.paragraphs[i].text.strip() and '研究内容' not in doc.paragraphs[i].text:
                        doc.paragraphs[i].text = ''
                
                # 添加详细的文献综述
                lines = LITERATURE_REVIEW.strip().split('\n')
                for line in lines:
                    if line.strip():
                        new_para = doc.paragraphs[idx]._element.addnext(doc.add_paragraph()._element)
                        new_para = doc.paragraphs[idx + 1]
                        new_para.text = line.strip()
                        new_para.paragraph_format.first_line_indent = Cm(0.74)  # 首行缩进2字符
                        new_para.paragraph_format.line_spacing = 1.5
                        idx += 1
                
                # 添加参考文献
                ref_lines = REFERENCES.strip().split('\n')
                for line in ref_lines:
                    if line.strip():
                        new_para = doc.paragraphs[idx]._element.addnext(doc.add_paragraph()._element)
                        new_para = doc.paragraphs[idx + 1]
                        new_para.text = line.strip()
                        new_para.paragraph_format.first_line_indent = Cm(0.74)
                        new_para.paragraph_format.line_spacing = 1.5
                        idx += 1
                break
        
        # 找到"研究内容"部分进行扩充
        for i, para in enumerate(doc.paragraphs):
            text = para.text.strip()
            if '研究内容' in text and '主要研究内容' not in text and len(text) < 15:
                # 清空原有简短内容
                for j in range(i + 1, min(i + 100, len(doc.paragraphs))):
                    if '项目重难点' in doc.paragraphs[j].text or '创新点' in doc.paragraphs[j].text:
                        break
                    doc.paragraphs[j].text = ''
                
                # 添加详细的研究内容
                idx = i
                content_lines = RESEARCH_CONTENT.strip().split('\n')
                for line in content_lines:
                    if line.strip():
                        new_para = doc.paragraphs[idx]._element.addnext(doc.add_paragraph()._element)
                        new_para = doc.paragraphs[idx + 1]
                        new_para.text = line.strip()
                        # 设置首行缩进
                        new_para.paragraph_format.first_line_indent = Cm(0.74)
                        new_para.paragraph_format.line_spacing = 1.5
                        idx += 1
                break
        
        # 保存新文档
        output_path = '安徽财经大学本科生科研创新基金项目申请书_完善版.docx'
        doc.save(output_path)
        print(f"申请书已完善并保存至: {output_path}")
        return True
        
    except ImportError:
        print("错误: 需要安装python-docx库")
        return False
    except Exception as e:
        print(f"错误: {e}")
        return False


if __name__ == '__main__':
    create_enhanced_proposal()
