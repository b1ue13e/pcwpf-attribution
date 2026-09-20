$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$wdAlignParagraphLeft = 0
$wdAlignParagraphCenter = 1
$wdCollapseEnd = 0
$wdLineSpace1pt5 = 1
$wdLineSpaceSingle = 0
$wdRowHeightAtLeast = 1
$wdCellAlignVerticalCenter = 1
$wdCellAlignVerticalTop = 0

$docPath = Join-Path $PSScriptRoot '安徽财经大学本科生科研创新基金项目申请书.docx'
$backupPath = Join-Path $PSScriptRoot '安徽财经大学本科生科研创新基金项目申请书_编辑前备份_20260311.docx'
$workingPath = Join-Path $PSScriptRoot '安徽财经大学本科生科研创新基金项目申请书_完善版_20260311.docx'
$workflowImage = Join-Path $PSScriptRoot 'workflow_pipeline.png'
$robustnessImage = Join-Path $PSScriptRoot 'fig_robustness_stress_test.png'
$dispatchImage = Join-Path $PSScriptRoot 'fig_dispatch_economics.png'

if (-not (Test-Path $docPath)) {
    throw "未找到目标文档：$docPath"
}

if (-not (Test-Path $backupPath)) {
    Copy-Item -Path $docPath -Destination $backupPath
}

Copy-Item -Path $backupPath -Destination $workingPath -Force

function Get-ContentRange {
    param([object]$Cell)

    $range = $Cell.Range.Duplicate
    $range.End = $range.End - 1
    return $range
}

function Move-SelectionToCellStart {
    param(
        [object]$Selection,
        [object]$Cell
    )

    $Selection.SetRange($Cell.Range.Start, $Cell.Range.Start)
}

function Clear-Cell {
    param(
        [object]$Selection,
        [object]$Cell
    )

    $Cell.Range.Text = ''
    Move-SelectionToCellStart -Selection $Selection -Cell $Cell
}

function Set-ParagraphStyle {
    param(
        [object]$Word,
        [object]$Selection,
        [string]$Kind
    )

    $Selection.Font.Bold = $false
    $Selection.Font.NameFarEast = '宋体'
    $Selection.Font.Name = 'Times New Roman'
    $Selection.Font.Size = 11
    $Selection.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $Selection.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $Selection.ParagraphFormat.SpaceBefore = 0
    $Selection.ParagraphFormat.SpaceAfter = 6
    $Selection.ParagraphFormat.FirstLineIndent = $Word.CentimetersToPoints(0.74)
    try {
        $Selection.ParagraphFormat.CharacterUnitFirstLineIndent = 2
    } catch {
    }

    switch ($Kind) {
        'Heading2' {
            $Selection.Font.Bold = $true
            $Selection.Font.Size = 11
            $Selection.ParagraphFormat.SpaceBefore = 6
            $Selection.ParagraphFormat.SpaceAfter = 3
            $Selection.ParagraphFormat.FirstLineIndent = 0
            try {
                $Selection.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Heading3' {
            $Selection.Font.Bold = $true
            $Selection.Font.Size = 11
            $Selection.ParagraphFormat.SpaceBefore = 3
            $Selection.ParagraphFormat.SpaceAfter = 3
            $Selection.ParagraphFormat.FirstLineIndent = 0
            try {
                $Selection.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Reference' {
            $Selection.Font.Size = 10.5
            $Selection.ParagraphFormat.SpaceBefore = 0
            $Selection.ParagraphFormat.SpaceAfter = 0
            $Selection.ParagraphFormat.FirstLineIndent = 0
            try {
                $Selection.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Caption' {
            $Selection.Font.Bold = $false
            $Selection.Font.Size = 10.5
            $Selection.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $Selection.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $Selection.ParagraphFormat.SpaceBefore = 3
            $Selection.ParagraphFormat.SpaceAfter = 3
            $Selection.ParagraphFormat.FirstLineIndent = 0
            try {
                $Selection.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
    }
}

function Set-RangeStyle {
    param(
        [object]$Word,
        [object]$Range,
        [string]$Kind
    )

    $Range.Font.Bold = $false
    $Range.Font.NameFarEast = '宋体'
    $Range.Font.Name = 'Times New Roman'
    $Range.Font.Size = 11
    $Range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
    $Range.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
    $Range.ParagraphFormat.SpaceBefore = 0
    $Range.ParagraphFormat.SpaceAfter = 6
    $Range.ParagraphFormat.FirstLineIndent = $Word.CentimetersToPoints(0.74)
    try {
        $Range.ParagraphFormat.CharacterUnitFirstLineIndent = 2
    } catch {
    }

    switch ($Kind) {
        'Heading2' {
            $Range.Font.Bold = $true
            $Range.Font.Size = 11
            $Range.ParagraphFormat.SpaceBefore = 6
            $Range.ParagraphFormat.SpaceAfter = 3
            $Range.ParagraphFormat.FirstLineIndent = 0
            try {
                $Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Heading3' {
            $Range.Font.Bold = $true
            $Range.Font.Size = 11
            $Range.ParagraphFormat.SpaceBefore = 3
            $Range.ParagraphFormat.SpaceAfter = 3
            $Range.ParagraphFormat.FirstLineIndent = 0
            try {
                $Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Reference' {
            $Range.Font.Size = 10.5
            $Range.ParagraphFormat.SpaceBefore = 0
            $Range.ParagraphFormat.SpaceAfter = 0
            $Range.ParagraphFormat.FirstLineIndent = 0
            try {
                $Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
        'Caption' {
            $Range.Font.Bold = $false
            $Range.Font.Size = 10.5
            $Range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
            $Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
            $Range.ParagraphFormat.SpaceBefore = 3
            $Range.ParagraphFormat.SpaceAfter = 3
            $Range.ParagraphFormat.FirstLineIndent = 0
            try {
                $Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
    }
}

function Type-Paragraph {
    param(
        [object]$Word,
        [object]$Selection,
        [string]$Text,
        [string]$Kind = 'Body'
    )

    Set-ParagraphStyle -Word $Word -Selection $Selection -Kind $Kind
    $Selection.TypeText($Text)
    $Selection.TypeParagraph()
}

function Write-ItemsToCell {
    param(
        [object]$Word,
        [object]$Selection,
        [object]$Cell,
        [object[]]$Items
    )

    $texts = foreach ($item in $Items) { $item.Text }
    $Cell.Range.Text = ($texts -join "`r")

    $paragraphs = $Cell.Range.Paragraphs
    for ($i = 1; $i -le [Math]::Min($Items.Count, $paragraphs.Count); $i++) {
        Set-RangeStyle -Word $Word -Range $paragraphs.Item($i).Range -Kind $Items[$i - 1].Kind
    }

    $Selection.SetRange($Cell.Range.End - 1, $Cell.Range.End - 1)
}

function Insert-CenteredPicture {
    param(
        [object]$Word,
        [object]$Selection,
        [string]$ImagePath,
        [string]$Caption,
        [double]$WidthCm = 11.8
    )

    if (-not (Test-Path $ImagePath)) {
        return
    }

    Type-Paragraph -Word $Word -Selection $Selection -Text $Caption -Kind 'Caption'
    Set-ParagraphStyle -Word $Word -Selection $Selection -Kind 'Caption'
    $shape = $Selection.InlineShapes.AddPicture($ImagePath, $false, $true)
    $shape.LockAspectRatio = $true
    $shape.Width = $Word.CentimetersToPoints($WidthCm)
    $Selection.TypeParagraph()
}

function Format-FrontTable {
    param(
        [object]$Word,
        [object]$Table
    )

    $Table.Range.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle
    $Table.Range.ParagraphFormat.SpaceBefore = 0
    $Table.Range.ParagraphFormat.SpaceAfter = 0
    $Table.Rows.AllowBreakAcrossPages = $true

    foreach ($cell in $Table.Range.Cells) {
        $cell.VerticalAlignment = $wdCellAlignVerticalCenter
        $range = Get-ContentRange -Cell $cell
        $text = ($range.Text -replace "`r", '' -replace "`a", '').Trim()
        $range.Font.NameFarEast = '宋体'
        $range.Font.Name = 'Times New Roman'
        $range.Font.Size = 11
        if ($text.Length -le 32 -and $text -notmatch '本项目聚焦|（限200字）') {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        } else {
            $range.ParagraphFormat.Alignment = $wdAlignParagraphLeft
            $range.ParagraphFormat.LineSpacingRule = $wdLineSpace1pt5
            $range.ParagraphFormat.SpaceAfter = 3
            $range.ParagraphFormat.FirstLineIndent = $Word.CentimetersToPoints(0.74)
            try {
                $range.ParagraphFormat.CharacterUnitFirstLineIndent = 2
            } catch {
            }
            $cell.VerticalAlignment = $wdCellAlignVerticalTop
        }
    }

    for ($i = 1; $i -le $Table.Rows.Count; $i++) {
        try {
            $Table.Rows.Item($i).HeightRule = $wdRowHeightAtLeast
            $Table.Rows.Item($i).Height = $Word.CentimetersToPoints(0.8)
        } catch {
        }
    }
}

function Format-ArgumentTable {
    param(
        [object]$Word,
        [object]$Table
    )

    $Table.Rows.AllowBreakAcrossPages = $true
    for ($i = 1; $i -le $Table.Rows.Count; $i++) {
        $labelCell = $Table.Cell($i, 1)
        $contentCell = $Table.Cell($i, 2)
        $labelCell.VerticalAlignment = $wdCellAlignVerticalCenter
        $contentCell.VerticalAlignment = $wdCellAlignVerticalTop

        $labelRange = Get-ContentRange -Cell $labelCell
        $labelRange.Font.NameFarEast = '宋体'
        $labelRange.Font.Name = 'Times New Roman'
        $labelRange.Font.Size = 11
        $labelRange.Font.Bold = $true
        $labelRange.ParagraphFormat.Alignment = $wdAlignParagraphCenter
        $labelRange.ParagraphFormat.SpaceBefore = 0
        $labelRange.ParagraphFormat.SpaceAfter = 0
        $labelRange.ParagraphFormat.LineSpacingRule = $wdLineSpaceSingle

        $contentRange = Get-ContentRange -Cell $contentCell
        $contentRange.Font.NameFarEast = '宋体'
        $contentRange.Font.Name = 'Times New Roman'
        $contentRange.Font.Size = 11
    }
}

$basisItems = @(
    @{ Kind = 'Heading2'; Text = '（一）研究背景与问题提出' },
    @{ Kind = 'Body'; Text = '风电并网规模持续扩大，短期功率预测已不只是统计意义上的误差比较，而会直接影响备用容量安排、弃风控制、平衡成本和调度安全。对于风电场而言，预测结果既是运行计划制定的依据，也是参与电力市场交易、调度协调和设备维护的重要信息，因此预测模型不能只在理想数据上“好看”，更要在真实生产场景下“能用”。' },
    @{ Kind = 'Body'; Text = '与公开基准数据不同，真实风电场SCADA数据通常伴随缺失、重复、异常点、量测漂移、通信中断与告警混杂等问题；同时，控制器侧部署还受到批量为1、时延敏感、内存受限、通信带宽有限等边缘计算条件的约束。若仍直接采用参数量大、推理路径固定的深层模型，往往会在受损数据和现场部署条件下出现精度下降、推理变慢甚至输出失真的问题。' },
    @{ Kind = 'Body'; Text = '基于论文现有工作，本项目拟围绕“受损SCADA场景下风电功率边缘鲁棒预测”这一主题，统筹考虑数据质量、模型鲁棒性、边缘部署效率和物理可行性四个维度，形成面向真实工程应用的研究框架，从而为新能源功率预测、调度优化和数据智能分析提供更贴近生产的解决方案。' },
    @{ Kind = 'Heading2'; Text = '（二）国内外研究现状与文献综述' },
    @{ Kind = 'Heading3'; Text = '1. 风电功率预测模型的发展脉络' },
    @{ Kind = 'Body'; Text = '早期风电功率预测主要依赖ARIMA、指数平滑等传统统计方法，这类方法在小样本、弱波动场景下具有较好可解释性，但面对强非线性、强非平稳的风电序列表现有限。随后，RNN/LSTM、CNN-LSTM等深度学习方法开始被用于挖掘时序依赖和局部特征。进入Transformer阶段后，Informer、Autoformer、PatchTST、iTransformer、TimeMixer等方法通过稀疏注意力、序列分解、变量维建模和多尺度混合等机制，显著提升了长序列建模能力，也为风电功率预测提供了新的模型基础。' },
    @{ Kind = 'Body'; Text = '与此同时，分解重构与混合建模路线也在持续发展。研究者尝试将VMD、EMD、小波分解、趋势季节分离等技术与机器学习或神经网络模型结合，以提升对非平稳信号的表达能力。这一方向表明，面对风电这类受气象因素强烈影响的时序问题，单一结构往往难以兼顾趋势、波动和突变，多模块协同、专家分工和动态融合成为更具潜力的研究路径。' },
    @{ Kind = 'Heading3'; Text = '2. SCADA数据质量与异常处理研究' },
    @{ Kind = 'Body'; Text = '围绕风电场SCADA数据质量问题，Leahy等、Letzgus、Renström等学者从条件监测、变点检测、深度自编码器异常识别等角度系统揭示了缺失、漂移、故障点和异常工况对风机状态分析和预测建模的影响；Maldonado-Correa等对SCADA在风机状态监测中的应用进行了系统综述；Vervlimmeren等进一步提出了面向海上风电机组的可扩展SCADA驱动故障预测方案。现有工作说明，数据质量治理本身已经形成较成熟研究脉络，但多数研究重点在于异常检测、健康监测或预处理流程，对“受损数据条件下的预测模型如何保持稳定性能”关注仍然不足。' },
    @{ Kind = 'Heading3'; Text = '3. 边缘智能与轻量化部署研究' },
    @{ Kind = 'Body'; Text = '随着边缘智能在工业互联网和智能电网中的应用不断深化，模型部署地点逐渐从云端前移至设备侧和站端。Zhou等提出边缘智能框架，Gooi等综述了边缘智能在智能电网中的关键应用与挑战，相关研究均强调了低时延、低内存占用、低通信依赖的重要性。对于风电功率预测而言，若模型不能在控制器侧实现稳定推理，就难以服务实时预报和现场闭环决策。因此，轻量化、稀疏计算和条件路由不只是模型优化技巧，而是支撑工程落地的必要条件。' },
    @{ Kind = 'Heading3'; Text = '4. 物理可行性与预测价值研究' },
    @{ Kind = 'Body'; Text = '除预测精度外，输出是否符合风电功率物理边界、是否能够降低调度成本，也已成为评价模型实用价值的重要维度。Astolfi等、Martin-Calzada等围绕风机功率曲线建模和传感器预处理开展研究，Liu等将物理信息嵌入极端事件条件下的风电概率预测；Wang、Khaloie、Houben、Yan等从机组组合、储能协调、多能系统调度和预测经济价值等角度表明，预测误差会沿着备用成本、购电成本、弃风水平和运行风险向下游决策传导。因此，将预测指标与调度价值联合评价，是提升研究工程解释力的关键方向。' },
    @{ Kind = 'Heading2'; Text = '（三）现有研究不足与本项目切入点' },
    @{ Kind = 'Body'; Text = '综上，现有研究仍存在三方面不足：其一，很多模型主要在清洗后的标准数据集上验证，缺少面向受损SCADA场景的系统评测；其二，深度预测模型往往默认较充足的算力资源，较少显式考虑控制器侧的低时延、低内存和批量为1等部署约束；其三，预测效果与下游调度价值常被分割评估，导致模型即便统计指标较优，也不一定能在实际运行中带来更低成本与更高安全性。' },
    @{ Kind = 'Body'; Text = '针对上述不足，本项目拟在论文现有成果基础上，构建受损SCADA场景库，设计面向边缘部署的稀疏混合专家预测模型，引入训练期可行性引导机制，并建立“预测精度—鲁棒性—边缘效率—调度价值”一体化评估框架。该研究既具备明确的问题导向，也有较扎实的前期实验支撑，能够在理论分析与工程应用之间形成有效衔接。' },
    @{ Kind = 'Heading2'; Text = '（四）主要参考文献' },
    @{ Kind = 'Reference'; Text = "[1] Leahy K, Gallagher C, O'Donovan P, et al. Issues with Data Quality for Wind Turbine Condition Monitoring and Reliability Analyses[J]. Energies, 2019, 12(2): 201." },
    @{ Kind = 'Reference'; Text = '[2] Letzgus S. Change-point detection in wind turbine SCADA data for robust condition monitoring with normal behaviour models[J]. Wind Energy Science, 2020, 5: 1375-1397.' },
    @{ Kind = 'Reference'; Text = '[3] Renstrom N, Bangalore P, Highcock E. System-wide anomaly detection in wind turbines using deep autoencoders[J]. Renewable Energy, 2020, 157: 647-659.' },
    @{ Kind = 'Reference'; Text = '[4] Maldonado-Correa J, Martin-Martinez S, Artigao E, et al. Using SCADA Data for Wind Turbine Condition Monitoring: A Systematic Literature Review[J]. Energies, 2020, 13(12): 3132.' },
    @{ Kind = 'Reference'; Text = '[5] Vervlimmeren I, Chesterman X, Verstraeten T, et al. Scalable SCADA-driven failure prediction for offshore wind turbines using autoencoder-based NBM and fleet-median filtering[J]. Wind Energy Science, 2025, 10: 2615-2637.' },
    @{ Kind = 'Reference'; Text = '[6] Zhou H, Zhang S, Peng J, et al. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting[C]//AAAI, 2021: 11106-11115.' },
    @{ Kind = 'Reference'; Text = '[7] Wu H, Xu J, Wang J, et al. Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting[C]//NeurIPS, 2021: 22419-22430.' },
    @{ Kind = 'Reference'; Text = '[8] Liu Y, Hu T, Zhang H, et al. iTransformer: Inverted Transformers Are Effective for Time Series Forecasting[C]//ICLR, 2024.' },
    @{ Kind = 'Reference'; Text = '[9] Nie Y, Nguyen N H, Sinthong P, et al. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers[C]//ICLR, 2023.' },
    @{ Kind = 'Reference'; Text = '[10] Wang S, Wu H, Shi X, et al. TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting[C]//ICLR, 2024.' },
    @{ Kind = 'Reference'; Text = '[11] Astolfi D, Castellani F, Garinei A, et al. Data mining techniques for performance analysis of onshore wind farms[J]. Applied Energy, 2015, 148: 220-233.' },
    @{ Kind = 'Reference'; Text = '[12] Martin-Calzada P, Martin Sanchez P, Rodriguez-Sanchez F J, et al. Optimized Sensor Data Preprocessing Using Parameter-Transfer Learning for Wind Turbine Power Curve Modeling[J]. Sensors, 2025, 25(17): 5329.' },
    @{ Kind = 'Reference'; Text = '[13] Liu Y, Wang J, Liu L. Physics-informed reinforcement learning for probabilistic wind power forecasting under extreme events[J]. Applied Energy, 2024, 376: 124068.' },
    @{ Kind = 'Reference'; Text = '[14] Zhou Z, Chen X, Li E, et al. Edge Intelligence: Paving the Last Mile of Artificial Intelligence With Edge Computing[J]. Proceedings of the IEEE, 2019, 107(8): 1738-1762.' },
    @{ Kind = 'Reference'; Text = '[15] Gooi H B, Wang Z, Tang Y. Edge Intelligence for Smart Grid: A Survey of Applications, Techniques and Challenges[J]. CSEE Journal of Power and Energy Systems, 2023, 9(5): 1623-1640.' },
    @{ Kind = 'Reference'; Text = '[16] Wang J, Botterud A, Bessa R, et al. Wind power forecasting uncertainty and unit commitment[J]. Applied Energy, 2011, 88(11): 4014-4023.' },
    @{ Kind = 'Reference'; Text = '[17] Khaloie H, Abdollahi A, Shafie-khah M, et al. Coordinated wind-thermal-energy storage offering strategy in energy and spinning reserve markets using a multi-stage model[J]. Applied Energy, 2020, 259: 114168.' },
    @{ Kind = 'Reference'; Text = '[18] Yan J, Li X, Wang H, et al. The economic value of wind power forecasting: a data-driven method and its application in various scenarios[J]. Energy, 2025, 340: 139151.' },
    @{ Kind = 'Reference'; Text = '[19] Houben N, Cosic A, Stadler M, et al. Optimal dispatch of a multi-energy system microgrid under uncertainty: A renewable energy community in Austria[J]. Applied Energy, 2023, 337: 120913.' },
    @{ Kind = 'Reference'; Text = '[20] Verdone A, Panella M, De Santis E, et al. A review of solar and wind energy forecasting: From single-site to multi-site paradigm[J]. Applied Energy, 2025, 392: 126016.' }
)

$researchItems = @(
    @{ Kind = 'Heading2'; Text = '（一）研究目的' },
    @{ Kind = 'Body'; Text = '本项目面向受损SCADA场景下的风电功率短期预测任务，目标是在“数据受损”“边缘部署”“物理可行”“应用落地”四类约束同时存在的条件下，构建一套兼顾精度、鲁棒性与效率的预测方案。与只关注单一指标的研究不同，本项目强调从真实工程问题出发，把模型设计、训练约束和应用评估放在同一研究链条中统筹推进。' },
    @{ Kind = 'Body'; Text = '具体而言，项目拟实现以下四个目标：一是构建适用于受损SCADA输入的轻量稀疏混合专家预测模型，在保证预测质量的同时降低边缘端推理开销；二是设计训练期可行性引导机制，将额定功率边界、风速代理信息和物理约束嵌入模型学习过程，减少超限输出；三是建立覆盖噪声、漂移、缺失和量化扰动的统一鲁棒性评估协议，客观刻画模型在复杂场景中的稳定性；四是搭建预测—调度耦合评估接口，从备用成本、弃风水平和综合运行成本等维度评估模型的实际价值。' },
    @{ Kind = 'Heading2'; Text = '（二）研究意义' },
    @{ Kind = 'Body'; Text = '从理论层面看，本项目有助于拓展稀疏MoE、条件计算与物理引导学习在时间序列预测领域的应用边界。现有长序列预测研究大多基于干净数据和通用模型比较，而本项目把受损SCADA、边缘部署与可行性控制纳入统一框架，有望为“真实数据条件下的时序建模”提供更贴近应用的研究范式。' },
    @{ Kind = 'Body'; Text = '从应用层面看，研究成果可直接服务于风电场功率预测、站端监控和电网调度。若模型能够在受损数据和资源受限设备上稳定工作，就能在不额外依赖高性能服务器的情况下，为风电场提供更可靠的实时预测支撑，进而降低备用安排压力、减少弃风损失、提升新能源消纳水平。' },
    @{ Kind = 'Body'; Text = '从人才培养和方法沉淀层面看，本项目融合统计建模、机器学习、能源系统分析和工程部署等多方面内容，能够形成数据构建、模型实现、实验验证、可视化分析和论文撰写的完整训练链条，为后续本科科研项目和毕业论文工作积累可复用的代码、图表和方法经验。' },
    @{ Kind = 'Heading2'; Text = '（三）主要研究内容' },
    @{ Kind = 'Heading3'; Text = '1. 受损SCADA场景库与统一预处理流程构建' },
    @{ Kind = 'Body'; Text = '围绕真实风电场SCADA常见问题，系统构建缺失、重复、噪声注入、量测漂移、平坦线、尖峰扰动等受损场景，并形成统一的数据清洗、标准化、切片、窗口构造和数据质控流程。通过场景化设计而非单一噪声注入，可以更真实地模拟现场采集、传输和设备状态变化对预测任务的影响。' },
    @{ Kind = 'Heading3'; Text = '2. 轻量稀疏混合专家预测模型设计' },
    @{ Kind = 'Body'; Text = '在模型层面，项目将基于论文现有Ultra-LSNT/PI-MoE思路，构建融合多尺度分解、轻量时序编码、Top-K路由和跳跃门控的预测框架。模型通过多专家分工处理不同类型的波动模式，对复杂时段分配更多计算、对平稳时段保持稀疏推理，从而在边缘部署场景中兼顾性能与效率。' },
    @{ Kind = 'Heading3'; Text = '3. 可行性引导训练机制与物理边界约束' },
    @{ Kind = 'Body'; Text = '针对风电功率预测中容易出现的负功率、超额定功率和切入切出区间不合理输出问题，项目拟构建训练期可行性引导机制，引入基于风机功率包络的惩罚项和部署一致性代理变量，使模型在训练阶段就学习符合物理规律的输出行为，而不是仅在推理后阶段进行简单裁剪。这样既能提升预测结果的可信度，也能减少后处理带来的信息损失。' },
    @{ Kind = 'Heading3'; Text = '4. 鲁棒性、边缘效率与多域迁移评估' },
    @{ Kind = 'Body'; Text = '除常规R²、RMSE、MAE等精度指标外，项目还将从鲁棒性和部署性两个方向开展系统评估：一方面测试模型在高斯噪声、分布漂移、量化压缩和结构化故障场景下的性能保持率；另一方面测量CPU/ARM条件下的单样本推理时延、内存占用、参数规模和能耗水平。与此同时，项目还将结合美国风电、空气质量与负荷数据开展多域迁移实验，观察模型在跨场景任务中的泛化能力。' },
    @{ Kind = 'Heading3'; Text = '5. 预测—调度耦合验证与成果凝练' },
    @{ Kind = 'Body'; Text = '在应用验证方面，项目将把预测结果接入共享调度筛选接口，比较不同模型对总运行成本、备用成本、弃风量和决策稳定性的影响，检验“预测更稳健”是否能够真实转化为“调度更经济”。在成果凝练方面，将同步完成图表整理、实验脚本复核、中文申请书撰写和后续论文修改，为项目执行和后续结项打下基础。' }
)

$schemeItems = @(
    @{ Kind = 'Heading2'; Text = '（一）技术路线' },
    @{ Kind = 'Body'; Text = '本项目总体采用“数据构建—模型设计—约束训练—对比评估—调度验证”的研究路线：首先基于风电SCADA数据构建干净集与受损集，并统一完成时间对齐、异常剔除、标准化和滑动窗口切片；其次实现DLinear、iTransformer、TimeMixer等代表性基线，并在此基础上构建稀疏MoE主模型与可行性引导模块；再次在精度、鲁棒性、超限率、推理时延与参数量等指标上开展系统对比；最后将预测结果接入共享调度接口，分析其经济价值和工程适配性。' },
    @{ Kind = 'Heading2'; Text = '（二）研究计划' },
    @{ Kind = 'Body'; Text = '项目执行拟分四个阶段推进：2026年3月至4月完成文献梳理、研究方案细化和数据场景整理；2026年5月至6月完成模型复现、核心模块编码和可行性约束实现；2026年7月至8月完成鲁棒性实验、消融实验、量化测试和边缘部署评估；2026年9月至10月完成图表整理、项目总结、申请材料完善和论文修改。每个阶段均以“可复现实验结果”作为阶段性交付物，避免只形成文字性总结。' },
    @{ Kind = 'Heading2'; Text = '（三）实施条件与可行性' },
    @{ Kind = 'Body'; Text = '申请人已完成相关英文论文手稿和主要实验，具备统计建模、Python实现、数据可视化和实验复核基础；指导教师长期从事统计学习与数据分析研究，可在研究设计、方法选择与成果写作方面提供稳定指导。现有工作已形成可直接延展的代码框架、图表资源和实验日志，因此项目具有较好的执行可行性与成果产出基础。' }
)

$priorItems = @(
    @{ Kind = 'Heading2'; Text = '（一）论文与代码基础' },
    @{ Kind = 'Body'; Text = '申请人已围绕本课题完成英文论文手稿《Wind power forecasting under corrupted SCADA for edge deployment: sparse MoE with feasibility guidance》，并完成数据预处理、受损场景构造、基线复现、鲁棒性评估、边缘部署测试和调度价值评估等核心实验。当前已经形成较完整的代码框架、实验日志、图表资源和参考文献整理结果，为本科科研创新基金项目的继续推进提供了较扎实的前期基础。' },
    @{ Kind = 'Heading2'; Text = '（二）已完成的阶段性实验结果' },
    @{ Kind = 'Body'; Text = '1. 基准实验方面，当前模型在Wind-CN主数据集上已经完成与DLinear、iTransformer、TimeMixer等方法的系统对比。现有结果表明，Ultra-LSNT在保证可部署性的前提下取得了R²=0.8728、RMSE=9080.55kW、MAE=7171.42kW的有效表现，轻量化版本Ultra-LSNT-Lite的单样本推理延迟可压缩至约4.5ms，说明模型具备“精度—效率”折中优化的潜力。' },
    @{ Kind = 'Body'; Text = '2. 鲁棒性评估方面，项目已完成多组高斯噪声扫描实验。与对比模型相比，Ultra-LSNT在噪声逐步增强条件下表现出更稳定的性能保持率：当噪声水平提升至σ=0.40时，模型仍可保持较高的R²水平，而DLinear等方法则出现明显性能塌缩。相关结果说明，面向受损SCADA场景进行模型设计具有现实必要性。' },
    @{ Kind = 'Body'; Text = '3. 调度价值评估方面，项目已构建预测—调度接口并完成初步滚动测试。现有结果显示，Ultra-LSNT对应的综合运行成本约为583,200元，相比DLinear的668,668元具有约12.8%的成本优势，同时在备用成本和弃风量方面也表现出更优的综合效果。这表明模型改进并不仅体现在统计指标上，也能够转化为更具工程意义的经济收益。' },
    @{ Kind = 'Body'; Text = '4. 跨域与可解释性方面，项目已完成部分风电、空气质量与负荷数据迁移实验，初步验证了模型在不同数据域中的泛化潜力；同时已经生成专家激活热力图、鲁棒性测试曲线和调度价值可视化结果，可直接作为后续申请书、阶段报告和论文修改中的支撑材料。' },
    @{ Kind = 'Heading2'; Text = '（三）已形成的图表与支撑材料' },
    @{ Kind = 'Body'; Text = '目前已沉淀的成果包括：技术流程图、鲁棒性测试图、调度价值评估图、主要结果汇总表、文献清单和实验总结文档。这些材料不仅能够支撑本次申请书完善，也为项目后续中期检查、结项汇报和论文投稿提供了完整的图文基础。' }
)

$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($workingPath)

    $frontTable = $doc.Tables.Item(1)
    $oldArgumentTable = $doc.Tables.Item(2)
    $argumentTableStart = $oldArgumentTable.Range.Start
    $firstColWidth = $oldArgumentTable.Columns.Item(1).Width
    $secondColWidth = $oldArgumentTable.Columns.Item(2).Width
    $oldArgumentTable.Delete()
    $argumentTable = $doc.Tables.Add($doc.Range($argumentTableStart, $argumentTableStart), 4, 2)
    $argumentTable.Borders.Enable = 1
    $argumentTable.Rows.AllowBreakAcrossPages = $true
    $argumentTable.Columns.Item(1).Width = $firstColWidth
    $argumentTable.Columns.Item(2).Width = $secondColWidth
    $argumentTable.Cell(1, 1).Range.Text = '选题依据'
    $argumentTable.Cell(2, 1).Range.Text = '研究内容'
    $argumentTable.Cell(3, 1).Range.Text = '研究方案'
    $argumentTable.Cell(4, 1).Range.Text = '前期成果'

    Format-FrontTable -Word $word -Table $frontTable
    Format-ArgumentTable -Word $word -Table $argumentTable

    $selection = $word.Selection

    Write-ItemsToCell -Word $word -Selection $selection -Cell $argumentTable.Cell(1, 2) -Items $basisItems

    Write-ItemsToCell -Word $word -Selection $selection -Cell $argumentTable.Cell(2, 2) -Items $researchItems

    Write-ItemsToCell -Word $word -Selection $selection -Cell $argumentTable.Cell(3, 2) -Items $schemeItems
    Insert-CenteredPicture -Word $word -Selection $selection -ImagePath $workflowImage -Caption '图1 数据构建—模型设计—鲁棒评估—调度验证的技术路线图' -WidthCm 11.6

    Write-ItemsToCell -Word $word -Selection $selection -Cell $argumentTable.Cell(4, 2) -Items $priorItems
    Insert-CenteredPicture -Word $word -Selection $selection -ImagePath $robustnessImage -Caption '图2 已完成的鲁棒性测试结果' -WidthCm 11.4
    Insert-CenteredPicture -Word $word -Selection $selection -ImagePath $dispatchImage -Caption '图3 已完成的调度价值评估结果' -WidthCm 11.4

    $doc.Save()
    $doc.Close()
    $word.Quit()

    try {
        Copy-Item -Path $workingPath -Destination $docPath -Force
    } catch {
    }
} finally {
    if ($doc -ne $null) {
        try {
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)
        } catch {
        }
    }
    if ($word -ne $null) {
        try {
            [void][System.Runtime.InteropServices.Marshal]::ReleaseComObject($word)
        } catch {
        }
    }
    [GC]::Collect()
    [GC]::WaitForPendingFinalizers()
}







