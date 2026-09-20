$ErrorActionPreference = 'Stop'
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$docPath = Join-Path $PSScriptRoot '安徽财经大学本科生科研创新基金项目申请书.docx'
$workingPath = Join-Path $PSScriptRoot '安徽财经大学本科生科研创新基金项目申请书_完善版_20260311.docx'

function Replace-AllText {
    param(
        [object]$Range,
        [string]$FindText,
        [string]$ReplaceWith
    )

    $find = $Range.Find
    $find.ClearFormatting()
    $find.Replacement.ClearFormatting()
    [void]$find.Execute($FindText, $false, $false, $false, $false, $false, $true, 1, $false, $ReplaceWith, 2)
}

$word = $null
$doc = $null

try {
    $word = New-Object -ComObject Word.Application
    $word.Visible = $false
    $word.DisplayAlerts = 0
    $doc = $word.Documents.Open($docPath)

    if ($doc.Revisions.Count -gt 0) {
        $doc.Revisions.AcceptAll()
    }

    while ($doc.Comments.Count -gt 0) {
        $doc.Comments.Item(1).Delete()
    }

    $doc.TrackRevisions = $false
    $doc.ShowRevisions = $false

    Replace-AllText -Range $doc.Content -FindText '成果产出基础。图1 数据构建—模型设计—鲁棒评估—调度验证的技术路线图' -ReplaceWith '成果产出基础。^p图1 数据构建—模型设计—鲁棒评估—调度验证的技术路线图'
    Replace-AllText -Range $doc.Content -FindText '图文基础。图2 已完成的鲁棒性测试结果' -ReplaceWith '图文基础。^p图2 已完成的鲁棒性测试结果'

    foreach ($paragraph in $doc.Paragraphs) {
        $text = ($paragraph.Range.Text -replace "`r", '' -replace "`a", '' -replace "`v", '').Trim()
        if ($text -match '^图[123]\s') {
            $paragraph.Range.Font.NameFarEast = '宋体'
            $paragraph.Range.Font.Name = 'Times New Roman'
            $paragraph.Range.Font.Size = 10.5
            $paragraph.Range.ParagraphFormat.Alignment = 1
            $paragraph.Range.ParagraphFormat.FirstLineIndent = 0
            $paragraph.Range.ParagraphFormat.SpaceBefore = 3
            $paragraph.Range.ParagraphFormat.SpaceAfter = 3
            try {
                $paragraph.Range.ParagraphFormat.CharacterUnitFirstLineIndent = 0
            } catch {
            }
        }
    }

    $doc.Save()
    $doc.Close()
    $word.Quit()

    Copy-Item -Path $docPath -Destination $workingPath -Force
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


