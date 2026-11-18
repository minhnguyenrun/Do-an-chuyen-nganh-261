Add-Type -AssemblyName System.Drawing
$out = Join-Path $PSScriptRoot 'method2_architecture'
New-Item -ItemType Directory -Force -Path $out | Out-Null
$bmp = New-Object System.Drawing.Bitmap 2200,1480
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = 'AntiAlias'
$g.TextRenderingHint = 'AntiAliasGridFit'
$g.Clear([System.Drawing.Color]::White)
$svg = [System.Collections.Generic.List[string]]::new()
$svg.Add('<svg xmlns="http://www.w3.org/2000/svg" width="2200" height="1480" viewBox="0 0 2200 1480"><rect width="2200" height="1480" fill="white"/>')
function Text($x,$y,$s,$size=24,$bold=$false,$color='#243247') {
 $style = if($bold){[System.Drawing.FontStyle]::Bold}else{[System.Drawing.FontStyle]::Regular}
 $font = [System.Drawing.Font]::new('Arial',$size,[System.Drawing.FontStyle]$style,[System.Drawing.GraphicsUnit]::Pixel)
 $brush = [System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($color))
 $g.DrawString($s,$font,$brush,$x,$y)
 $escaped = [System.Security.SecurityElement]::Escape($s)
 $weight = if($bold){'bold'}else{'normal'}
 $svg.Add("<text x='$x' y='$($y+$size)' font-family='Arial,sans-serif' font-size='$size' font-weight='$weight' fill='$color'>$escaped</text>")
 $font.Dispose(); $brush.Dispose()
}
function Box($x,$y,$w,$h,$title,$lines,$fill='#F2F5F9',$stroke='#9AA7B7') {
 $brush=[System.Drawing.SolidBrush]::new([System.Drawing.ColorTranslator]::FromHtml($fill))
 $pen=[System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml($stroke),2)
 $g.FillRectangle($brush,$x,$y,$w,$h); $g.DrawRectangle($pen,$x,$y,$w,$h)
 $svg.Add("<rect x='$x' y='$y' width='$w' height='$h' fill='$fill' stroke='$stroke' stroke-width='2'/>")
 Text ($x+20) ($y+15) $title 27 $true
 $i=0; foreach($line in $lines){Text ($x+20) ($y+57+31*$i) $line 23; $i++}
 $brush.Dispose(); $pen.Dispose()
}
function Arrow($x1,$y1,$x2,$y2,$dash=$false) {
 $pen=[System.Drawing.Pen]::new([System.Drawing.ColorTranslator]::FromHtml('#536478'),2.5)
 if($dash){$pen.DashStyle='Dash'}
 $pen.CustomEndCap=[System.Drawing.Drawing2D.AdjustableArrowCap]::new(5,6)
 $g.DrawLine($pen,$x1,$y1,$x2,$y2)
 $dx=$x2-$x1; $dy=$y2-$y1; $len=[Math]::Sqrt($dx*$dx+$dy*$dy); $ux=$dx/$len; $uy=$dy/$len
 $ax=$x2-14*$ux; $ay=$y2-14*$uy
 $dashattr=if($dash){"stroke-dasharray='9 6'"}else{''}
 $svg.Add("<line x1='$x1' y1='$y1' x2='$x2' y2='$y2' stroke='#536478' stroke-width='2.5' $dashattr/><polygon points='$x2,$y2 $($ax-6*$uy),$($ay+6*$ux) $($ax+6*$uy),$($ay-6*$ux)' fill='#536478'/>")
 $pen.Dispose()
}
Text 65 30 'METHOD 2  |  Cascaded Document Layout Analysis' 39 $true
Text 65 86 'Current implementation: v2.1  /  Two independent variants with a shared downstream architecture' 25
Box 65 155 310 180 'Input document page' @('Original page pixels','Width W, height H') '#F6F7F9'
Box 465 155 500 180 '(a) Pipeline S' @('PP-DocLayout-S','Region boxes, labels, detector scores') '#EAF1FA' '#7595BC'
Box 1065 155 500 180 '(b) Pipeline V3' @('PP-DocLayoutV3','Region boxes, labels, detector scores') '#EAF1FA' '#7595BC'
Arrow 375 205 465 205
Arrow 375 295 420 295
Arrow 420 295 420 125
Arrow 420 125 1315 125
Arrow 1315 125 1315 155
Text 1640 174 'VARIANT SELECTION' 24 $true
Text 1640 217 'Run S or V3 independently.' 23
Text 1640 251 'S and V3 outputs are not fused.' 23
Box 65 405 530 190 'OCR evidence' @('PP-OCRv5 mobile_det + mobile_rec','Line polygons, transcripts, rec_score','Trusted recognition: rec_score >= 0.75') '#EDF6F3' '#83A99D'
Box 65 665 530 190 'Auxiliary layout detector' @('DocLayout-YOLO / DocStructBench','Full page, imgsz = 1024, score >= 0.10','Small boxes: area <= 2%, height <= 8%') '#F6F1E8' '#B39C77'
Arrow 220 335 220 405
Arrow 65 335 35 335
Arrow 35 335 35 760
Arrow 35 760 65 760
Box 710 405 850 190 '1  |  Baseline-preserving fusion' @('PP baseline: detector score >= 0.45','Normalize labels; test overlap and parent-child relations','Add verified YOLO regions; retain PP conflict fallback') '#EAF1FA' '#7595BC'
Arrow 715 335 935 405
Arrow 1315 335 1315 405
Arrow 595 490 710 490 $true
Arrow 595 760 650 760
Arrow 650 760 650 555
Arrow 650 555 710 555
Box 710 665 850 190 '2  |  OCR-guided recovery and target gating' @('Recover weak proposals; extend compatible Text/List regions','Line polygon coverage >= 0.80; semantic score >= 0.60','Targets: true conflicts and unresolved OCR line groups') '#EDF6F3' '#83A99D'
Arrow 1135 595 1135 665
Arrow 595 570 620 570 $true
Arrow 620 570 620 695 $true
Arrow 620 695 710 695 $true
Box 1640 665 500 190 '3  |  Selective VLM inference' @('MinerU2.5-2509-1.2B','Official layout_detect on context crops','<= 8 crops/page; edge <= 768 px','Batch = 1; <= 512 tokens/crop') '#F2EDF8' '#A38BB8'
Arrow 1560 760 1640 760
Box 710 940 1430 190 '4  |  OCR-line attribution and constrained selection (v2.1)' @('Map crop boxes to page coordinates; assign target lines; protect neighboring regions','Reject foreign-line contamination; select a valid partition; preserve fallback lines','Q = 0.45 Coverage - 0.30 SplitPenalty + 0.25 Semantic; require Q(new) > Q(KEEP) + 0.10') '#F2EDF8' '#A38BB8'
Arrow 1890 855 1890 940
Arrow 1135 855 1135 940 $true
Text 1160 879 'KEEP candidate + fixed OCR ownership' 22
Box 710 1205 650 145 'Accepted update' @('ADD / REPLACE / ABSORB','Final regions: bbox, label, lineage') '#EDF6F3' '#83A99D'
Box 1450 1205 690 145 'Fallback / review' @('KEEP baseline on rejection, error or budget limit','Record proposals, decisions and review evidence') '#F6F1E8' '#B39C77'
Arrow 1035 1130 1035 1205
Arrow 1795 1130 1795 1205
Text 65 970 'OUTPUT-LEVEL CASCADE' 24 $true
Text 65 1015 'Independent pretrained models.' 23
Text 65 1050 'Rule-based geometric and' 23
Text 65 1085 'OCR-semantic verification.' 23
Text 65 1140 'Solid: prediction / control flow' 22
Text 65 1175 'Dashed: evidence / fallback flow' 22
Text 65 1390 'Execution schedule: Paddle (layout + OCR) -> release -> YOLO -> release -> selective MinerU + final decision' 25 $true
Text 65 1430 'The shared stages are instantiated separately for S and V3. WBF is disabled by default. Layout-only output; no shared-backbone training.' 22
$svg.Add('</svg>')
[System.IO.File]::WriteAllLines((Join-Path $out 'method2_cascaded_architecture.svg'),$svg,[System.Text.UTF8Encoding]::new($false))
$bmp.Save((Join-Path $out 'method2_cascaded_architecture.png'),[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
