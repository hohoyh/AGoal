# ============================================================
# 下载 AGoal 所需的模型权重与第三方源码 (Windows / PowerShell)
# 用法:  powershell -ExecutionPolicy Bypass -File scripts\download_models.ps1
# 可选:  ...\download_models.ps1 -WithSamH   # 额外下载 SAM ViT-H (2.5GB)
# ============================================================
param(
    [switch]$WithSamH
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $Root

$ModelDir = "data\models"
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null

$ProgressPreference = "Continue"

function Get-Model {
    param([string]$Url, [string]$Dest)

    if (Test-Path $Dest) {
        Write-Host "[skip] $(Split-Path -Leaf $Dest) 已存在" -ForegroundColor Yellow
        return
    }
    Write-Host "[down] $(Split-Path -Leaf $Dest) <- $Url" -ForegroundColor Cyan
    # 先下载到临时文件，避免中断后留下半截权重
    $tmp = "$Dest.part"
    Invoke-WebRequest -Uri $Url -OutFile $tmp -UseBasicParsing
    Move-Item -Force $tmp $Dest
}

# ---------- 1. 检测与分割模型 ----------
Get-Model "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" `
          "$ModelDir\groundingdino_swint_ogc.pth"

Get-Model "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" `
          "$ModelDir\sam_vit_b_01ec64.pth"

if ($WithSamH) {
    Get-Model "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" `
              "$ModelDir\sam_vit_h_4b8939.pth"
}

Get-Model "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" `
          "$ModelDir\yolov8n.pt"

# ---------- 2. Grounded-Segment-Anything ----------
$gsa = "third_party\Grounded-Segment-Anything"
if (-not (Test-Path $gsa)) {
    Write-Host "[clone] Grounded-Segment-Anything" -ForegroundColor Cyan
    New-Item -ItemType Directory -Force -Path "third_party" | Out-Null
    git clone --depth 1 https://github.com/IDEA-Research/Grounded-Segment-Anything.git $gsa
} else {
    Write-Host "[skip] third_party\Grounded-Segment-Anything 已存在" -ForegroundColor Yellow
}

Write-Host @"

------------------------------------------------------------
下载完成。接下来还需要两步手工安装：

1) 编译安装 GroundingDINO（需要 CUDA 环境 + MSVC）
     cd third_party\Grounded-Segment-Anything\GroundingDINO
     pip install -e .

2) 安装 SAM
     cd third_party\Grounded-Segment-Anything\segment_anything
     pip install -e .

CLIP 权重由 transformers 在首次运行时自动下载
（openai/clip-vit-base-patch32）。
------------------------------------------------------------
"@ -ForegroundColor Green
