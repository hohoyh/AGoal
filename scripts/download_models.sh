#!/usr/bin/env bash
# ============================================================
# 下载 AGoal 所需的模型权重与第三方源码
# 用法:  bash scripts/download_models.sh
# 可选:  bash scripts/download_models.sh --with-sam-h   # 额外下载 SAM ViT-H (2.5GB)
# ============================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL_DIR="data/models"
THIRD_PARTY="third_party"
mkdir -p "$MODEL_DIR"

download() {
    local url="$1" dest="$2"
    if [ -f "$dest" ]; then
        echo "[skip] $(basename "$dest") 已存在"
        return
    fi
    echo "[down] $(basename "$dest") <- $url"
    if command -v wget >/dev/null 2>&1; then
        wget -q --show-progress -O "$dest" "$url"
    else
        curl -L --progress-bar -o "$dest" "$url"
    fi
}

# ---------- 1. 检测与分割模型 ----------
download "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth" \
         "$MODEL_DIR/groundingdino_swint_ogc.pth"

download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
         "$MODEL_DIR/sam_vit_b_01ec64.pth"

if [[ "${1:-}" == "--with-sam-h" ]]; then
    download "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth" \
             "$MODEL_DIR/sam_vit_h_4b8939.pth"
fi

# YOLOv8n（文本目标链路用，首次运行 ultralytics 也会自动下载）
download "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt" \
         "$MODEL_DIR/yolov8n.pt"

# ---------- 2. Grounded-Segment-Anything（GroundingDINO + SAM 源码）----------
if [ ! -d "$THIRD_PARTY/Grounded-Segment-Anything" ]; then
    echo "[clone] Grounded-Segment-Anything"
    mkdir -p "$THIRD_PARTY"
    git clone --depth 1 https://github.com/IDEA-Research/Grounded-Segment-Anything.git \
        "$THIRD_PARTY/Grounded-Segment-Anything"
else
    echo "[skip] third_party/Grounded-Segment-Anything 已存在"
fi

cat <<'EOF'

------------------------------------------------------------
下载完成。接下来还需要两步手工安装：

1) 编译安装 GroundingDINO（需要 CUDA 环境）
     cd third_party/Grounded-Segment-Anything/GroundingDINO
     pip install -e .

2) 安装 SAM
     cd third_party/Grounded-Segment-Anything/segment_anything
     pip install -e .

   然后把 third_party/Grounded-Segment-Anything 加入 PYTHONPATH，
   或直接使用仓库根目录下的入口脚本（它们已自动 sys.path.append）。

CLIP 权重由 transformers 在首次运行时自动下载
（openai/clip-vit-base-patch32）。
------------------------------------------------------------
EOF
