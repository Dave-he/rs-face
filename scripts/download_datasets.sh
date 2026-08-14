#!/usr/bin/env bash
# 下载 + 解压 + 标准化公开人脸数据集, 供 rs-face benchmark 使用。
# 数据集目录约定: <root>/<人物>/<图片>.<ext>
# 零 Rust 依赖: 仅依赖 curl / tar / gunzip / 可选 ImageMagick (convert)。
set -euo pipefail

DATASETS_ROOT="${DATASETS_ROOT:-./datasets}"
mkdir -p "$DATASETS_ROOT"

# ---------- AT&T / ORL Face Database ----------
# 40 subjects × 10 images = 400 PGM (92×112), 4.5MB
# License: research use, courtesy AT&T Laboratories Cambridge
ORL_URL="http://www.cl.cam.ac.uk/research/dtg/attarchive/pub/data/att_faces.zip"

download_orl() {
    local dst="$DATASETS_ROOT/att_faces"
    if [[ -d "$dst/s1" ]]; then
        echo "[orl] 已存在: $dst"
        return 0
    fi
    local tmp="$(mktemp -d)"
    echo "[orl] 下载: $ORL_URL"
    # Cambridge 服务器偶发断流, 加 --retry 5 + 断点续传 (-C -)
    curl -4 -L -C - --fail --retry 5 --retry-delay 3 --max-time 600 -o "$tmp/orl.zip" "$ORL_URL"
    echo "[orl] 解压..."
    (cd "$tmp" && unzip -q orl.zip)
    if [[ ! -d "$tmp/att_faces/s1" ]]; then
        echo "[orl] 错误: 解压后未找到 att_faces/s1" >&2
        return 1
    fi
    mv "$tmp/att_faces" "$dst"
    rm -rf "$tmp"
    echo "[orl] 完成: $dst ($(find "$dst" -name '*.pgm' | wc -l | tr -d ' ') 张 PGM)"
}

# ---------- Yale Face Database A ----------
# 15 subjects × 11 images = 165 GIF (320×243), 6.4MB
# License: research use, courtesy Yale Vision Lab
YALE_URL="http://vision.ucsd.edu/datasets/yale_face_dataset_original/yalefaces.zip"

download_yale() {
    local dst="$DATASETS_ROOT/yalefaces"
    if [[ -d "$dst/subject01" || -d "$dst/subject01.centerlight" ]]; then
        echo "[yale] 已存在: $dst"
        convert_yale_gif_to_pgm "$dst"
        return 0
    fi
    local tmp="$(mktemp -d)"
    echo "[yale] 下载: $YALE_URL"
    curl -4 -L -C - --fail --retry 5 --retry-delay 3 --max-time 300 -o "$tmp/yale.zip" "$YALE_URL"
    echo "[yale] 解压..."
    (cd "$tmp" && unzip -q yale.zip)
    local src
    src="$(find "$tmp" -type d -name 'yalefaces' | head -n 1)"
    if [[ -z "$src" ]]; then
        src="$(find "$tmp" -maxdepth 2 -name 'subject*' -printf '%h\n' | sort -u | head -n 1)"
        src="${src:-$(dirname "$(find "$tmp" -maxdepth 3 -name 'subject01.gif' | head -n 1)")}"
    fi
    if [[ -z "$src" || ! -d "$src" ]]; then
        echo "[yale] 错误: 找不到 yalefaces 目录" >&2
        return 1
    fi
    mkdir -p "$dst"
    # subject01.gif / subject01.centerlight.gif / ...
    cp -R "$src"/. "$dst"/
    rm -rf "$tmp"
    convert_yale_gif_to_pgm "$dst"
    echo "[yale] 完成: $dst ($(find "$dst" -name '*.pgm' | wc -l | tr -d ' ') 张 PGM)"
}

# Yale 解压后是 .gif, 训练需要 PGM/PPM/PNG。优先用 ImageMagick, 其次 macOS sips, 最后 ffmpeg。
detect_converter() {
    if command -v magick >/dev/null 2>&1; then echo "magick"
    elif command -v convert >/dev/null 2>&1; then echo "convert"
    elif command -v sips >/dev/null 2>&1; then echo "sips"
    elif command -v ffmpeg >/dev/null 2>&1; then echo "ffmpeg"
    else echo ""
    fi
}

convert_yale_gif_to_pgm() {
    local dir="$1"
    local converter
    converter="$(detect_converter)"
    if [[ -z "$converter" ]]; then
        echo "[yale] 未检测到任何 GIF→PPM 转换器 (magick/convert/sips/ffmpeg), GIF 无法用于训练" >&2
        return 0
    fi
    local gif_count
    gif_count="$(find "$dir" -name '*.gif' | wc -l | tr -d ' ')"
    if [[ "$gif_count" == "0" ]]; then
        return 0
    fi
    echo "[yale] GIF → PGM via $converter ($gif_count 张)..."
    find "$dir" -maxdepth 1 -name '*.gif' -print0 | while IFS= read -r -d '' g; do
        local base="${g%.gif}"
        if [[ -f "${base}.pgm" ]]; then continue; fi
        case "$converter" in
            magick)  magick "$g" -colorspace Gray "${base}.pgm" >/dev/null 2>&1 ;;
            convert) convert "$g" -colorspace Gray "${base}.pgm" >/dev/null 2>&1 ;;
            sips)    sips -s format pgm "$g" --out "${base}.pgm" >/dev/null 2>&1 ;;
            ffmpeg)  ffmpeg -y -loglevel error -i "$g" -pix_fmt gray -vcodec pgm "${base}.pgm" >/dev/null 2>&1 ;;
        esac
    done
}

# ---------- LFW (Labeled Faces in the Wild) ----------
# 13233 images (250x250 JPG, deep-funneled), 5749 identities, ~250MB
# 标准人脸验证协议: pairs.txt 6000 对 (3000 同人, 3000 不同人)
# 默认不下载 (体积大); 通过环境变量 RS_FACE_DOWNLOAD_LFW=1 触发。
LFW_IMG_URL="http://vis-www.cs.umass.edu/lfw/lfw-deepfunneled.tgz"
LFW_PAIRS_URL="http://vis-www.cs.umass.edu/lfw/pairs.txt"

download_lfw() {
    local dst="$DATASETS_ROOT/lfw"
    if [[ -d "$dst" ]]; then
        echo "[lfw] 已存在: $dst"
        return 0
    fi
    if [[ "${RS_FACE_DOWNLOAD_LFW:-0}" != "1" ]]; then
        echo "[lfw] 跳过 (设置 RS_FACE_DOWNLOAD_LFW=1 启用, ~250MB)"
        return 0
    fi
    local tmp="$(mktemp -d)"
    echo "[lfw] 下载图片: $LFW_IMG_URL (慢, ~250MB)"
    curl -4 -L --fail --retry 5 --retry-delay 5 --max-time 1800 -o "$tmp/lfw.tgz" "$LFW_IMG_URL"
    echo "[lfw] 下载 pairs: $LFW_PAIRS_URL"
    curl -4 -L --fail --retry 3 --retry-delay 2 --max-time 60 -o "$tmp/pairs.txt" "$LFW_PAIRS_URL"
    echo "[lfw] 解压..."
    tar -xzf "$tmp/lfw.tgz" -C "$tmp"
    if [[ ! -d "$tmp/lfw" ]]; then
        echo "[lfw] 错误: 解压后未找到 lfw 目录" >&2
        return 1
    fi
    mv "$tmp/lfw" "$dst"
    cp "$tmp/pairs.txt" "$dst/pairs.txt"
    rm -rf "$tmp"
    echo "[lfw] 完成: $dst"
}

usage() {
    cat <<EOF
用法: $0 [orl|yale|lfw|all]
  orl   下载 AT&T/ORL (4.5MB, PGM, 主推)
  yale  下载 Yale Face Database A (6.4MB, GIF)
  lfw   下载 LFW (250MB, 可选, RS_FACE_DOWNLOAD_LFW=1)
  all   下载 orl + yale (+ lfw 若 RS_FACE_DOWNLOAD_LFW=1)
  默认: all

环境变量:
  DATASETS_ROOT           数据集根目录 [默认: ./datasets]
  RS_FACE_DOWNLOAD_LFW    设为 1 启用 LFW 下载

示例:
  ./scripts/download_datasets.sh                # 下载 ORL + Yale
  RS_FACE_DOWNLOAD_LFW=1 ./scripts/download_datasets.sh lfw
EOF
}

main() {
    local target="${1:-all}"
    case "$target" in
        orl)  download_orl ;;
        yale) download_yale ;;
        lfw)  download_lfw ;;
        all)  download_orl; download_yale; download_lfw ;;
        -h|--help|help) usage ;;
        *) echo "未知目标: $target" >&2; usage; exit 1 ;;
    esac
    echo
    echo "数据集根目录: $DATASETS_ROOT"
    echo "已下载:"
    for d in "$DATASETS_ROOT"/*/; do
        [[ -d "$d" ]] || continue
        local count="$(find "$d" -maxdepth 4 \( -name '*.pgm' -o -name '*.ppm' -o -name '*.jpg' -o -name '*.png' \) | wc -l | tr -d ' ')"
        echo "  $(basename "$d")  ($count 张图像)"
    done
}

main "$@"