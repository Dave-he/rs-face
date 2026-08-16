FROM debian:bookworm-slim

# ffmpeg 是 rs-face 唯一的外部依赖 (用于抽帧)
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY target/release/rs-face /usr/local/bin/rs-face
COPY data /app/data

# rs-face 默认 cascade 在 /app/data/
ENV RS_FACE_DATA_DIR=/app/data

# 用法:
#   docker run --rm -v $PWD:/data rs-face detect \
#     --input /data/lecture.mp4 --output /data/out

ENTRYPOINT ["rs-face"]
CMD ["--help"]