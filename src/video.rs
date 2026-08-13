use crate::image::BoxError;
use std::path::Path;
use std::process::Command;

pub fn extract_frames_pgm<P: AsRef<Path>, Q: AsRef<Path>>(
    video: P,
    out_dir: Q,
    fps: f64,
) -> Result<(), BoxError> {
    let out_dir = out_dir.as_ref();
    std::fs::create_dir_all(out_dir)?;
    let pattern = out_dir.join("frame_%06d.pgm");
    let pattern_str = pattern.to_string_lossy().into_owned();
    let vf = format!("fps={:.6}", fps);
    let status = Command::new("ffmpeg")
        .arg("-nostdin")
        .arg("-y")
        .arg("-i").arg(video.as_ref())
        .arg("-vf").arg(&vf)
        .arg("-pix_fmt").arg("gray")
        .arg("-compression_level").arg("0")
        .arg("-start_number").arg("1")
        .arg(&pattern_str)
        .status()
        .map_err(|e| format!("无法执行 ffmpeg: {}. 请确保系统已安装 ffmpeg 并在 PATH 中", e))?;
    if !status.success() {
        return Err(format!("ffmpeg 退出码: {:?}", status.code()).into());
    }
    Ok(())
}

pub fn extract_frames_pgm_with_timestamps<P: AsRef<Path>, Q: AsRef<Path>>(
    video: P,
    out_dir: Q,
    fps: f64,
) -> Result<(), BoxError> {
    extract_frames_pgm(video, out_dir, fps)
}
