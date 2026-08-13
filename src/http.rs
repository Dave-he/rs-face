use crate::image::BoxError;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};

pub fn download_to_file(url: &str, tmp_dir: &Path) -> Result<PathBuf, BoxError> {
    let (host, port, path) = parse_url(url)?;
    let filename = extract_filename(url);
    let file_path = tmp_dir.join(filename);
    {
        let addr = format!("{}:{}", host, port);
        let mut stream = TcpStream::connect(&addr).map_err(|e| format!("HTTP 连接 {} 失败: {}", addr, e))?;
        stream.set_read_timeout(Some(std::time::Duration::from_secs(600)))?;
        let req = format!(
            "GET {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\nUser-Agent: rs-face/0.1\r\nAccept: */*\r\n\r\n",
            path, host
        );
        stream.write_all(req.as_bytes())?;
        stream.flush()?;
        let mut resp = Vec::new();
        let mut tmp = [0u8; 8192];
        loop {
            match stream.read(&mut tmp) {
                Ok(0) => break,
                Ok(n) => resp.extend_from_slice(&tmp[..n]),
                Err(ref e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
        let header_end = find_header_end(&resp).ok_or("HTTP 响应不完整")?;
        let status = parse_status(&resp[..header_end])?;
        if status != 200 && status != 206 {
            return Err(format!("HTTP 状态码: {}", status).into());
        }
        let body = if let Some(cl) = parse_content_length(&resp[..header_end]) {
            let start = header_end + 4;
            let end = (start + cl).min(resp.len());
            resp[start..end].to_vec()
        } else {
            let start = header_end + 4;
            resp[start..].to_vec()
        };
        let mut f = std::fs::File::create(&file_path)?;
        f.write_all(&body)?;
        f.flush()?;
    }
    Ok(file_path)
}

fn parse_url(url: &str) -> Result<(String, u16, String), BoxError> {
    let rest = url.strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or("仅支持 http/https URL")?;
    let is_https = url.starts_with("https://");
    let (host_port, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let (host, port) = match host_port.find(':') {
        Some(i) => (
            host_port[..i].to_string(),
            host_port[i + 1..].parse::<u16>().unwrap_or(if is_https { 443 } else { 80 }),
        ),
        None => (host_port.to_string(), if is_https { 443 } else { 80 }),
    };
    if is_https {
        return Err("HTTPS 下载需要 openssl 或 rustls，为维持零依赖请用 http:// URL，或先用 curl/wget 下载后 --input 指定本地文件".into());
    }
    Ok((host, port, path.to_string()))
}

fn extract_filename(url: &str) -> String {
    let trimmed = url.trim_end_matches('/');
    match trimmed.rfind('/') {
        Some(i) => {
            let name = &trimmed[i + 1..];
            if name.is_empty() { "download".to_string() }
            else {
                name.split('?').next().unwrap_or("download").split('&').next().unwrap_or("download").to_string()
            }
        }
        None => "download".to_string(),
    }
}

fn find_header_end(data: &[u8]) -> Option<usize> {
    for i in 0..data.len().saturating_sub(4) {
        if &data[i..i + 4] == b"\r\n\r\n" {
            return Some(i);
        }
    }
    None
}

fn parse_status(header: &[u8]) -> Result<u32, BoxError> {
    let line = header.split(|&b| b == b'\r' || b == b'\n').next().unwrap_or(b"");
    let parts: Vec<&[u8]> = line.split(|&b| b == b' ').collect();
    if parts.len() >= 2 {
        let s = std::str::from_utf8(parts[1]).unwrap_or("");
        return Ok(s.parse::<u32>().unwrap_or(0));
    }
    Ok(0)
}

fn parse_content_length(header: &[u8]) -> Option<usize> {
    for line in header.split(|&b| b == b'\n') {
        let line = line.strip_suffix(b"\r").unwrap_or(line);
        if let Ok(s) = std::str::from_utf8(line) {
            let lower = s.to_lowercase();
            if let Some(rest) = lower.strip_prefix("content-length:") {
                return rest.trim().parse::<usize>().ok();
            }
        }
    }
    None
}
