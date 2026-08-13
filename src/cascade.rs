use crate::image::{BoxError, Rect};
use crate::imgproc::IntegralImage;

#[derive(Debug, Clone, Copy)]
pub struct HaarRect {
    pub x: i32,
    pub y: i32,
    pub w: i32,
    pub h: i32,
    pub weight: f32,
}

#[derive(Debug, Clone)]
pub struct HaarFeature {
    pub rects: [HaarRect; 3],
    pub num_rects: usize,
    pub tilted: bool,
}

impl HaarFeature {
    pub fn eval(&self, ii: &IntegralImage, x: i32, y: i32, scale: f32) -> f64 {
        let mut sum = 0.0f64;
        if self.tilted {
            for i in 0..self.num_rects {
                let r = self.rects[i];
                let rx = x + (r.x as f32 * scale) as i32;
                let ry = y + (r.y as f32 * scale) as i32;
                let rw = (r.w as f32 * scale) as i32;
                let rh = (r.h as f32 * scale) as i32;
                let area: i64 = ii.width as i64 + 1;
                let x1 = rx as i64;
                let y1 = ry as i64;
                let idx1 = (y1 * area) as usize + (x1 + 1) as usize;
                let idx2 = ((y1 + rh as i64) * area) as usize + (x1 + rh as i64 + 1) as usize;
                let idx3 = ((y1 - rw as i64) * area) as usize + (x1 + rw as i64 + 1) as usize;
                let idx4 = ((y1 + rh as i64 - rw as i64) * area) as usize + (x1 + rw as i64 + rh as i64 + 1) as usize;
                let rect_sum = if idx4 < ii.data.len() && idx3 < ii.data.len() && idx2 < ii.data.len() && idx1 < ii.data.len() {
                    ii.data[idx4] - ii.data[idx3] - ii.data[idx2] + ii.data[idx1]
                } else {
                    0
                };
                sum += rect_sum as f64 * r.weight as f64;
            }
        } else {
            for i in 0..self.num_rects {
                let r = self.rects[i];
                let rx = x + (r.x as f32 * scale) as i32;
                let ry = y + (r.y as f32 * scale) as i32;
                let rw = (r.w as f32 * scale) as i32;
                let rh = (r.h as f32 * scale) as i32;
                let rect_sum = ii.sum(rx, ry, rw, rh);
                sum += rect_sum as f64 * r.weight as f64;
            }
        }
        sum
    }
}

#[derive(Debug, Clone)]
pub struct WeakClassifier {
    pub feature_idx: usize,
    pub threshold: f32,
    pub left_val: f32,
    pub right_val: f32,
}

impl WeakClassifier {
    pub fn predict(&self, value: f64, std_dev_norm: f64) -> f64 {
        let t = self.threshold as f64 * std_dev_norm;
        if value < t {
            self.left_val as f64
        } else {
            self.right_val as f64
        }
    }
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub threshold: f32,
    pub classifiers: Vec<WeakClassifier>,
}

impl Stage {
    pub fn pass(&self, ii: &IntegralImage, features: &[HaarFeature], x: i32, y: i32, scale: f32, std_dev_norm: f64) -> Option<f64> {
        let mut sum = 0.0f64;
        for wc in &self.classifiers {
            let val = features[wc.feature_idx].eval(ii, x, y, scale);
            sum += wc.predict(val, std_dev_norm);
        }
        if sum >= self.threshold as f64 {
            Some(sum)
        } else {
            None
        }
    }
}

#[derive(Debug, Clone)]
pub struct Cascade {
    pub window: (i32, i32),
    pub features: Vec<HaarFeature>,
    pub stages: Vec<Stage>,
}

impl Cascade {
    pub fn detect(
        &self,
        ii: &IntegralImage,
        min_size: u32,
        max_size: u32,
        scale_factor: f32,
        step: u32,
    ) -> Vec<(Rect, f32)> {
        let mut results = Vec::new();
        let (win_w, win_h) = self.window;
        let mut scale = 1.0f32;
        loop {
            let w = (win_w as f32 * scale) as u32;
            let h = (win_h as f32 * scale) as u32;
            if w > max_size || h > max_size {
                break;
            }
            if w >= min_size && h >= min_size {
                let step_x = (step as f32 * scale).max(1.0) as i32;
                let step_y = step_x;
                let max_x = ii.width as i32 - w as i32;
                let max_y = ii.height as i32 - h as i32;
                let mut y = 0;
                while y <= max_y {
                    let mut x = 0;
                    while x <= max_x {
                        let (mean, std_dev) = ii.mean_stdev(x, y, w as i32, h as i32);
                        let std_dev_norm = std_dev.max(1.0);
                        let area = w as f64 * h as f64;
                        let mean_val = mean * area;
                        let mut passed_all = true;
                        let mut stage_sum = 0.0f64;
                        for stage in &self.stages {
                            let mut s = 0.0f64;
                            for wc in &stage.classifiers {
                                let val = self.features[wc.feature_idx].eval(ii, x, y, scale);
                                let std_val = (val - mean_val) / std_dev_norm;
                                let t = wc.threshold as f64;
                                s += if std_val < t {
                                    wc.left_val as f64
                                } else {
                                    wc.right_val as f64
                                };
                            }
                            stage_sum += s;
                            if s < stage.threshold as f64 {
                                passed_all = false;
                                break;
                            }
                        }
                        if passed_all {
                            results.push((
                                Rect::new(x, y, w as i32, h as i32),
                                stage_sum as f32,
                            ));
                        }
                        x += step_x;
                    }
                    y += step_y;
                }
            }
            scale *= scale_factor;
            if (win_w as f32 * scale) > ii.width as f32 || (win_h as f32 * scale) > ii.height as f32 {
                break;
            }
        }
        results
    }

    pub fn load_default() -> Result<Self, BoxError> {
        let path = "data/haarcascade_frontalface_alt2.xml";
        Self::load_from_xml(path)
    }

    pub fn load_from_xml<P: AsRef<std::path::Path>>(path: P) -> Result<Self, BoxError> {
        let content = std::fs::read_to_string(path)?;
        parse_haarcascade_xml(&content)
    }
}

#[derive(Debug, Clone)]
enum XmlNode {
    Element {
        name: String,
        attrs: Vec<(String, String)>,
        children: Vec<XmlNode>,
    },
    Text(String),
}

fn parse_haarcascade_xml(content: &str) -> Result<Cascade, BoxError> {
    let dom = parse_xml(content)?;
    let root = find_element(&dom, "opencv_storage")
        .or_else(|| find_element(&dom, "haarcascade_frontalface_alt2"))
        .ok_or("No opencv_storage root")?;
    let cascade_elem = find_element(root, "cascade")
        .or_else(|| find_element(root, "haarcascade_frontalface_alt2"))
        .unwrap_or(root);
    let (win_w, win_h) = if let Some(s) = find_element(cascade_elem, "size") {
        let size_text = get_text(s);
        let size_parts: Vec<&str> = size_text.trim().split_whitespace().collect();
        let w: i32 = size_parts.get(0).ok_or("bad size")?.parse()?;
        let h: i32 = size_parts.get(1).ok_or("bad size")?.parse()?;
        (w, h)
    } else {
        // OpenCV 风格: <height>24</height> <width>24</width>
        let h: i32 = get_text(find_element(cascade_elem, "height").ok_or("No <height>")?)
            .trim().parse()?;
        let w: i32 = get_text(find_element(cascade_elem, "width").ok_or("No <width>")?)
            .trim().parse()?;
        (w, h)
    };
    let features_elem = find_element(cascade_elem, "features").ok_or("No <features>")?;
    let features = parse_features(features_elem)?;
    let stages_elem = find_element(cascade_elem, "stages").ok_or("No <stages>")?;
    let stages = parse_stages(stages_elem)?;
    Ok(Cascade {
        window: (win_w, win_h),
        features,
        stages,
    })
}

fn parse_features(root: &XmlNode) -> Result<Vec<HaarFeature>, BoxError> {
    let mut out = Vec::new();
    for child in children_of(root) {
        if let XmlNode::Element { name, children, .. } = child {
            if name == "_" {
                let tilted_elem = find_element_in(children, "tilted");
                let tilted = tilted_elem.map(|e| get_text(e).trim() == "1").unwrap_or(false);
                let rects_elem = find_element_in(children, "rects").ok_or("no rects")?;
                let rects_nodes = children_of(rects_elem);
                let mut hr = [HaarRect { x: 0, y: 0, w: 0, h: 0, weight: 0.0 }; 3];
                let mut count = 0usize;
                for rn in rects_nodes.iter().take(3) {
                    if let XmlNode::Text(t) = rn {
                        let parts: Vec<&str> = t.trim().split_whitespace().collect();
                        if parts.len() >= 5 {
                            hr[count] = HaarRect {
                                x: parts[0].parse::<i32>().unwrap_or(0),
                                y: parts[1].parse::<i32>().unwrap_or(0),
                                w: parts[2].parse::<i32>().unwrap_or(0),
                                h: parts[3].parse::<i32>().unwrap_or(0),
                                weight: parts[4].parse::<f32>().unwrap_or(0.0),
                            };
                            count += 1;
                        }
                    }
                }
                out.push(HaarFeature {
                    rects: hr,
                    num_rects: count,
                    tilted,
                });
            }
        }
    }
    Ok(out)
}

fn parse_stages(root: &XmlNode) -> Result<Vec<Stage>, BoxError> {
    let mut out = Vec::new();
    let stage_nodes: Vec<&XmlNode> = children_of(root).iter().filter(|c| {
        if let XmlNode::Element { name, .. } = c { name == "_" } else { false }
    }).collect();
    let trees_elem_opt = find_element(root, "trees");
    if let Some(trees_elem) = trees_elem_opt {
        let stage_thresholds: Vec<f32> = stage_nodes.iter().filter_map(|s| {
            find_element_in_children(s, "stage_threshold").map(get_text).and_then(|t| t.trim().parse::<f32>().ok())
        }).collect();
        let trees: Vec<&XmlNode> = children_of(trees_elem).iter().filter(|c| {
            if let XmlNode::Element { name, .. } = c { name == "_" } else { false }
        }).collect();
        let mut tree_idx = 0usize;
        for (stage_idx, _stage_node) in stage_nodes.iter().enumerate() {
            let threshold = *stage_thresholds.get(stage_idx).unwrap_or(&0.0);
            let mut classifiers = Vec::new();
            if stage_idx < stage_thresholds.len() {
                let trees_per_stage = if stage_idx < 2 {
                    stage_idx + 1
                } else {
                    3.min(trees.len().saturating_sub(tree_idx).max(1))
                };
                for _ in 0..trees_per_stage {
                    if tree_idx >= trees.len() { break; }
                    let tree = trees[tree_idx];
                    tree_idx += 1;
                    if let Some(wc) = parse_weak_from_tree(tree) {
                        classifiers.push(wc);
                    }
                }
            }
            if classifiers.is_empty() {
                for c in children_of(_stage_node) {
                    if let Some(wc) = parse_weak_classifier(c) {
                        classifiers.push(wc);
                    }
                }
            }
            out.push(Stage { threshold, classifiers });
        }
    } else {
        for stage_node in stage_nodes {
            let threshold = find_element_in_children(stage_node, "stage_threshold")
                .map(get_text)
                .and_then(|t| t.trim().parse::<f32>().ok())
                .unwrap_or(0.0);
            let mut classifiers = Vec::new();
            for c in children_of(stage_node) {
                if let Some(wc) = parse_weak_classifier(c) {
                    classifiers.push(wc);
                }
            }
            if classifiers.is_empty() {
                if let Some(trees) = find_element_in_children(stage_node, "trees") {
                    for t in children_of(trees) {
                        if let Some(wc) = parse_weak_from_tree(t) {
                            classifiers.push(wc);
                        }
                    }
                }
            }
            out.push(Stage { threshold, classifiers });
        }
    }
    Ok(out)
}

fn parse_weak_from_tree(node: &XmlNode) -> Option<WeakClassifier> {
    if let XmlNode::Element { children, .. } = node {
        for c in children {
            if let Some(wc) = parse_weak_classifier(c) {
                return Some(wc);
            }
        }
    }
    None
}

fn parse_weak_classifier(node: &XmlNode) -> Option<WeakClassifier> {
    if let XmlNode::Element { name, children, .. } = node {
        if name == "_" || name == "weakClassifiers" {
            for c in children {
                if let Some(r) = parse_weak_classifier(c) {
                    return Some(r);
                }
            }
        }
        if name == "internalNodes" || name == "leafValues" {
            return parse_opencv_weak(node, children);
        }
        if name == "feature" || name == "threshold" || name == "left_val" || name == "leftValue" {
            let feat_idx = find_element_in(children, "feature").map(get_text).and_then(|t| t.trim().parse::<usize>().ok())?;
            let threshold = find_element_in(children, "threshold").map(get_text).and_then(|t| t.trim().parse::<f32>().ok()).unwrap_or(0.0);
            let left = find_element_in(children, "left_val").or_else(|| find_element_in(children, "leftValue"))
                .map(get_text).and_then(|t| t.trim().parse::<f32>().ok()).unwrap_or(0.0);
            let right = find_element_in(children, "right_val").or_else(|| find_element_in(children, "rightValue"))
                .map(get_text).and_then(|t| t.trim().parse::<f32>().ok()).unwrap_or(1.0);
            return Some(WeakClassifier {
                feature_idx: feat_idx,
                threshold,
                left_val: left,
                right_val: right,
            });
        }
    }
    None
}

fn parse_opencv_weak(node: &XmlNode, children: &[XmlNode]) -> Option<WeakClassifier> {
    let _ = node;
    let internal = find_node_by_names(children, &["internalNodes", "InternalNodes", "internal_nodes"]).map(get_text).unwrap_or_default();
    let leaf = find_node_by_names(children, &["leafValues", "LeafValues", "leaf_values"]).map(get_text).unwrap_or_default();
    let internal_parts: Vec<&str> = internal.trim().split_whitespace().collect();
    let leaf_parts: Vec<&str> = leaf.trim().split_whitespace().collect();
    if internal_parts.len() >= 4 && leaf_parts.len() >= 2 {
        let feat_idx = internal_parts.get(2)?.parse::<usize>().ok()?;
        let threshold = internal_parts.get(3)?.parse::<f32>().ok().unwrap_or(0.0);
        let left = leaf_parts.get(0)?.parse::<f32>().ok().unwrap_or(0.0);
        let right = leaf_parts.get(1)?.parse::<f32>().ok().unwrap_or(1.0);
        return Some(WeakClassifier {
            feature_idx: feat_idx,
            threshold,
            left_val: left,
            right_val: right,
        });
    }
    None
}

fn find_node_by_names<'a>(nodes: &'a [XmlNode], names: &[&str]) -> Option<&'a XmlNode> {
    for n in nodes {
        if let XmlNode::Element { name, .. } = n {
            if names.iter().any(|x| x == name) {
                return Some(n);
            }
        }
    }
    None
}

fn parse_xml(content: &str) -> Result<XmlNode, BoxError> {
    let bytes = content.as_bytes();
    let mut i = 0;
    let mut stack: Vec<XmlNode> = Vec::new();
    let root_children = Vec::new();
    let root = XmlNode::Element {
        name: "##ROOT##".to_string(),
        attrs: Vec::new(),
        children: root_children,
    };
    stack.push(root);
    while i < bytes.len() {
        skip_ws(bytes, &mut i);
        if i >= bytes.len() { break; }
        if bytes[i] == b'<' {
            if i + 1 < bytes.len() && bytes[i + 1] == b'/' {
                i += 2;
                skip_ws(bytes, &mut i);
                let start = i;
                while i < bytes.len() && bytes[i] != b'>' { i += 1; }
                i = (i + 1).min(bytes.len());
                let _close_name = std::str::from_utf8(&bytes[start..(i - 1).min(start)]).unwrap_or("").trim();
                if stack.len() > 1 {
                    let closed = stack.pop().unwrap();
                    if let Some(parent) = stack.last_mut() {
                        if let XmlNode::Element { children, .. } = parent {
                            children.push(closed);
                        }
                    }
                }
                continue;
            }
            if i + 3 < bytes.len() && &bytes[i..i + 4] == b"<!--" {
                i += 4;
                while i + 2 < bytes.len() && &bytes[i..i + 3] != b"-->" { i += 1; }
                i += 3;
                continue;
            }
            if i + 1 < bytes.len() && bytes[i + 1] == b'?' {
                i += 2;
                while i + 1 < bytes.len() && !(bytes[i] == b'?' && bytes[i + 1] == b'>') { i += 1; }
                i += 2;
                continue;
            }
            if i + 8 < bytes.len() && &bytes[i..i + 9] == b"<!DOCTYPE" {
                while i < bytes.len() && bytes[i] != b'>' { i += 1; }
                i += 1;
                continue;
            }
            i += 1;
            let name_start = i;
            while i < bytes.len() && !bytes[i].is_ascii_whitespace() && bytes[i] != b'/' && bytes[i] != b'>' { i += 1; }
            let name = std::str::from_utf8(&bytes[name_start..i]).unwrap_or("").to_string();
            let mut attrs = Vec::new();
            loop {
                skip_ws(bytes, &mut i);
                if i >= bytes.len() { break; }
                if bytes[i] == b'/' { i += 1; continue; }
                if bytes[i] == b'>' { break; }
                let a_start = i;
                while i < bytes.len() && bytes[i] != b'=' && !bytes[i].is_ascii_whitespace() && bytes[i] != b'>' && bytes[i] != b'/' { i += 1; }
                let attr_name = std::str::from_utf8(&bytes[a_start..i]).unwrap_or("").to_string();
                skip_ws(bytes, &mut i);
                if i < bytes.len() && bytes[i] == b'=' {
                    i += 1;
                    skip_ws(bytes, &mut i);
                    let quote = if i < bytes.len() { bytes[i] } else { 0 };
                    if quote == b'"' || quote == b'\'' {
                        i += 1;
                        let v_start = i;
                        while i < bytes.len() && bytes[i] != quote { i += 1; }
                        let attr_val = std::str::from_utf8(&bytes[v_start..i]).unwrap_or("").to_string();
                        i += 1;
                        if !attr_name.is_empty() {
                            attrs.push((attr_name, attr_val));
                        }
                    } else { continue; }
                } else if !attr_name.is_empty() {
                    attrs.push((attr_name, String::new()));
                }
            }
            if i < bytes.len() {
                let self_closing = bytes[i] == b'/' || (i + 1 < bytes.len() && bytes[i] == b'/' && bytes[i + 1] == b'>');
                if bytes[i] == b'>' { i += 1; }
                else if self_closing { i = (i + 2).min(bytes.len()); }
                let elem = XmlNode::Element { name: name.clone(), attrs, children: Vec::new() };
                if self_closing || is_void(&name) {
                    if let Some(parent) = stack.last_mut() {
                        if let XmlNode::Element { children, .. } = parent {
                            children.push(elem);
                        }
                    }
                } else {
                    stack.push(elem);
                }
            }
        } else {
            let start = i;
            while i < bytes.len() && bytes[i] != b'<' { i += 1; }
            let text = std::str::from_utf8(&bytes[start..i]).unwrap_or("");
            let trimmed = text.trim();
            if !trimmed.is_empty() {
                if let Some(parent) = stack.last_mut() {
                    if let XmlNode::Element { children, .. } = parent {
                        children.push(XmlNode::Text(text.to_string()));
                    }
                }
            }
        }
    }
    while stack.len() > 1 {
        let closed = stack.pop().unwrap();
        if let Some(parent) = stack.last_mut() {
            if let XmlNode::Element { children, .. } = parent {
                children.push(closed);
            }
        }
    }
    Ok(stack.pop().unwrap())
}

fn is_void(_name: &str) -> bool { false }

fn skip_ws(b: &[u8], i: &mut usize) {
    while *i < b.len() && b[*i].is_ascii_whitespace() { *i += 1; }
}

fn find_element<'a>(node: &'a XmlNode, name: &str) -> Option<&'a XmlNode> {
    if let XmlNode::Element { name: n, children, .. } = node {
        if n == name { return Some(node); }
        for c in children {
            if let Some(r) = find_element(c, name) { return Some(r); }
        }
    }
    None
}

fn find_element_in<'a>(nodes: &'a [XmlNode], name: &str) -> Option<&'a XmlNode> {
    for n in nodes {
        if let Some(r) = find_element(n, name) { return Some(r); }
    }
    None
}

fn find_element_in_children<'a>(node: &'a XmlNode, name: &str) -> Option<&'a XmlNode> {
    if let XmlNode::Element { children, .. } = node {
        find_element_in(children, name)
    } else { None }
}

fn children_of(node: &XmlNode) -> &[XmlNode] {
    if let XmlNode::Element { children, .. } = node { children.as_slice() } else { &[] }
}

fn get_text(node: &XmlNode) -> String {
    let mut out = String::new();
    collect_text(node, &mut out);
    out
}

fn collect_text(node: &XmlNode, out: &mut String) {
    match node {
        XmlNode::Element { children, .. } => {
            for c in children { collect_text(c, out); }
        }
        XmlNode::Text(t) => out.push_str(t),
    }
}
