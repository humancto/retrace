//! Live preview: boxes, track ids and trails drawn straight into RGB frames, JPEG-encoded and
//! shared with the API so the UI can show detections as they happen (MJPEG). No extra crates:
//! a 3x5 bitmap font is enough for labels like "PERSON #12".

use anyhow::Result;
use image::RgbImage;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Latest annotated frame per camera, replaced in place by the ingest thread.
#[derive(Clone, Default)]
pub struct LiveFrames(pub Arc<Mutex<HashMap<String, LiveFrame>>>);

impl std::fmt::Debug for LiveFrames {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "LiveFrames({} cams)", self.cams().len())
    }
}

#[derive(Clone)]
pub struct LiveFrame {
    pub jpeg: Vec<u8>,
    pub seq: u64,
    pub t: i64,
    pub tracks: usize,
    pub width: u32,
    pub height: u32,
}

impl LiveFrames {
    pub fn publish(
        &self,
        cam: &str,
        jpeg: Vec<u8>,
        t: i64,
        tracks: usize,
        width: u32,
        height: u32,
    ) {
        if let Ok(mut m) = self.0.lock() {
            let seq = m.get(cam).map(|f| f.seq + 1).unwrap_or(1);
            m.insert(
                cam.to_string(),
                LiveFrame {
                    jpeg,
                    seq,
                    t,
                    tracks,
                    width,
                    height,
                },
            );
        }
    }

    pub fn get(&self, cam: &str) -> Option<LiveFrame> {
        self.0.lock().ok().and_then(|m| m.get(cam).cloned())
    }

    pub fn cams(&self) -> Vec<(String, i64, u64)> {
        self.0
            .lock()
            .map(|m| m.iter().map(|(k, f)| (k.clone(), f.t, f.seq)).collect())
            .unwrap_or_default()
    }
}

pub fn class_color(class: &str) -> [u8; 3] {
    match class {
        "person" => [46, 204, 113],
        "car" | "truck" | "bus" => [241, 196, 15],
        "bicycle" | "motorcycle" => [52, 152, 219],
        _ => [231, 76, 60],
    }
}

pub fn encode_jpeg(img: &RgbImage, quality: u8) -> Result<Vec<u8>> {
    let mut buf = Vec::with_capacity(64 * 1024);
    let mut enc = image::codecs::jpeg::JpegEncoder::new_with_quality(&mut buf, quality);
    enc.encode_image(img)?;
    Ok(buf)
}

fn put(img: &mut RgbImage, x: i32, y: i32, c: [u8; 3]) {
    if x >= 0 && y >= 0 && (x as u32) < img.width() && (y as u32) < img.height() {
        img.put_pixel(x as u32, y as u32, image::Rgb(c));
    }
}

pub fn fill_rect(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, c: [u8; 3]) {
    for yy in y..y + h {
        for xx in x..x + w {
            put(img, xx, yy, c);
        }
    }
}

pub fn draw_rect(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, thick: i32, c: [u8; 3]) {
    fill_rect(img, x, y, w, thick, c);
    fill_rect(img, x, y + h - thick, w, thick, c);
    fill_rect(img, x, y, thick, h, c);
    fill_rect(img, x + w - thick, y, thick, h, c);
}

/// 3x5 glyphs, one row per entry, MSB = left pixel.
fn glyph(ch: char) -> [u8; 5] {
    match ch.to_ascii_uppercase() {
        'A' => [0b010, 0b101, 0b111, 0b101, 0b101],
        'B' => [0b110, 0b101, 0b110, 0b101, 0b110],
        'C' => [0b011, 0b100, 0b100, 0b100, 0b011],
        'D' => [0b110, 0b101, 0b101, 0b101, 0b110],
        'E' => [0b111, 0b100, 0b110, 0b100, 0b111],
        'F' => [0b111, 0b100, 0b110, 0b100, 0b100],
        'G' => [0b011, 0b100, 0b101, 0b101, 0b011],
        'H' => [0b101, 0b101, 0b111, 0b101, 0b101],
        'I' => [0b111, 0b010, 0b010, 0b010, 0b111],
        'J' => [0b001, 0b001, 0b001, 0b101, 0b010],
        'K' => [0b101, 0b101, 0b110, 0b101, 0b101],
        'L' => [0b100, 0b100, 0b100, 0b100, 0b111],
        'M' => [0b101, 0b111, 0b111, 0b101, 0b101],
        'N' => [0b110, 0b101, 0b101, 0b101, 0b101],
        'O' => [0b010, 0b101, 0b101, 0b101, 0b010],
        'P' => [0b110, 0b101, 0b110, 0b100, 0b100],
        'Q' => [0b010, 0b101, 0b101, 0b111, 0b011],
        'R' => [0b110, 0b101, 0b110, 0b101, 0b101],
        'S' => [0b011, 0b100, 0b010, 0b001, 0b110],
        'T' => [0b111, 0b010, 0b010, 0b010, 0b010],
        'U' => [0b101, 0b101, 0b101, 0b101, 0b111],
        'V' => [0b101, 0b101, 0b101, 0b101, 0b010],
        'W' => [0b101, 0b101, 0b111, 0b111, 0b101],
        'X' => [0b101, 0b101, 0b010, 0b101, 0b101],
        'Y' => [0b101, 0b101, 0b010, 0b010, 0b010],
        'Z' => [0b111, 0b001, 0b010, 0b100, 0b111],
        '0' => [0b111, 0b101, 0b101, 0b101, 0b111],
        '1' => [0b010, 0b110, 0b010, 0b010, 0b111],
        '2' => [0b111, 0b001, 0b111, 0b100, 0b111],
        '3' => [0b111, 0b001, 0b011, 0b001, 0b111],
        '4' => [0b101, 0b101, 0b111, 0b001, 0b001],
        '5' => [0b111, 0b100, 0b111, 0b001, 0b111],
        '6' => [0b111, 0b100, 0b111, 0b101, 0b111],
        '7' => [0b111, 0b001, 0b010, 0b010, 0b010],
        '8' => [0b111, 0b101, 0b111, 0b101, 0b111],
        '9' => [0b111, 0b101, 0b111, 0b001, 0b111],
        '#' => [0b101, 0b111, 0b101, 0b111, 0b101],
        ':' => [0b000, 0b010, 0b000, 0b010, 0b000],
        '.' => [0b000, 0b000, 0b000, 0b000, 0b010],
        '-' => [0b000, 0b000, 0b111, 0b000, 0b000],
        '·' => [0b000, 0b000, 0b010, 0b000, 0b000],
        '|' => [0b010, 0b010, 0b010, 0b010, 0b010],
        '/' => [0b001, 0b001, 0b010, 0b100, 0b100],
        '%' => [0b101, 0b001, 0b010, 0b100, 0b101],
        _ => [0b000; 5],
    }
}

/// Width in pixels of `text` at `scale`.
pub fn text_width(text: &str, scale: i32) -> i32 {
    text.chars().count() as i32 * 4 * scale
}

pub fn draw_text(img: &mut RgbImage, x: i32, y: i32, text: &str, scale: i32, c: [u8; 3]) {
    let mut cx = x;
    for ch in text.chars() {
        let g = glyph(ch);
        for (row, bits) in g.iter().enumerate() {
            for col in 0..3 {
                if bits & (0b100 >> col) != 0 {
                    fill_rect(
                        img,
                        cx + col * scale,
                        y + row as i32 * scale,
                        scale,
                        scale,
                        c,
                    );
                }
            }
        }
        cx += 4 * scale;
    }
}

/// A box with a filled label tab above it.
pub fn draw_box_label(img: &mut RgbImage, x: i32, y: i32, w: i32, h: i32, label: &str, c: [u8; 3]) {
    draw_rect(img, x, y, w, h, 2, c);
    let scale = 2;
    let tw = text_width(label, scale) + 6;
    let th = 5 * scale + 6;
    let ly = if y - th >= 0 { y - th } else { y };
    fill_rect(img, x, ly, tw, th, c);
    draw_text(img, x + 3, ly + 3, label, scale, [0, 0, 0]);
}

/// Trail of recent centre points (normalised coordinates).
pub fn draw_trail(img: &mut RgbImage, pts: &[(f32, f32)], c: [u8; 3]) {
    let (w, h) = (img.width() as f32, img.height() as f32);
    for (cx, cy) in pts {
        fill_rect(img, (cx * w) as i32 - 1, (cy * h) as i32 - 1, 3, 3, c);
    }
}

/// Status bar in the top-left corner.
pub fn draw_status(img: &mut RgbImage, text: &str) {
    let scale = 2;
    let tw = text_width(text, scale) + 8;
    let th = 5 * scale + 8;
    fill_rect(img, 0, 0, tw, th, [0, 0, 0]);
    draw_text(img, 4, 4, text, scale, [255, 255, 255]);
    fill_rect(img, tw - 2, 0, 2, th, [231, 76, 60]);
}
