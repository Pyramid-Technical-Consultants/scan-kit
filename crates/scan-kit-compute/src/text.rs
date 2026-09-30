//! One glyph atlas. Titles, ticks, and category labels all go through `layout`.

use std::sync::OnceLock;

const PX: f32 = 18.0;
const ATLAS: u32 = 256;
const FONT: &[u8] = include_bytes!("../assets/SourceSans3-Regular.ttf");

#[derive(Clone, Copy)]
pub struct Glyph {
    pub advance: f32,
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub uv: [f32; 4],
}

pub struct Atlas {
    pub pixels: Vec<u8>,
    pub size: u32,
    pub glyphs: Vec<Glyph>,
    pub ascent: f32,
    pub line_height: f32,
}

#[derive(Clone, Copy)]
pub struct Stamp {
    pub x: f32,
    pub y: f32,
    pub w: f32,
    pub h: f32,
    pub uv: [f32; 4],
}

pub fn atlas() -> &'static Atlas {
    static ATLAS_CELL: OnceLock<Atlas> = OnceLock::new();
    ATLAS_CELL.get_or_init(build)
}

pub fn layout(text: &str) -> (Vec<Stamp>, f32) {
    let atlas = atlas();
    let mut pen = 0.0;
    let mut stamps = Vec::new();
    for ch in text.chars() {
        let Some(glyph) = glyph(atlas, ch) else {
            pen += atlas.glyphs[0].advance;
            continue;
        };
        if glyph.w > 0.0 && glyph.h > 0.0 {
            stamps.push(Stamp {
                x: pen + glyph.x,
                y: glyph.y,
                w: glyph.w,
                h: glyph.h,
                uv: glyph.uv,
            });
        }
        pen += glyph.advance;
    }
    (stamps, pen)
}

pub fn text_width(text: &str) -> f32 {
    layout(text).1
}

fn glyph(atlas: &Atlas, ch: char) -> Option<&Glyph> {
    let index = ch as u32;
    if (32..127).contains(&index) {
        Some(&atlas.glyphs[(index - 32) as usize])
    } else {
        None
    }
}

fn build() -> Atlas {
    let font = fontdue::Font::from_bytes(FONT, fontdue::FontSettings::default()).expect("plot font");
    let line = font
        .horizontal_line_metrics(PX)
        .expect("plot font metrics");
    let mut pixels = vec![0u8; (ATLAS * ATLAS) as usize];
    let mut glyphs = Vec::with_capacity(95);
    let mut cursor_x = 1u32;
    let mut cursor_y = 1u32;
    let mut row_h = 0u32;
    for code in 32u32..127 {
        let (metrics, bitmap) = font.rasterize(char::from_u32(code).unwrap_or(' '), PX);
        let w = metrics.width as u32;
        let h = metrics.height as u32;
        if w > 0 && cursor_x + w + 1 >= ATLAS {
            cursor_x = 1;
            cursor_y += row_h + 1;
            row_h = 0;
        }
        if w > 0 && h > 0 && cursor_y + h < ATLAS {
            for row in 0..h {
                let dest = ((cursor_y + row) * ATLAS + cursor_x) as usize;
                let src = (row * w) as usize;
                pixels[dest..dest + w as usize].copy_from_slice(&bitmap[src..src + w as usize]);
            }
        }
        let top = -(metrics.ymin + metrics.height as i32) as f32;
        glyphs.push(Glyph {
            advance: metrics.advance_width,
            x: metrics.xmin as f32,
            y: top,
            w: w as f32,
            h: h as f32,
            uv: [
                cursor_x as f32 / ATLAS as f32,
                cursor_y as f32 / ATLAS as f32,
                (cursor_x + w) as f32 / ATLAS as f32,
                (cursor_y + h) as f32 / ATLAS as f32,
            ],
        });
        if w > 0 {
            cursor_x += w + 1;
            row_h = row_h.max(h);
        }
    }
    Atlas {
        pixels,
        size: ATLAS,
        glyphs,
        ascent: line.ascent,
        line_height: (line.ascent - line.descent).max(PX),
    }
}
