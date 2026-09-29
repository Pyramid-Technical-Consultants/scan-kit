//! `termination_summary.txt` metadata.
//!
//! The field rules match `scan_kit/common/session_meta.py`. The Python module
//! stays until the Qt app no longer imports it.

#[derive(Clone, Debug, PartialEq)]
pub struct SessionMeta {
    pub date: Option<SummaryDate>,
    pub primary_mu: Option<f64>,
    pub treatment_time_s: Option<i32>,
    pub room_number: Option<i32>,
    pub config_name: Option<String>,
    pub map_extent_mm: Option<f64>,
    pub layer_count: Option<i32>,
}

/// A naive local timestamp from a summary line. No timezone is stored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SummaryDate {
    pub year: i32,
    pub month: u32,
    pub day: u32,
    pub hour: u32,
    pub minute: u32,
    pub second: u32,
}

impl SessionMeta {
    pub fn empty() -> Self {
        Self {
            date: None,
            primary_mu: None,
            treatment_time_s: None,
            room_number: None,
            config_name: None,
            map_extent_mm: None,
            layer_count: None,
        }
    }

    pub fn short_date(&self) -> String {
        match self.date {
            Some(date) => format!("{:02}/{:02}/{:02}", date.month, date.day, date.year % 100),
            None => "?".to_owned(),
        }
    }

    pub fn date_iso(&self) -> Option<String> {
        self.date.map(|date| {
            format!(
                "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
                date.year, date.month, date.day, date.hour, date.minute, date.second
            )
        })
    }

    pub fn short_mu(&self) -> String {
        match self.primary_mu {
            Some(mu) => format!("{mu:.1}"),
            None => "?".to_owned(),
        }
    }

    pub fn short_extent(&self) -> String {
        match self.map_extent_mm {
            Some(extent) => python_round(extent).to_string(),
            None => "?".to_owned(),
        }
    }

    pub fn short_layers(&self) -> String {
        match self.layer_count {
            Some(layers) => layers.to_string(),
            None => "?".to_owned(),
        }
    }

    pub fn short_time(&self) -> String {
        match self.treatment_time_s {
            Some(seconds) => {
                let minutes = seconds.div_euclid(60);
                let remain = seconds.rem_euclid(60);
                format!("{minutes}:{remain:02}")
            }
            None => "?".to_owned(),
        }
    }

    pub fn short_room(&self) -> String {
        match self.room_number {
            Some(room) => room.to_string(),
            None => "?".to_owned(),
        }
    }

    pub fn short_config(&self) -> String {
        match self.config_name.as_deref().map(str::trim) {
            Some(name) if !name.is_empty() => name.to_owned(),
            _ => "?".to_owned(),
        }
    }
}

impl SummaryDate {
    pub fn parse_iso(text: &str) -> Option<Self> {
        let (date, time) = text.split_once('T')?;
        let mut date_parts = date.split('-');
        let year: i32 = date_parts.next()?.parse().ok()?;
        let month: u32 = date_parts.next()?.parse().ok()?;
        let day: u32 = date_parts.next()?.parse().ok()?;
        if date_parts.next().is_some() {
            return None;
        }
        let mut time_parts = time.split(':');
        let hour: u32 = time_parts.next()?.parse().ok()?;
        let minute: u32 = time_parts.next()?.parse().ok()?;
        let second_text = time_parts.next()?;
        if time_parts.next().is_some() {
            return None;
        }
        let second: u32 = second_text.split('.').next()?.parse().ok()?;
        Some(Self {
            year,
            month,
            day,
            hour,
            minute,
            second,
        })
    }
}

/// Fill missing extent and layer fields. Do not overwrite values already set.
pub fn merge_session_geom(
    meta: Option<SessionMeta>,
    map_extent_mm: Option<f64>,
    layer_count: Option<i32>,
) -> Option<SessionMeta> {
    if map_extent_mm.is_none() && layer_count.is_none() {
        return meta;
    }
    let mut base = meta.unwrap_or_else(SessionMeta::empty);
    if base.map_extent_mm.is_none() {
        base.map_extent_mm = map_extent_mm;
    }
    if base.layer_count.is_none() {
        base.layer_count = layer_count;
    }
    Some(base)
}

pub fn parse_termination_summary_text(text: &str) -> SessionMeta {
    let mut date = None;
    let mut primary_mu = None;
    let mut treatment_s = None;
    let mut room_number = None;
    let mut config_name = None;
    let mut extent_w = None;
    let mut extent_h = None;
    let mut layer_count = None;

    for line in text.lines() {
        let line = line.trim();
        if let Some(rest) = labeled(line, "Date") {
            if let Some(parsed) = parse_summary_date(rest) {
                date = Some(parsed);
            }
        } else if line.starts_with("Primary total dose:") {
            if let Some(parsed) = labeled_number(line, "Primary total dose") {
                primary_mu = Some(parsed);
            }
        } else if line.starts_with("Treatment time:") {
            if let Some(parsed) = labeled_number(line, "Treatment time") {
                treatment_s = Some(parsed as i32);
            }
        } else if line.starts_with("Room number:") {
            if let Some(parsed) = labeled_number(line, "Room number") {
                room_number = Some(parsed as i32);
            }
        } else if let Some(rest) = labeled(line, "Configuration name") {
            let name = rest.trim();
            config_name = if name.is_empty() {
                None
            } else {
                Some(name.to_owned())
            };
        } else if line.starts_with("Spot extent width:") {
            if let Some(parsed) = labeled_number(line, "Spot extent width") {
                extent_w = Some(parsed);
            }
        } else if line.starts_with("Spot extent height:") {
            if let Some(parsed) = labeled_number(line, "Spot extent height") {
                extent_h = Some(parsed);
            }
        } else if line.starts_with("Layer delivery:") {
            if let Some(parsed) = parse_layer_delivery(line) {
                layer_count = Some(parsed);
            }
        }
    }

    let map_extent_mm = match (extent_w, extent_h) {
        (Some(width), Some(height)) => Some(width.max(height)),
        (Some(width), None) => Some(width),
        (None, Some(height)) => Some(height),
        (None, None) => None,
    };

    SessionMeta {
        date,
        primary_mu,
        treatment_time_s: treatment_s,
        room_number,
        config_name,
        map_extent_mm,
        layer_count,
    }
}

fn labeled<'a>(line: &'a str, label: &str) -> Option<&'a str> {
    line.strip_prefix(label)?.strip_prefix(':')
}

fn labeled_number(line: &str, label: &str) -> Option<f64> {
    let rest = labeled(line, label)?.trim();
    leading_number(rest)
}

fn parse_layer_delivery(line: &str) -> Option<i32> {
    let mut rest = labeled(line, "Layer delivery")?.trim();
    if rest.is_empty() {
        return None;
    }
    if let Some((_, planned)) = rest.split_once('/') {
        rest = planned.trim();
    }
    Some(leading_number(rest)? as i32)
}

fn leading_number(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    if bytes.is_empty() {
        return None;
    }
    let mut index = 0;
    if bytes[0] == b'+' || bytes[0] == b'-' {
        index = 1;
    }
    let mut saw_digit = false;
    let mut saw_dot = false;
    while index < bytes.len() {
        match bytes[index] {
            b'0'..=b'9' => {
                saw_digit = true;
                index += 1;
            }
            b'.' if !saw_dot => {
                saw_dot = true;
                index += 1;
            }
            _ => break,
        }
    }
    if !saw_digit {
        return None;
    }
    text[..index].parse().ok()
}

fn parse_summary_date(rest: &str) -> Option<SummaryDate> {
    let parts: Vec<&str> = rest.split_whitespace().collect();
    if parts.len() != 5 || weekday_index(parts[0]).is_none() {
        return None;
    }
    let month = month_index(parts[1])?;
    let day: u32 = parts[2].parse().ok()?;
    let year: i32 = parts[4].parse().ok()?;
    let mut clock = parts[3].split(':');
    let hour: u32 = clock.next()?.parse().ok()?;
    let minute: u32 = clock.next()?.parse().ok()?;
    let second: u32 = clock.next()?.parse().ok()?;
    if clock.next().is_some()
        || !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return None;
    }
    Some(SummaryDate {
        year,
        month,
        day,
        hour,
        minute,
        second,
    })
}

fn weekday_index(name: &str) -> Option<u8> {
    Some(match name {
        "Sun" => 0,
        "Mon" => 1,
        "Tue" => 2,
        "Wed" => 3,
        "Thu" => 4,
        "Fri" => 5,
        "Sat" => 6,
        _ => return None,
    })
}

fn month_index(name: &str) -> Option<u32> {
    Some(match name {
        "Jan" => 1,
        "Feb" => 2,
        "Mar" => 3,
        "Apr" => 4,
        "May" => 5,
        "Jun" => 6,
        "Jul" => 7,
        "Aug" => 8,
        "Sep" => 9,
        "Oct" => 10,
        "Nov" => 11,
        "Dec" => 12,
        _ => return None,
    })
}

/// Python 3 `round` is half-to-even. `int(round(x))` is what the extent column shows.
fn python_round(value: f64) -> i64 {
    if !value.is_finite() {
        return 0;
    }
    let floor = value.floor();
    let diff = value - floor;
    if (diff - 0.5).abs() < 1e-9 {
        let even = floor as i64;
        if even % 2 == 0 {
            even
        } else {
            even + 1
        }
    } else {
        value.round() as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\
Termination status: eNORMAL
Date: Thu Sep 10 21:07:41 2026
Session ID: 1584882828
Error code: 0
Room number: 4
Primary total dose: 38.0223 MU
Treatment time: 251.604 seconds
Configuration name: working_hvtt_new_cal_gates_tuned
";

    #[test]
    fn sk_req_004_termination_summary_parses() {
        let meta = parse_termination_summary_text(SAMPLE);
        assert_eq!(
            meta.config_name.as_deref(),
            Some("working_hvtt_new_cal_gates_tuned")
        );
        assert_eq!(meta.short_config(), "working_hvtt_new_cal_gates_tuned");
        assert_eq!(meta.room_number, Some(4));
        assert_eq!(meta.primary_mu, Some(38.0223));
        assert_eq!(meta.short_mu(), "38.0");
        assert_eq!(meta.treatment_time_s, Some(251));
        assert_eq!(meta.short_time(), "4:11");
        assert_eq!(
            meta.date,
            Some(SummaryDate {
                year: 2026,
                month: 9,
                day: 10,
                hour: 21,
                minute: 7,
                second: 41,
            })
        );
        assert_eq!(meta.short_date(), "09/10/26");
        assert_eq!(meta.date_iso().as_deref(), Some("2026-09-10T21:07:41"));
        assert_eq!(meta.map_extent_mm, None);
        assert_eq!(meta.layer_count, None);
        assert_eq!(meta.short_extent(), "?");
        assert_eq!(meta.short_layers(), "?");

        let missing = parse_termination_summary_text(
            "Date: Thu Sep 10 21:07:41 2026\nPrimary total dose: 1.0\n",
        );
        assert_eq!(missing.config_name, None);
        assert_eq!(missing.short_config(), "?");
        assert_eq!(missing.room_number, None);
        assert_eq!(missing.short_room(), "?");

        let blank = parse_termination_summary_text("Configuration name:   \n");
        assert_eq!(blank.config_name, None);
        assert_eq!(blank.short_config(), "?");

        let geom = parse_termination_summary_text(
            "\
Date: Tue Aug 25 21:35:22 2026
Layer delivery: 27/27
Spot extent width: 118.75 mm
Spot extent height: 110.592 mm
Configuration name: working_no_hvttt_new_cal
",
        );
        assert_eq!(geom.map_extent_mm, Some(118.75));
        assert_eq!(geom.short_extent(), "119");
        assert_eq!(geom.layer_count, Some(27));
        assert_eq!(geom.short_layers(), "27");

        let zero =
            parse_termination_summary_text("Spot extent width: 0 mm\nSpot extent height: 0 mm\n");
        assert_eq!(zero.map_extent_mm, Some(0.0));
        assert_eq!(zero.short_extent(), "0");

        assert_eq!(
            parse_termination_summary_text("Layer delivery: 56/76\n").layer_count,
            Some(76)
        );
        assert_eq!(
            parse_termination_summary_text("Layer delivery: 18\n").layer_count,
            Some(18)
        );
        let no_geom = parse_termination_summary_text("Primary total dose: 1.0\n");
        assert_eq!(no_geom.map_extent_mm, None);
        assert_eq!(no_geom.layer_count, None);

        let merged = merge_session_geom(Some(geom.clone()), Some(1.0), Some(3));
        assert_eq!(merged.unwrap().map_extent_mm, Some(118.75));
        let filled = merge_session_geom(Some(no_geom), Some(12.0), Some(4)).unwrap();
        assert_eq!(filled.map_extent_mm, Some(12.0));
        assert_eq!(filled.layer_count, Some(4));
        assert!(merge_session_geom(None, None, None).is_none());
    }
}
