//! Parser for DCS `SessionLogFile.log`.

use std::collections::BTreeMap;

#[derive(Clone, Debug, PartialEq)]
pub struct LayerEvent {
    pub layer: i32,
    pub kind: String,
    pub seconds: f32,
}

#[derive(Clone, Debug, Default)]
pub struct SessionLog {
    pub lines: usize,
    pub level_counts: BTreeMap<String, usize>,
    pub templates: BTreeMap<String, usize>,
    pub timeline: Vec<LayerEvent>,
    pub issues: Vec<String>,
    pub wdt: BTreeMap<String, usize>,
}

/// `(template, count_a, count_b, a - b)` sorted by the largest absolute delta.
pub fn compare_templates(a: &SessionLog, b: &SessionLog) -> Vec<(String, usize, usize, i32)> {
    let mut keys: Vec<&str> = a
        .templates
        .keys()
        .chain(b.templates.keys())
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys.dedup();
    let mut rows = Vec::new();
    for key in keys {
        let ca = a.templates.get(key).copied().unwrap_or(0);
        let cb = b.templates.get(key).copied().unwrap_or(0);
        rows.push((key.to_owned(), ca, cb, ca as i32 - cb as i32));
    }
    rows.sort_by(|left, right| {
        right
            .3
            .abs()
            .cmp(&left.3.abs())
            .then(right.1.max(right.2).cmp(&(left.1.max(left.2))))
    });
    rows
}

pub fn parse_session_log(text: &str) -> SessionLog {
    let mut log = SessionLog::default();
    for raw in text.lines() {
        let Some((level, message)) = split_log_line(raw) else {
            continue;
        };
        log.lines += 1;
        *log.level_counts.entry(level.to_owned()).or_default() += 1;
        if level == "ERROR" {
            log.issues.push(message.to_owned());
        }
        if !is_noise(message) {
            *log.templates.entry(message_template(message)).or_default() += 1;
        }
        if let Some(event) = timeline_event(message) {
            log.timeline.push(event);
        }
        if let Some(device) = wdt_device(message) {
            *log.wdt.entry(device).or_default() += 1;
        }
    }
    log
}

fn split_log_line(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    let bracket = line.find(" [")?;
    let level = line[..bracket].rsplit(' ').next()?;
    if level.is_empty() || !level.chars().all(|ch| ch.is_ascii_uppercase()) {
        return None;
    }
    let rest = &line[bracket + 2..];
    let close = rest.find("] ")?;
    Some((level, &rest[close + 2..]))
}

fn is_noise(message: &str) -> bool {
    message.contains("Got ACK to ") || message.contains("Received command:")
}

fn message_template(message: &str) -> String {
    let chars: Vec<char> = message.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        if chars[i].is_ascii_digit() {
            out.push('#');
            i += 1;
            while i < chars.len()
                && (chars[i].is_ascii_digit()
                    || (chars[i] == '.' && i + 1 < chars.len() && chars[i + 1].is_ascii_digit()))
            {
                i += 1;
            }
        } else {
            out.push(chars[i]);
            i += 1;
        }
    }
    let collapsed = out.split_whitespace().collect::<Vec<_>>().join(" ");
    collapsed.replace("/var/log/", "<path>/")
}

fn timeline_event(message: &str) -> Option<LayerEvent> {
    if message.starts_with("LOAD 3D MAP") {
        return None;
    }
    let kind = if message.starts_with("START MAP") {
        "start"
    } else if message.starts_with("SCAN EXECUTING") {
        "scan"
    } else if message.starts_with("LOAD 2D MAP") {
        "load"
    } else {
        return None;
    };
    let layer = between(message, "LAYER: ", " ")?.parse().ok()?;
    let seconds = between(message, "T=", "s")?.parse().ok()?;
    Some(LayerEvent {
        layer,
        kind: kind.to_owned(),
        seconds,
    })
}

fn wdt_device(message: &str) -> Option<String> {
    let index = message.find(" wdt read cnt:")?;
    let device = message[..index].trim();
    (!device.is_empty()).then(|| device.to_owned())
}

fn between<'a>(text: &'a str, start: &str, end: &str) -> Option<&'a str> {
    let from = text.find(start)? + start.len();
    let rest = &text[from..];
    let to = rest.find(end)?;
    Some(&rest[..to])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sk_req_033_session_log_timeline_issue_and_diff() {
        let a = parse_session_log(
            "2026-01-01 00:00:00,000 INFO [dcs] START MAP FOR LAYER: 1 - TIMELINE(begin): T=0.1s\n\
             2026-01-01 00:00:01,000 ERROR [dcs] fault 12\n\
             2026-01-01 00:00:02,000 DEBUG [dcs] IC1 wdt read cnt: 3 - write counter: 4\n",
        );
        assert_eq!(a.level_counts.get("ERROR"), Some(&1));
        assert_eq!(a.issues, vec!["fault 12".to_owned()]);
        assert_eq!(a.timeline.len(), 1);
        assert_eq!(a.timeline[0].layer, 1);
        assert!((a.timeline[0].seconds - 0.1).abs() < 1e-4);
        assert_eq!(a.wdt.get("IC1"), Some(&1));
        assert_eq!(a.templates.get("fault #"), Some(&1));
        let b = parse_session_log("2026-01-01 00:00:01,000 ERROR [dcs] fault 9\n");
        let diff = compare_templates(&a, &b);
        assert!(diff.iter().any(|row| row.0 == "fault #" && row.3 == 0));
    }
}
