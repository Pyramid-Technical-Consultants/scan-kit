use std::path::Path;

use scan_kit_core::{compare_templates, parse_session_log, DataTable, PlotScene};

use super::{scene, session_text};

pub(super) fn session_log(root: &Path, session_ids: &[String]) -> PlotScene {
    let mut rows = Vec::new();
    let mut parsed = Vec::new();
    for session in session_ids {
        let text = session_text(root, session, "SessionLogFile.log");
        let log = parse_session_log(&text);
        rows.push(vec![
            session.clone(),
            "Overview".into(),
            format!("{} lines", log.lines),
        ]);
        for (level, count) in &log.level_counts {
            rows.push(vec![
                session.clone(),
                "Overview".into(),
                format!("{level} {count}"),
            ]);
        }
        for event in &log.timeline {
            rows.push(vec![
                session.clone(),
                "Timeline".into(),
                format!("layer {} {} {:.3}s", event.layer, event.kind, event.seconds),
            ]);
        }
        for issue in log.issues.iter().take(40) {
            rows.push(vec![session.clone(), "Errors".into(), issue.clone()]);
        }
        for (device, count) in &log.wdt {
            rows.push(vec![
                session.clone(),
                "Watchdog".into(),
                format!("{device} {count}"),
            ]);
        }
        parsed.push((session.clone(), log));
    }
    if parsed.len() == 2 {
        for (template, a, b, delta) in compare_templates(&parsed[0].1, &parsed[1].1)
            .into_iter()
            .take(40)
        {
            rows.push(vec![
                format!("{} vs {}", parsed[0].0, parsed[1].0),
                "Diff".into(),
                format!("{template} {a} {b} {delta}"),
            ]);
        }
    }
    let mut scene = scene("Session Log Compare", Vec::new(), Vec::new());
    scene.table = Some(DataTable {
        columns: vec!["Session".into(), "Section".into(), "Detail".into()],
        rows,
    });
    scene
}
