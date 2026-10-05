//! One progress report for every long task.
//!
//! A view starts a task, polls it, and cancels it. `total == 0` means the length
//! is not known yet. These types do not touch files, the GPU, or the window.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use serde_json::{json, Value};

/// Where a task is in its work. The desktop formats this; it does not compute it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Chrome,
    Read,
    Parse,
    Scene,
    Compute,
    Preview,
    Done,
    Failed,
    Cancelled,
}

impl Phase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Chrome => "chrome",
            Self::Read => "read",
            Self::Parse => "parse",
            Self::Scene => "scene",
            Self::Compute => "compute",
            Self::Preview => "preview",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
        }
    }
}

/// Progress for one poll. `total == 0` is an unknown length.
#[derive(Clone, Debug, PartialEq)]
pub struct Report {
    pub task: u64,
    pub generation: u64,
    pub phase: Phase,
    pub done: u64,
    pub total: u64,
    pub note: String,
}

impl Report {
    /// `None` while the length is unknown. Otherwise `done / total`, clamped.
    pub fn fraction(&self) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            Some((self.done as f64 / self.total as f64).clamp(0.0, 1.0))
        }
    }
}

/// The outcome of one slice. `Ready` is the finished value. A later `Ready` is
/// dropped when [`Cancel`] is set; see [`settle`].
#[derive(Clone, Debug, PartialEq)]
pub enum Poll<T> {
    Pending { report: Report, preview: Option<T> },
    Ready(T),
    Cancelled,
    Failed(String),
}

/// Shared cancel flag. Slices check it between units of work, not inside one.
#[derive(Clone, Debug)]
pub struct Cancel(Arc<AtomicBool>);

impl Cancel {
    pub fn new() -> Self {
        Self(Arc::new(AtomicBool::new(false)))
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}

impl Default for Cancel {
    fn default() -> Self {
        Self::new()
    }
}

/// A cancelled flag wins over a later `Ready` or preview.
pub fn settle<T>(poll: Poll<T>, cancel: &Cancel) -> Poll<T> {
    if cancel.is_cancelled() {
        Poll::Cancelled
    } else {
        poll
    }
}

/// Whether a report still belongs to the generation the view started.
pub fn generation_matches(report: &Report, task: u64, generation: u64) -> bool {
    report.task == task && report.generation == generation
}

/// Python `McRun` chunk rule. A wait longer than half the budget halves the
/// chunk, down to `floor`. A wait shorter than a tenth doubles it, up to `cap`.
pub fn adapt_chunk(chunk: u32, floor: u32, cap: u32, blocked: Duration, budget: Duration) -> u32 {
    let floor = floor.max(1);
    let cap = cap.max(floor);
    let chunk = chunk.clamp(floor, cap);
    if budget.is_zero() {
        return chunk;
    }
    let blocked = blocked.as_secs_f64();
    let budget = budget.as_secs_f64();
    if blocked > 0.5 * budget {
        (chunk / 2).max(floor)
    } else if blocked < 0.1 * budget {
        chunk.saturating_mul(2).min(cap)
    } else {
        chunk
    }
}

/// `u32` little-endian JSON length, the report JSON, then the payload bytes.
///
/// `apps/desktop/src/task-client.ts` reads this layout. Rename both together.
pub fn encode_poll(report: &Report, payload: Option<&[u8]>, finished: bool) -> Vec<u8> {
    let body = json!({
        "task": report.task,
        "generation": report.generation,
        "phase": report.phase.as_str(),
        "done": report.done,
        "total": report.total,
        "note": report.note,
        "finished": finished,
    });
    let json = serde_json::to_vec(&body).unwrap_or_else(|_| b"{}".to_vec());
    let mut out = Vec::with_capacity(4 + json.len() + payload.map(<[u8]>::len).unwrap_or(0));
    out.extend_from_slice(&(json.len() as u32).to_le_bytes());
    out.extend_from_slice(&json);
    if let Some(payload) = payload {
        out.extend_from_slice(payload);
    }
    out
}

/// The report JSON from [`encode_poll`]. The payload, when present, follows it.
pub fn decode_poll(bytes: &[u8]) -> Result<(Value, Option<&[u8]>), String> {
    if bytes.len() < 4 {
        return Err("task poll is truncated".into());
    }
    let length = u32::from_le_bytes(bytes[..4].try_into().unwrap()) as usize;
    let end = 4 + length;
    if bytes.len() < end {
        return Err("task poll is truncated".into());
    }
    let report: Value = serde_json::from_slice(&bytes[4..end]).map_err(|err| err.to_string())?;
    let payload = if bytes.len() > end {
        Some(&bytes[end..])
    } else {
        None
    };
    Ok((report, payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn report(done: u64, total: u64) -> Report {
        Report {
            task: 1,
            generation: 1,
            phase: Phase::Parse,
            done,
            total,
            note: String::new(),
        }
    }

    #[test]
    fn fraction_clamps_and_an_unknown_length_stays_indeterminate() {
        assert_eq!(report(1, 4).fraction(), Some(0.25));
        assert_eq!(report(5, 4).fraction(), Some(1.0));
        assert_eq!(report(0, 0).fraction(), None);
    }

    #[test]
    fn a_cancel_flag_wins_over_a_later_ready() {
        let cancel = Cancel::new();
        cancel.cancel();
        let settled = settle(Poll::Ready(7u32), &cancel);
        assert_eq!(settled, Poll::Cancelled);
    }

    #[test]
    fn a_mismatched_generation_is_droppable() {
        let report = report(1, 2);
        assert!(generation_matches(&report, 1, 1));
        assert!(!generation_matches(&report, 1, 2));
        assert!(!generation_matches(&report, 9, 1));
    }

    #[test]
    fn a_slow_wait_shrinks_the_chunk_and_a_fast_wait_grows_it() {
        let budget = Duration::from_millis(12);
        let shrunk = adapt_chunk(800, 100, 4_000, Duration::from_millis(9), budget);
        assert_eq!(shrunk, 400);
        let grown = adapt_chunk(800, 100, 4_000, Duration::from_millis(1), budget);
        assert_eq!(grown, 1_600);
        let held = adapt_chunk(800, 100, 4_000, Duration::from_millis(3), budget);
        assert_eq!(held, 800);
        let floored = adapt_chunk(100, 100, 4_000, budget, budget);
        assert_eq!(floored, 100);
        let capped = adapt_chunk(4_000, 100, 4_000, Duration::from_millis(0), budget);
        assert_eq!(capped, 4_000);
    }

    #[test]
    fn poll_bytes_round_trip_the_report_and_keep_the_payload() {
        let bytes = encode_poll(&report(2, 5), Some(b"plot"), false);
        let (value, payload) = decode_poll(&bytes).unwrap();
        assert_eq!(value["phase"], "parse");
        assert_eq!(value["done"], 2);
        assert_eq!(value["finished"], false);
        assert_eq!(payload, Some(b"plot".as_slice()));
    }
}
