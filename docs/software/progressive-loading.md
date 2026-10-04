# Progressive loading

This is the engineering design for keeping the desktop responsive while session parses and GPU work finish. It is not an IEC 62304 architectural design. The crate rules in [architecture.md](architecture.md) still apply.

## What is slow today

`scan_kit_open_plot` builds the whole scene on one blocking thread and returns one payload. The webview drops a late response with `openSeq`, but the work still runs to the end. Monte Carlo in `scan-kit-compute` waits until every history is folded. The Python `McRun` already does the useful version of this: `step(budget)` queues a slice and returns while the GPU runs it, `preview` reads the mean dose so far, `extend` keeps the tallies, and the next slice grows or shrinks from how long the last wait blocked the caller. A sliced run matches a one-shot run because the tallies are integers and the fold gain is `1 / histories`.

## What this is

One task type. A view starts it, polls it, and cancels it. Each poll does a bounded slice of CPU or GPU work and may return a progress report and a partial plot payload. The webview draws that payload with the plot it already has. An untouched view refits to the data in hand. A zoom or pan stays when the new panel is the same quantity and the data span stays within 3×.

MCP and tests drain the same task to the end and still get one answer. `scan_kit_run_view` stays that drain. There is no second Monte Carlo, no second plot path, and no RGBA frame on the Tauri bridge.

## What this is not

- Not a job database, actor runtime, or thread-pool crate. One driver thread, `std::thread` joins, and the `wgpu` queue that compute already owns.
- Not streaming CSV into the webview. Partial results are plot payloads.
- Not a change to tally math. Slice size must not change the dose.
- Not a PyO3 bridge. The Python `McRun` stays until the dose view is ported, and the Rust task copies its contract.

## Types

These live in `scan-kit-core`. They do not touch `wgpu`, files, or Tauri.

```rust
pub struct Report {
    pub task: u64,
    pub generation: u64,
    pub phase: Phase,
    pub done: u64,
    pub total: u64, // 0 means the length is unknown
    pub note: String,
}

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

pub enum Poll<T> {
    Pending { report: Report, preview: Option<T> },
    Ready(T),
    Cancelled,
    Failed(String),
}
```

`Cancel` is an `Arc<AtomicBool>`. A slice checks it between units of work, not inside a dispatch. The in-flight dispatch finishes; its bytes are dropped when the generation no longer matches.

A task's `T` for the desktop is the existing plot payload (`Vec<u8>`). The final payload is the same encoding as today's `open_plot`. A partial payload sets `quality` to `partial` in the JSON header. The final payload sets `final`.

## Who runs it

```text
React face          formats Report, calls plot.load on each payload
Tauri               start, poll, cancel
scan-kit-compute    the driver thread and every GPU slice
scan-kit-io         parse, cache, and a scene from the columns ready so far
scan-kit-core       Report, Poll, Cancel, McJob
```

`scan-kit-compute` already depends on `scan-kit-io` and `wgpu`, so the driver lives there. `scan-kit-io` does not grow a second scheduler. A CPU-only tool (library index, runner copy) is still a task: its stage does no GPU work, and the same `poll` runs it.

The stage trait has one method:

```rust
fn poll(&mut self, budget: Duration, cancel: &Cancel) -> Poll<Vec<u8>>;
```

Views do not implement that trait in React. Each heavy operation exposes one constructor: `open_plot_task`, `mc_task`, `library_task`. Adding a view means adding a stage, not a new command protocol.

## Slice rules

Monte Carlo still grows its dispatch with the Python `McRun.step` rule. An analysis load keeps publishing a picture on every poll. The first picture is one window. Each later picture doubles that window, and each window reads only the bytes it adds.

- The UI poll budget is 12 ms of CPU time. A GPU submit returns without waiting.
- A Monte Carlo poll waits for the previous GPU fence. If that wait took more than half the budget, the next chunk is halved, down to a floor. If it took less than a tenth, the next chunk doubles, up to a cap. The floor and cap stay the ones already in the transport (dispatch size and in-flight depth).
- A CPU stage's first timed poll publishes one timeslice file, or 4096 rows of a long file, and reads only that window. A spot file uses the same row window. Each later timed poll publishes again, with twice as many of those windows as the last picture, up to 128. The hairline and the plot both advance on every poll, and a window does not read the file again from the start. A zero budget still publishes one window per poll. Workers for one task are joined on cancel and on `Ready`. No detached pool.
- A preview readback happens only when at least 100 ms have passed since the last one, and only after at least one unit of work. Readback is the expensive part; the submit is not.
- One device queue. Two GPU tasks do not submit together. A plot rebuild waits for the current MC slice fence, or cancels that task when the user left the view.
- Panic or device loss ends the task as `Failed`. The last good payload stays on screen. The next `start` creates the device again. A slice does not retry on its own.

## What the user sees first

A cache hit (the existing length and mtime stamp) skips to the final payload. A miss draws the first file, or the first 4096 rows of a long file, on the first poll, without reading the rest of that file. Later polls keep drawing a larger picture. The picture already on screen stays up, and can be zoomed or panned, while the next window is read. An untouched view refits as those windows arrive, so a later session stays inside the window. A zoom or pan the user already made is kept when the axes are still the same quantity and a close span.

Order for an analysis view:

1. `Chrome` — controls from the catalog or the last header cache.
2. `Read` — stamps. Cache hit jumps to `Done`.
3. `Parse` — one timeslice file, or 4096 rows when that file is long. A spot file uses the same row blocks. Emit the scene built from the rows in hand so the plot can be zoomed while the rest arrives.
4. `Scene` — each later poll appends a doubled window and publishes again.
5. `Compute` — gamma, DVH, splat, or transport, if this view has any.
6. `Done` — the same panel labels, `quality: final`.

A partial scene with no finite values uses the note panel the binned view already has. It is not marked final.

Switching Spot and Timeslice, or changing a control, starts a new generation and cancels the previous task. The picture on screen stays until the new generation publishes a payload, so the canvas does not flash empty. A report or payload whose generation does not match is ignored in the shell and in `plot.load`.

## Monte Carlo

`run_mc` becomes `mc_task`. The stage copies `McRun`:

- The first slice creates buffers and pipelines and publishes the empty grid so the axes exist.
- Each later slice settles the previous fence, queues the next chunk, and returns `Pending` with `done / total` histories.
- A preview fold (Python mode 2, gain `1 / histories_so_far`) runs on the 100 ms cadence and comes back as a plot payload. The grid does not change, so the camera stays.
- The last slice folds in store mode and publishes `Ready`. Uncertainty uses the same batch formula as a one-shot run.
- `extend` keeps the buffers when the grid, seed, and spots match and the new history count is higher. Progress becomes `already_done / new_total`. A lower target does nothing. A different phantom cancels and starts over.
- Closing the view or replacing the generation drops the buffers.

The Rust test that guards this is the Python one: a sliced 60k-history run equals a one-shot run of the same seed; raising 30k to 60k resumes at 0.5 and finishes on the same ledger.

## Commands

Three commands, shared by every view:

- `scan_kit_start(view, path, sessions, options, colors) -> { task, generation }`
- `scan_kit_poll(task) -> report, and a payload when this slice produced one`
- `scan_kit_cancel(task)`

`poll` is `spawn_blocking` around one `stage.poll(12ms)`. The webview calls it while the task is open and paints between calls, so a wheel or a source change is not stuck behind a parse. `scan_kit_open_plot` remains the drain of that task until the desktop hook replaces the call sites. After that it can stay as the one-shot helper for tests.

MCP does not grow a status tool. `scan_kit_run_view` polls with an unlimited budget until `Ready` or `Failed` and returns the final frame, as it does now.

## Faces

One hook, `useTask`, owns start, poll, cancel, and the generation check. A view only chooses how to draw the `Report`. It does not compute a fraction.

| View | First paint | Later slices | Face |
|---|---|---|---|
| Binned, distribution, timeline, trajectory, spectrum | Controls, then the first session | The other sessions on the same axes | Hairline, "2 of 5" |
| Timeslice files | First file's rows | The rest, by bytes read | Hairline, determinate |
| Dose, analytic | Coarse grid or the first spots | The remaining spots, same axes | Hairline |
| Dose, Monte Carlo | Empty grid after setup | Preview dose and uncertainty | Histories, "40% · ±1.4%" |
| Gamma, DVH, resample | Controls | One compute slice, unless the volume is split by plane | Hairline, indeterminate when `total` is 0 |
| Library index | Rows as folders finish | Notes filled in | "12 of 40" |
| Runner copy | Bytes copied | The same report the copy loop already has | Determinate, cancel sets the flag |

One hairline sits on the top edge of the window for every load: an analysis change (including Spot and Timeslice), the library index, a runner command or copy, plan and phantom synthesis, and configuration open, preview, and save. Unknown length sweeps. A known fraction fills. It hides when the load finishes, and it does not take clicks. It uses the stock accent token. No view invents its own thread or its own progress event.

## Tests

- Core: a fraction clamps, `total == 0` stays indeterminate, a cancelled flag wins over a later `Ready`, a mismatched generation is droppable.
- Compute, skipped with no adapter: sliced MC equals one-shot; extend from 30k to 60k; cancel after a preview publishes nothing further; a slow fake wait shrinks the next chunk and a fast wait grows it.
- IO: three small spot sessions at a zero budget emit two partial scenes and one final; a one-second budget still emits a partial before the final; an eight-file timeslice keeps publishing partials and finishes in fewer polls than one file at a time; a three-file timeslice grows the payload on each zero-budget poll; a cache hit emits one final slice; cancel after the first session joins the workers and does not emit the third.
- Desktop: a fake poll of partial then final calls `load` twice; a stale generation does not; the hairline treats `total == 0` as a sweep and a fraction as a width.
- Plot: the existing `adopt_view` tests. An untouched view refits when a later payload is wider. A zoomed view with the same labels and a close span keeps the zoom. A different quantity refits.

## Order of work

1. Add `Report`, `Poll`, and `Cancel`, and make `open_plot` a drain of a one-slice task. The bytes match today's payload.
2. Analysis scenes emit controls, then one payload per session. The desktop hook and the hairline land with that. Zoom behavior does not change.
3. `mc_task` with preview folds, the history face on dose volume, and the sliced-equals-one-shot test.
4. Library index and runner copy publish the same `Report`. Their private progress dialogs go away in that change.
