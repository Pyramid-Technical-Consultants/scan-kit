"""Cycles are freed on the GUI thread, never in a worker that happens to allocate."""

from __future__ import annotations

import gc
import threading
import weakref


def test_worker_allocations_leave_cycles_for_the_gui_thread(qapp) -> None:
    from scan_kit.common.gui_gc import collect_due

    class Node:
        pass

    node = Node()
    node.me = node
    ref = weakref.ref(node)
    del node
    kept = []  # alive, so the allocation count stays past the threshold
    worker = threading.Thread(target=lambda: kept.extend([] for _ in range(50_000)))
    worker.start()
    worker.join()
    assert not gc.isenabled() and ref() is not None
    collect_due()
    assert ref() is None
