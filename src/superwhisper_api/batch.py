"""Bounded-concurrency mapping shared by the batch CLIs and project runners."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

T = TypeVar("T")
R = TypeVar("R")


def bounded_map(
    items: Iterable[T],
    process: Callable[[T], R],
    *,
    max_workers: int,
) -> Iterator[tuple[T, R]]:
    """Run ``process(item)`` across a thread pool, yielding ``(item, result)``.

    Keeps at most ``max_workers * 4`` futures in flight so a large or streaming
    input is not submitted all at once. Results are yielded in completion order,
    each paired with the item that produced it. The caller owns all result
    handling (writing, counting, logging).
    """
    max_in_flight = max(max_workers * 4, max_workers)
    pending = iter(items)
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures: dict[Future[R], T] = {}

        def submit_next() -> bool:
            try:
                item = next(pending)
            except StopIteration:
                return False
            futures[pool.submit(process, item)] = item
            return True

        while len(futures) < max_in_flight and submit_next():
            pass
        while futures:
            done, _ = wait(set(futures), return_when=FIRST_COMPLETED)
            for fut in done:
                item = futures.pop(fut)
                yield item, fut.result()
                submit_next()
