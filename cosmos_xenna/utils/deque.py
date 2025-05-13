import collections
import typing

T = typing.TypeVar("T")


def pop_all_deque_elements(deque: collections.deque[T]) -> list[T]:
    out = []
    while deque:
        out.append(deque.pop())
    return out
