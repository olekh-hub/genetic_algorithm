from __future__ import annotations

from typing import Callable, TypeVar

T = TypeVar('T', bound=Callable)


class OperatorRegistry:
    """Named registry for GA operators (crossover, mutation, etc.).

    Usage::

        registry = OperatorRegistry()

        @registry.add('single_point')
        def single_point_crossover(p1, p2):
            ...

        fn = registry['single_point']
        registry.names  # ['single_point']
    """

    def __init__(self) -> None:
        self._ops: dict[str, Callable] = {}

    # -- registration ------------------------------------------------------

    def add(self, name: str) -> Callable[[T], T]:
        """Decorator that registers *fn* under *name*."""
        def decorator(fn: T) -> T:
            self._ops[name] = fn
            return fn
        return decorator

    # -- lookup ------------------------------------------------------------

    def __getitem__(self, name: str) -> Callable:
        if name not in self._ops:
            raise KeyError(
                f"Unknown operator '{name}'. "
                f"Available: {self.names}"
            )
        return self._ops[name]

    def __contains__(self, name: str) -> bool:
        return name in self._ops

    @property
    def names(self) -> list[str]:
        return list(self._ops.keys())

    def __repr__(self) -> str:
        return f"OperatorRegistry({self.names})"
