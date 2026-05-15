from __future__ import annotations

from typing import Any

from car_census import cli as _cli

app = _cli.app


def __getattr__(name: str) -> Any:
    return getattr(_cli, name)


if __name__ == "__main__":
    app()
