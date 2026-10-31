"""Serving the built dashboard (``dashboard/dist``) with sensible caching.

Vite emits content-hashed bundles under ``assets/`` (``index-BQJl5FOn.js``):
their name changes whenever their content does, so they are cached for a
year as ``immutable`` and never revalidated. Everything else - above all
``index.html``, which names the current bundles - is ``no-cache``: browsers
keep a copy but revalidate it (cheaply, via the ETag Starlette sets), so a
redeployed dashboard is picked up on the next load.

API and socket paths are never looked up on disk, so ``/api/typo`` is a JSON
404 rather than whatever file the build happens to contain.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Final

from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles
from starlette.types import Scope

__all__ = ["DashboardFiles", "dashboard_available"]

IMMUTABLE: Final = "public, max-age=31536000, immutable"
REVALIDATE: Final = "no-cache"
_HASHED_ASSET: Final = re.compile(r"assets/.+[-.][A-Za-z0-9_-]{8,}\.[A-Za-z0-9]+")
_RESERVED: Final = ("api", "ws")


def dashboard_available(static_dir: Path | None) -> bool:
    """True when ``static_dir`` holds a built dashboard (an ``index.html``)."""
    return static_dir is not None and (static_dir / "index.html").is_file()


def cache_control_for(relative_path: str) -> str:
    """Cache policy for a served file, by its path relative to the static root."""
    normalized = relative_path.replace(os.sep, "/")
    return IMMUTABLE if _HASHED_ASSET.fullmatch(normalized) else REVALIDATE


class DashboardFiles(StaticFiles):
    """``StaticFiles(html=True)`` plus cache headers and reserved API prefixes."""

    def __init__(self, directory: Path) -> None:
        super().__init__(directory=directory, html=True)
        self._root = Path(directory).resolve()

    async def get_response(self, path: str, scope: Scope) -> Response:
        if path.replace(os.sep, "/").split("/", 1)[0] in _RESERVED:
            raise HTTPException(status_code=404)
        return await super().get_response(path, scope)

    def file_response(
        self,
        full_path: str | os.PathLike[str],
        stat_result: os.stat_result,
        scope: Scope,
        status_code: int = 200,
    ) -> Response:
        response = super().file_response(full_path, stat_result, scope, status_code)
        try:
            relative = Path(full_path).resolve().relative_to(self._root).as_posix()
        except ValueError:  # pragma: no cover - StaticFiles never serves outside its root
            relative = ""
        response.headers["cache-control"] = cache_control_for(relative)
        return response
