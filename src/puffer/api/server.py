"""``puffer serve``: the engine, the beacon and the API in one process.

Threads and ownership::

    main thread     uvicorn (asyncio loop: REST, /ws/live, live hub)
    engine thread   EntropyEngine.start(): camera -> vision -> pool, in real time
    beacon thread   BeaconService.run(stop): one signed pulse per period

Start order is engine, then beacon (the beacon draws from the engine's RNG),
then uvicorn. Shutdown is the reverse and happens whatever way uvicorn exits
(Ctrl-C, SIGTERM, a startup error): uvicorn first closes the WebSockets and
runs the app's lifespan shutdown (the hub stops broadcasting), then the
beacon is told to stop and joined, then the engine is closed (camera
released, LED off, statistics flushed), and finally the beacon store.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import uvicorn
from fastapi import FastAPI

from puffer.api.app import create_app
from puffer.api.settings import ApiSettings

if TYPE_CHECKING:
    from puffer.beacon.service import BeaconService
    from puffer.config import PufferConfig

logger = logging.getLogger(__name__)

__all__ = ["UVICORN_OPTIONS", "running", "serve"]

UVICORN_OPTIONS: Final[dict[str, Any]] = {
    "ws_max_size": 64 * 1024,  # clients never need to send more than a few bytes
    "ws_ping_interval": 20.0,  # notice half-open connections (laptop sleep, NAT timeouts)
    "ws_ping_timeout": 20.0,
    "server_header": False,  # do not advertise server software versions
    "timeout_graceful_shutdown": 5,  # a stuck client must not hold the camera hostage
}

_BEACON_JOIN_S: Final = 10.0


def _run_beacon(service: BeaconService, stop: threading.Event) -> None:
    """Beacon thread body. A beacon that fails closes; the API keeps serving its chain."""
    try:
        service.run(stop)
    except Exception:
        logger.critical("the beacon stopped and will not publish further pulses", exc_info=True)


@contextlib.contextmanager
def running(config: PufferConfig) -> Iterator[FastAPI]:
    """The engine and beacon running, plus an app serving them; all stopped on exit.

    Imports the runtime lazily: it pulls in OpenCV and the vision pipeline,
    which nothing else in :mod:`puffer.api` needs.
    """
    from puffer.runtime import build_beacon, build_engine

    engine = build_engine(config)
    stop = threading.Event()
    beacon: BeaconService | None = None
    thread: threading.Thread | None = None
    try:
        if config.beacon_enabled:
            beacon = build_beacon(config, engine)
        app = create_app(
            engine,
            beacon_service=beacon,
            static_dir=Path(config.static_dir),
            settings=ApiSettings.from_config(config),
        )
        engine.start()
        if beacon is not None:
            thread = threading.Thread(
                target=_run_beacon, args=(beacon, stop), name="puffer-beacon", daemon=True
            )
            thread.start()
        yield app
    finally:
        stop.set()
        if thread is not None:
            thread.join(timeout=_BEACON_JOIN_S)
        engine.close()
        if beacon is not None and not (thread is not None and thread.is_alive()):
            beacon.store.close()


def serve(config: PufferConfig) -> None:
    """Run engine, beacon and API until interrupted (``puffer serve``)."""
    with running(config) as app:
        logger.info("puffer serving on http://%s:%d", config.host, config.port)
        uvicorn.run(app, host=config.host, port=config.port, **UVICORN_OPTIONS)
