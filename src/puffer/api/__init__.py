"""HTTP and WebSocket interface: FastAPI REST + the live feed (DESIGN.md section 7).

* :func:`~puffer.api.app.create_app` - the ASGI app around an engine
  (REST endpoints, ``/ws/live``, static dashboard, security headers).
* :func:`~puffer.api.server.serve` - ``puffer serve``: engine, beacon and
  uvicorn in one process with an orderly shutdown.
* :mod:`~puffer.api.wire` - every JSON shape the dashboard decodes, built in
  one place.
* :mod:`~puffer.api.hub` - the broadcaster behind ``/ws/live``.
* :mod:`~puffer.api.ports` - the structural interface the API needs from the
  engine, so tests can drive it with fakes.
"""

from puffer.api.app import create_app
from puffer.api.server import serve
from puffer.api.settings import ApiSettings

__all__ = ["ApiSettings", "create_app", "serve"]
