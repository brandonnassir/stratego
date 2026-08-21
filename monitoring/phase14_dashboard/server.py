"""The localhost HTTP server, and the reasons it is as small as it is.

Read-only over HTTP as well as over the filesystem
--------------------------------------------------
:meth:`Handler.do_POST` and friends do not exist. Anything that is not a GET
gets 405 with a sentence pointing at the runbook, and a GET outside
:data:`ROUTES` gets 404, so the server has no route that could ever be given a
side effect by a later edit without that edit being obvious.

Bound to the loopback interface
-------------------------------
The default address is 127.0.0.1 and :func:`serve` refuses any other host
unless `--allow-nonlocal` is passed explicitly, because a passive monitor with
no authentication is fine on loopback and is not fine on a LAN.

Single-threaded on purpose
--------------------------
One browser and perhaps one monitoring agent are the entire expected load, a
request costs about a millisecond, and a serialised handler means the caches in
:class:`DashboardState` need no locking. Fewer moving parts is the feature.

Nothing runs between requests. An open dashboard with no browser attached
performs no reads at all, which is the cheapest possible answer to "reduce
refresh work when no client is connected".
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from .contract import DASHBOARD_VERSION, EXTERNAL_RUN_DIRECTORY
from .sources import RunPaths
from .state import DashboardState

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8714

LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")

PAGE = Path(__file__).with_name("index.html")


class DashboardError(RuntimeError):
    """Raised when the dashboard may not start as asked."""


class Handler(BaseHTTPRequestHandler):
    """GET-only. There is deliberately no do_POST, do_PUT or do_DELETE."""

    server_version = f"phase14-dashboard/{DASHBOARD_VERSION}"
    protocol_version = "HTTP/1.1"

    #: Every route the server has. A path outside this map is a 404 — the
    #: dashboard exposes no file browser and no arbitrary read of the disk.
    ROUTES = ("/", "/index.html", "/api/status", "/api/health", "/api/sources")

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        # A passive local monitor has no business being framed or sniffed.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, document: dict) -> None:
        body = json.dumps(document, default=str).encode("utf-8")
        self._send(code, body, "application/json; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's spelling
        path = self.path.split("?", 1)[0]
        if path not in self.ROUTES:
            self._json(404, {"error": f"no such route {path!r}", "routes": list(self.ROUTES)})
            return
        state: DashboardState = self.server.dashboard_state
        try:
            if path in ("/", "/index.html"):
                self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            elif path == "/api/status":
                self._json(200, state.status())
            elif path == "/api/health":
                # The cheap endpoint: enough for a monitoring agent to decide
                # whether anything needs a closer look, without the history.
                document = state.status()
                self._json(
                    200,
                    {
                        "artifact": DASHBOARD_VERSION,
                        "read_utc": document["read_utc"],
                        "overall": document["overall"],
                        "checks": {
                            name: check["status"]
                            for name, check in document["health"]["checks"].items()
                        },
                        "elapsed_hours": document["clock"].get("elapsed_hours"),
                        "remaining_hours": document["clock"].get("remaining_hours"),
                        "committed_games": document["games"]["committed_games"],
                        "optimizer_step": document["training"]["global_optimizer_step"],
                    },
                )
            else:
                self._json(200, state.status()["sources"])
        except BrokenPipeError:
            # The operator closed the tab mid-response. Not an incident.
            pass
        except Exception as error:  # noqa: BLE001
            # A dashboard that 500s is a dashboard that stopped telling the
            # operator anything, at the moment they were looking. Report the
            # failure as content instead.
            self._json(
                500,
                {
                    "error": f"{type(error).__name__}: {error}",
                    "note": "the dashboard failed to read; training is unaffected",
                },
            )

    def _refuse(self) -> None:
        self._json(
            405,
            {
                "error": "the Phase 14 dashboard is read-only and accepts GET only",
                "note": "recovery and control use the accepted supervisor; see PHASE_14_RUNBOOK.md",
            },
        )

    # HEAD is absent deliberately: the base class answers it 501, and a
    # handler that returned a body for HEAD would be wrong anyway.
    do_POST = do_PUT = do_PATCH = do_DELETE = _refuse  # noqa: N815

    def log_message(self, format, *args):  # noqa: A002 - the base class's name
        if self.server.verbose:
            sys.stderr.write("%s %s\n" % (self.address_string(), format % args))


class Dashboard(HTTPServer):
    """An HTTPServer carrying the shared read-only state."""

    allow_reuse_address = True

    def __init__(self, address, state: DashboardState, verbose: bool = False) -> None:
        super().__init__(address, Handler)
        self.dashboard_state = state
        self.verbose = verbose


def build(external_root=None, hot_root=None, ttl=None) -> DashboardState:
    return DashboardState(RunPaths(external_root, hot_root), ttl=ttl)


def serve(
    *,
    external_root=None,
    hot_root=None,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    allow_nonlocal: bool = False,
    verbose: bool = False,
) -> Dashboard:
    """Bind and return the server, without serving. The caller runs the loop."""
    if host not in LOCAL_HOSTS and not allow_nonlocal:
        raise DashboardError(
            f"refusing to bind {host!r}: the Phase 14 dashboard has no authentication "
            "and is intended for 127.0.0.1 only; pass --allow-nonlocal to override"
        )
    return Dashboard((host, int(port)), build(external_root, hot_root), verbose=verbose)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Phase 14 local read-only monitoring dashboard"
    )
    parser.add_argument(
        "--external-root",
        default=None,
        help=f"the run directory to watch (default: {EXTERNAL_RUN_DIRECTORY})",
    )
    parser.add_argument("--hot-root", default=None, help="the hot checkpoint ring")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--allow-nonlocal", action="store_true")
    parser.add_argument("--verbose", action="store_true", help="log every request")
    parser.add_argument(
        "--once", action="store_true", help="print one status document as JSON and exit"
    )
    args = parser.parse_args(argv)

    if args.once:
        print(
            json.dumps(
                build(args.external_root, args.hot_root).status(),
                indent=2,
                sort_keys=True,
                default=str,
            )
        )
        return 0

    server = serve(
        external_root=args.external_root,
        hot_root=args.hot_root,
        host=args.host,
        port=args.port,
        allow_nonlocal=args.allow_nonlocal,
        verbose=args.verbose,
    )
    paths = server.dashboard_state.paths
    sys.stderr.write(
        f"Phase 14 dashboard (read-only) on http://{args.host}:{server.server_address[1]}\n"
        f"  watching {paths.external_root}\n"
        f"  hot ring {paths.hot_root}\n"
        "  read-only: this process never writes to the run, and training does not\n"
        "  depend on it. Killing it affects nothing. Ctrl-C to stop.\n"
    )
    try:
        server.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        sys.stderr.write("\nstopped; training is unaffected\n")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
