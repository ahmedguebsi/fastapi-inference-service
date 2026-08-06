"""Console entry point: ``python -m inference_service`` / ``inference-service``.

Uvicorn is configured here rather than via a CLI invocation so that the
timeouts that matter for correctness — keep-alive and graceful shutdown — come
from the same validated config object the application uses, instead of being
duplicated in a Dockerfile CMD where they silently drift.
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from . import __version__
from .config import ConfigError, load_config
from .logging_setup import configure_logging, get_logger

logger = get_logger("entrypoint")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="inference-service",
        description="Run the inference service HTTP server.",
    )
    parser.add_argument("--version", action="version", version=f"inference-service {__version__}")
    parser.add_argument("-c", "--config", metavar="PATH", help="path to service.toml")
    parser.add_argument("--host", help="override server.host")
    parser.add_argument("--port", type=int, help="override server.port")
    parser.add_argument("--workers", type=int, help="override server.workers")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="auto-reload on code changes (development only)",
    )
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="validate configuration and exit without starting the server",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"inference-service: configuration error: {exc}", file=sys.stderr)
        return 2

    host = args.host or config.server.host
    port = args.port or config.server.port
    workers = args.workers or config.server.workers

    configure_logging(
        config.logging,
        service_name=config.service_name,
        environment=config.environment,
    )

    if args.check_config:
        logger.info(
            "configuration is valid",
            extra={"source": config.source_path or "<defaults>", "host": host, "port": port},
        )
        return 0

    # Multiple workers means multiple processes, each with its own model copy
    # and its own batching queue. That multiplies memory by the worker count and
    # fragments batches across processes, so batches fill more slowly. Prefer
    # scaling replicas over workers for a model server; warn rather than refuse,
    # because a small model with a slow tokenizer can legitimately want both.
    if workers > 1:
        logger.warning(
            "running multiple workers: each loads its own model copy and keeps a "
            "separate batch queue, so memory scales linearly and batches fill more "
            "slowly. Prefer horizontal replicas for model serving.",
            extra={"workers": workers},
        )

    uvicorn.run(
        "inference_service.main:create_app",
        factory=True,
        host=host,
        port=port,
        workers=workers if not args.reload else 1,
        reload=args.reload,
        # Must exceed the load balancer's idle timeout, or the server closes
        # connections the LB still considers open and clients see random 502s.
        timeout_keep_alive=config.server.keep_alive_timeout,
        # Bounds how long uvicorn waits for in-flight requests during shutdown
        # before force-closing. Must be under the orchestrator's kill grace.
        timeout_graceful_shutdown=int(config.server.graceful_shutdown_timeout),
        # We install our own structured handlers; letting uvicorn configure
        # logging would overwrite them with its colourised defaults.
        log_config=None,
        access_log=False,
        server_header=False,  # do not advertise the server implementation
        date_header=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
