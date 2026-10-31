"""The ``puffer`` Typer application: global options, error handling, command table.

Error handling lives in one place, :class:`PufferGroup`, which wraps the
invocation of every (sub)command:

* :class:`~puffer.cli.errors.CliError` and every library exception listed in
  :func:`~puffer.cli.errors.translate` become a one-line ``error:`` message,
  an optional hint, and the documented exit code.
* ``Ctrl-C`` exits with 130 and a short note instead of a traceback.
* Anything else is an internal error: one line, plus the traceback only
  with ``-v``. Typer's "pretty exceptions" are disabled on purpose - they
  can print local variables, and in this program locals hold keys and
  passwords.
"""

from __future__ import annotations

import logging
import traceback
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.logging import RichHandler
from rich.text import Text
from typer.core import TyperGroup

from puffer import __version__
from puffer.cli.commands import beacon, fish, info, keys, rand, serve, simulate
from puffer.cli.context import PASSWORD_ENV, CliState, GlobalOptions, LivenessMode
from puffer.cli.errors import EXIT_CODE_HELP, CliError, ExitCode, translate

__all__ = ["PufferGroup", "app"]


def _report(error: CliError, console: Console) -> None:
    message = Text("error: ", style="bold red")
    message.append(error.message)
    console.print(message)
    if error.details:
        console.print(Text(error.details, style="dim"))
    if error.hint:
        console.print(Text(f"hint: {error.hint}", style="yellow"))


def _verbosity(ctx: Any) -> int:
    obj = ctx.find_root().obj if ctx is not None else None
    return obj.verbose if isinstance(obj, CliState) else 0


class PufferGroup(TyperGroup):
    """Root command group that maps exceptions to messages and exit codes."""

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except (typer.Exit, typer.Abort, typer.TyperException):
            raise  # Typer's own control flow and usage errors
        except KeyboardInterrupt:
            Console(stderr=True).print(Text("interrupted", style="yellow"))
            raise typer.Exit(ExitCode.INTERRUPTED) from None
        except Exception as exc:
            console = Console(stderr=True, highlight=False)
            error = translate(exc)
            if error is None:
                if _verbosity(ctx) > 0:
                    console.print(Text("".join(traceback.format_exception(exc)), style="dim"))
                error = CliError(
                    f"internal error: {type(exc).__name__}: {exc}",
                    ExitCode.FAILURE,
                    hint=None if _verbosity(ctx) else "re-run with -v for a traceback",
                )
            elif _verbosity(ctx) > 1:
                console.print(Text("".join(traceback.format_exception(exc)), style="dim"))
            _report(error, console)
            raise typer.Exit(error.code) from None


app = typer.Typer(
    name="puffer",
    cls=PufferGroup,
    help="Cryptographic randomness from live fish: camera sensor noise and fish motion, "
    "mixed with os.urandom, health-tested (SP 800-90B), liveness-checked with an LED "
    "challenge, and published as a signed beacon.",
    epilog=f"{EXIT_CODE_HELP}  Environment: PUFFER_CONFIG (config file), PUFFER_<FIELD> "
    f"(any setting, e.g. PUFFER_SOURCE), {PASSWORD_ENV} (password for --password).",
    no_args_is_help=True,
    add_completion=False,
    pretty_exceptions_enable=False,
    rich_markup_mode="rich",
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"puffer-cipher {__version__}")
        raise typer.Exit()


def _configure_logging(verbose: int) -> None:
    level = logging.WARNING if verbose == 0 else logging.INFO if verbose == 1 else logging.DEBUG
    handler = RichHandler(
        console=Console(stderr=True), show_time=verbose > 1, show_path=verbose > 1
    )
    root = logging.getLogger()
    for existing in list(root.handlers):
        if isinstance(existing, RichHandler):
            root.removeHandler(existing)
    root.addHandler(handler)
    root.setLevel(level)


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Annotated[
        Path | None,
        typer.Option(
            "--config",
            metavar="PATH",
            exists=True,
            dir_okay=False,
            help="TOML configuration file (\\[puffer] table) \\[default: $PUFFER_CONFIG].",
        ),
    ] = None,
    data_dir: Annotated[
        Path | None,
        typer.Option(
            "--data-dir",
            metavar="DIR",
            file_okay=False,
            help="Data directory (puffer.db, beacon.key).",
        ),
    ] = None,
    source: Annotated[
        str | None,
        typer.Option(
            "--source",
            metavar="SPEC",
            help="Frame source: synthetic[:seed=7,...], video:tank.mp4, camera:0, picamera.",
        ),
    ] = None,
    liveness: Annotated[
        LivenessMode | None,
        typer.Option(
            "--liveness",
            case_sensitive=False,
            help="LED liveness policy (auto: required for cameras and the simulator, off "
            "for video files).",
        ),
    ] = None,
    verbose: Annotated[
        int, typer.Option("-v", "--verbose", count=True, help="More logging (-vv: debug).")
    ] = 0,
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_version_callback,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Global options; each overrides the matching configuration field."""
    _configure_logging(verbose)
    ctx.obj = CliState(
        GlobalOptions(
            config_path=config,
            data_dir=data_dir,
            source=source,
            liveness=liveness,
            verbose=verbose,
        )
    )


app.command("keygen")(keys.keygen)
app.command("encrypt")(keys.encrypt)
app.command("decrypt")(keys.decrypt)
app.command("random")(rand.random)
app.command("simulate")(simulate.simulate)
app.command("serve")(serve.serve)
app.add_typer(beacon.beacon_app, name="beacon")
app.command("analyze")(serve.analyze)
app.command("fish")(fish.fish)
app.command("info")(info.info)
