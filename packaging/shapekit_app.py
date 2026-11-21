"""Entry point of the portable build.

ShapeKit.exe              -> desktop GUI (optionally: ShapeKit.exe PROJECT_FOLDER)
shapekit-cli.exe ARGS...  -> command line (same commands as `python -m shapekit`)
"""
import multiprocessing
import os
import sys


def _setup():
    base = os.path.dirname(sys.executable) if getattr(sys, "frozen", False) else os.path.dirname(os.path.abspath(__file__))
    models = os.path.join(base, "models")
    if os.path.isdir(models):
        os.environ.setdefault("TORCH_HOME", models)          # bundled network weights -> works offline
    os.environ.setdefault("QT_API", "pyside6")
    return base


def _redirect_logs(base):
    """Windowed exe has no console: send prints / tracebacks to a log file instead."""
    if sys.stdout is not None and sys.stderr is not None:
        return
    d = os.path.join(os.environ.get("LOCALAPPDATA", base), "ShapeKit")
    try:
        os.makedirs(d, exist_ok=True)
        f = open(os.path.join(d, "shapekit.log"), "a", buffering=1, encoding="utf-8")
    except OSError:
        f = open(os.devnull, "w")
    sys.stdout = sys.stdout or f
    sys.stderr = sys.stderr or f


def main():
    base = _setup()
    exe = os.path.splitext(os.path.basename(sys.executable))[0].lower()
    cli = "cli" in exe or (len(sys.argv) > 1 and sys.argv[1] in ("--cli",))
    if cli:
        if len(sys.argv) > 1 and sys.argv[1] == "--cli":
            del sys.argv[1]
        from shapekit.cli import main as cli_main
        return cli_main()
    _redirect_logs(base)
    from shapekit.gui import run
    project = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else None
    return run(project)


if __name__ == "__main__":
    multiprocessing.freeze_support()            # process pools (depth filtering, markers) re-launch the exe
    sys.exit(main())
