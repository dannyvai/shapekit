"""Desktop GUI. Launch: uv run python -m shapekit gui [PROJECT]"""


def run(project_dir=None):
    from .app import run as _run
    return _run(project_dir)
