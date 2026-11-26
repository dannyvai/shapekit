"""User settings shared by the GUI, the CLI and child processes (e.g. the fly-through renderer).

Stored as JSON in %APPDATA%\\shapekit\\config.json (Windows) or ~/.config/shapekit/config.json.
"""
import json
import os
import sys


def path():
    if sys.platform == "win32":
        base = os.environ.get("APPDATA") or os.path.expanduser("~")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(os.path.expanduser("~"), ".config")
    return os.path.join(base, "shapekit", "config.json")


def load():
    try:
        with open(path(), encoding="utf-8") as f:
            d = json.load(f)
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def get(key, default=None):
    return load().get(key, default)


def set(key, value):
    """Set (or with value=None remove) one setting."""
    d = load()
    if value is None:
        d.pop(key, None)
    else:
        d[key] = value
    p = path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)
    os.replace(tmp, p)
