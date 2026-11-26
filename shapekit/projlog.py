"""Per-project log: <project>/shapekit.log, always appended (never overwritten).

Every line of shapekit's log (util.log) is written with the wall-clock time, plus:
  ===== SESSION START ...        app / version / PC / GPU-independent facts, once per run
  TASK START  <title>            each GUI task (or CLI command)
  TASK DONE   <title>  (12m 03s) or  TASK FAILED <title> (...) + the traceback
  ===== SESSION END (clean exit)
A session without SESSION END was killed or crashed; the next session says so at its start, and with a
project open, native crash stacks (faulthandler: GPU driver, out of memory...) are written here too.
The file is rotated to shapekit.1.log once it passes 20 MB.
"""
import os
import platform
import sys
import threading
import time
import traceback

from . import util

NAME = "shapekit.log"
MAX_BYTES = 20 * 1024 * 1024
ENV_OFF = "SHAPEKIT_NO_PROJECT_LOG"          # set for child processes whose output the parent relays

_lock = threading.Lock()
_f = None
_path = None
_tasks = {}
_crash_file = None                          # app-wide crash.log (GUI): faulthandler goes back to it on close


def set_crash_file(f):
    global _crash_file
    _crash_file = f


def _stamp():
    t = time.time()
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t)) + ".%03d" % int((t % 1) * 1000)


def fmt_duration(s):
    s = int(round(s))
    if s < 60:
        return "%ds" % s
    if s < 3600:
        return "%dm %02ds" % (s // 60, s % 60)
    return "%dh %02dm %02ds" % (s // 3600, s % 3600 // 60, s % 60)


def write(text):
    """Write raw text (one or more lines) with the current time."""
    with _lock:
        if _f is None:
            return
        try:
            for line in str(text).rstrip("\n").splitlines() or [""]:
                _f.write("%s | %s\n" % (_stamp(), line))
            _f.flush()
        except (OSError, ValueError):
            pass


def _hook(line):
    write(line)


def _previous_session_unclean(p):
    """True if the last session in the log has a START without an END."""
    try:
        with open(p, "rb") as f:
            f.seek(max(0, os.path.getsize(p) - 256 * 1024))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return False
    i, j = tail.rfind("===== SESSION START"), tail.rfind("===== SESSION END")
    return i >= 0 and j < i


def open_log(project_dir, what="gui"):
    """Start appending to <project_dir>/shapekit.log (closes a log of another project first)."""
    global _f, _path
    if os.environ.get(ENV_OFF):
        return None
    p = os.path.join(os.path.abspath(project_dir), NAME)
    if _path and os.path.normcase(_path) == os.path.normcase(p):
        return p
    close_log("switched to another project")
    try:
        if os.path.exists(p) and os.path.getsize(p) > MAX_BYTES:
            os.replace(p, os.path.join(os.path.dirname(p), "shapekit.1.log"))
        unclean = _previous_session_unclean(p)
        f = open(p, "a", encoding="utf-8", buffering=1)
    except OSError as e:
        util.log("project log not available (%s)" % e)
        return None
    with _lock:
        _f, _path = f, p
    from . import __version__
    if unclean:
        write("!!!!! previous session has no SESSION END: it crashed, was killed or the PC shut down")
    write("===== SESSION START  %s  shapekit %s  pid %d  frozen=%s" % (what, __version__, os.getpid(),
                                                                       getattr(sys, "frozen", False)))
    write("      PC %s | %s | Python %s | %d CPUs" % (platform.node(), platform.platform(), platform.python_version(),
                                                     os.cpu_count() or 0))
    write("      project %s" % os.path.dirname(p))
    if _hook not in util.LOG_HOOKS:
        util.LOG_HOOKS.append(_hook)
    try:                                         # native crash stacks -> this log while the project is open
        import faulthandler
        faulthandler.enable(file=f, all_threads=True)
    except Exception:
        pass
    return p


def close_log(reason="clean exit"):
    global _f, _path
    if _f is None:
        return
    for title in list(_tasks):
        write("TASK ABORTED %s: still running when the session ended (%s)" % (title, reason))
        _tasks.pop(title, None)
    write("===== SESSION END (%s)" % reason)
    try:
        util.LOG_HOOKS.remove(_hook)
    except ValueError:
        pass
    with _lock:
        f, _f, _path = _f, None, None
    try:
        import faulthandler
        if _crash_file is not None:
            faulthandler.enable(file=_crash_file, all_threads=True)
        else:
            faulthandler.disable()
    except Exception:
        pass
    try:
        f.close()
    except OSError:
        pass


def path():
    return _path


def task_start(title, resume=None):
    _tasks[title] = time.time()
    write("TASK START  %s" % title)
    if _path:
        journal_start(title, resume)


def task_end(title, error=None):
    t0 = _tasks.pop(title, None)
    journal_end()
    d = " (%s)" % fmt_duration(time.time() - t0) if t0 else ""
    if error:
        err = error if isinstance(error, str) else "".join(traceback.format_exception(error))
        first = err.strip().splitlines()[-1] if err.strip() else "error"
        write("TASK FAILED %s%s: %s" % (title, d, first))
        if "\n" in err.strip():
            write(err)
    else:
        write("TASK DONE   %s%s" % (title, d))


# ---------------------------------------------------------------------------------------------- crash recovery
# While a task runs, <project>/.shapekit_running.json describes it (title, batch step, how to resume).
# It is removed when the task ends - also when it fails with a Python error (that is reported, not a crash).
# If it is still there when the project is opened again, and the process that wrote it is gone, the app
# crashed (or was killed / the PC shut down) in that task.
JOURNAL = ".shapekit_running.json"


def _journal_path(project_dir=None):
    d = project_dir or (os.path.dirname(_path) if _path else None)
    return os.path.join(d, JOURNAL) if d else None


def _pid_alive(pid):
    if not pid or pid == os.getpid():
        return False
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x1000, False, int(pid))           # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = k.GetExitCodeProcess(h, ctypes.byref(code))
        k.CloseHandle(h)
        return bool(ok) and code.value == 259                # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def _journal_write(d):
    p = _journal_path()
    if not p:
        return
    import json
    try:
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=1, default=str)
        os.replace(tmp, p)
    except OSError:
        pass


_journal = None


def journal_start(title, resume=None):
    """Mark `title` as running. resume: JSON-able dict describing how to continue it (GUI action, params...)."""
    global _journal
    _journal = dict(title=title, started=time.strftime("%Y-%m-%d %H:%M:%S"), t0=time.time(), pid=os.getpid(),
                    stage=None, resume=resume or {})
    _journal_write(_journal)


def journal_update(**kw):
    """e.g. journal_update(stage="Build Mesh") from a batch run, or progress of multi-file imports."""
    if _journal is None:
        return
    for k, v in kw.items():
        if k == "stage":
            _journal["stage"] = v
            write("      step: %s" % v)
        else:
            _journal["resume"][k] = v
    _journal_write(_journal)


def journal_end():
    global _journal
    _journal = None
    p = _journal_path()
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass


def crashed_task(project_dir):
    """The task that was running when the app last died in this project, or None."""
    p = _journal_path(project_dir)
    if not p or not os.path.exists(p):
        return None
    import json
    try:
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
    except (OSError, ValueError):
        return None
    if _pid_alive(d.get("pid")):               # still running in another shapekit window / process
        return None
    d["ran_for"] = fmt_duration(max(0.0, os.path.getmtime(p) - float(d.get("t0") or os.path.getmtime(p))))
    return d


def clear_crashed(project_dir):
    p = _journal_path(project_dir)
    if p and os.path.exists(p):
        try:
            os.remove(p)
        except OSError:
            pass
