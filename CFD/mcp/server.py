"""MCP server that lets Claude Code drive Ansys Fluent through PyFluent on the DLL remote PCs.

It exposes the 13 tools of 2026_C_FD_REPORT-2_MCP-FLUENT (Section 8), so the guide's workflow
applies: read the case in Fluent, run server/start-server in its console, then connect_to_fluent.
bootstrap-vm.ps1 installs the packages and registers this file in the project .mcp.json.

stdout carries the MCP messages, so nothing else may write to it. Fluent's console output is taken
from the PyFluent transcript instead: the tools that run commands return the end of it, and all of
it goes to fluent-transcript-<time>.log in the run folder. A Fluent started by launch_fluent gets
no stdout either.

Written for ansys-fluent-core 0.43 and mcp 2.3 (settings API of Fluent 2024 R2 and 2026 R1).
"""
from __future__ import annotations

import functools
import os
import subprocess
import threading
import time
from pathlib import Path

import ansys.fluent.core as pyfluent
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

# os.path.abspath, not Path.resolve: resolve turns the Z:\ drive into a UNC path, which cmd.exe
# (used by PyFluent to start Fluent) cannot use as a working directory.
RUNS = Path(os.path.abspath(__file__)).parent.parent / "runs"
TAIL_LINES = 60

pyfluent.config.launch_fluent_stdout = subprocess.DEVNULL
pyfluent.config.launch_fluent_stderr = subprocess.DEVNULL

server = MCPServer(
    "fluent",
    instructions=(
        "Drives one Ansys Fluent solver session. Attach first with connect_to_fluent (Fluent "
        "already open, after server/start-server in its console) or launch_fluent. Geometry and "
        "meshing stay manual. execute_tui and scheme_eval can change the case: say what a "
        "command does before running it."
    ),
)

_session = None
_launched = False  # True when this server started the Fluent process
_console: list[str] = []
_console_lock = threading.Lock()


def _on_transcript(text: str) -> None:
    with _console_lock:
        _console.append(text)


def _attach(session, run_dir: Path, launched: bool) -> None:
    global _session, _launched
    _session, _launched = session, launched
    session.transcript.register_callback(_on_transcript, keep_new_lines=True)
    log = run_dir / f"fluent-transcript-{time.strftime('%Y%m%d-%H%M%S')}.log"
    session.transcript.start(file_name=str(log), write_to_stdout=False)


def _solver():
    if _session is None:
        raise RuntimeError("Not attached to Fluent: call connect_to_fluent or launch_fluent first.")
    if not _session.is_active():
        raise RuntimeError("The Fluent session is gone (closed or crashed). Attach again.")
    return _session


def _run(action) -> tuple[object, str]:
    """Runs action() and returns its result with the end of the console text it produced."""
    with _console_lock:
        _console.clear()
    result = action()
    time.sleep(0.5)  # the transcript stream lags the command slightly
    with _console_lock:
        lines = "".join(_console).splitlines()
    if len(lines) > TAIL_LINES:
        lines = [f"[... {len(lines) - TAIL_LINES} earlier lines in the transcript log]"] + lines[-TAIL_LINES:]
    return result, "\n".join(lines)


def _report(headline: str, console: str) -> str:
    return f"{headline}\n\nConsole:\n{console}" if console.strip() else headline


def _tool(fn):
    """Registers fn as a tool whose errors reach Claude with their message (mcp 2.x replaces the
    text of any exception other than ToolError with a generic one)."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ToolError:
            raise
        except Exception as exc:
            raise ToolError(f"{type(exc).__name__}: {exc}") from exc
    return server.tool()(wrapper)


@_tool
def launch_fluent(dimension: int, processor_count: int = 4, precision: str = "double",
                  show_gui: bool = True, working_dir: str = "") -> str:
    """Start a new Fluent solver session and attach to it.

    dimension: 2 (also for axisymmetric cases) or 3. working_dir: where Fluent runs and writes its
    files, by default CFD/runs of the repository. The newest installed Fluent is used.
    """
    run_dir = Path(working_dir) if working_dir else RUNS
    run_dir.mkdir(parents=True, exist_ok=True)
    session = pyfluent.launch_fluent(
        dimension=dimension, precision=precision, processor_count=processor_count,
        ui_mode="gui" if show_gui else "no_gui", cwd=str(run_dir),
        start_transcript=False, cleanup_on_exit=True,
    )
    _attach(session, run_dir, launched=True)
    return f"Started {session.get_fluent_version()} ({dimension}D, {processor_count} processes) in {run_dir}."


@_tool
def connect_to_fluent(server_info_file: str) -> str:
    """Attach to a Fluent session started with server/start-server in its console.

    server_info_file: the server_info-<pid>.txt Fluent wrote, or the folder that holds it (the
    newest file there is used). Run pwd in the Fluent console if you cannot find it.
    """
    path = Path(server_info_file)
    if path.is_dir():
        files = sorted(path.glob("server_info-*.txt"), key=lambda p: p.stat().st_mtime)
        if not files:
            raise FileNotFoundError(f"No server_info-*.txt in {path}.")
        path = files[-1]
    common = dict(cleanup_on_exit=False, start_transcript=False)
    address, password = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()[:2]]
    ip, _, port = address.rpartition(":")
    if address.startswith("unix:") or ip in ("127.0.0.1", "localhost", "[::1]"):
        session = pyfluent.connect_to_fluent(server_info_file_name=str(path), **common)
    else:
        # Fluent may write the PC's network address instead of localhost. PyFluent accepts that
        # only as an explicit insecure connection, so try the loopback address first: Fluent runs
        # on this same PC.
        try:
            session = pyfluent.connect_to_fluent(ip="127.0.0.1", port=int(port), password=password, **common)
        except Exception:
            session = pyfluent.connect_to_fluent(ip=ip, port=int(port), password=password,
                                                 allow_remote_host=True, insecure_mode=True, **common)
    _attach(session, path.parent, launched=False)
    return f"Connected to {session.get_fluent_version()} through {path.name}."


@_tool
def read_case(path: str) -> str:
    """Read a case file (.cas.h5 or .cas)."""
    _, console = _run(lambda: _solver().settings.file.read_case(file_name=path))
    return _report(f"Read {path}.", console)


@_tool
def read_case_and_data(path: str) -> str:
    """Read a case and its data file together (give the case file name)."""
    _, console = _run(lambda: _solver().settings.file.read_case_data(file_name=path))
    return _report(f"Read {path} with its data.", console)


@_tool
def initialize(method: str = "hybrid") -> str:
    """Initialise the solution: 'hybrid' or 'standard' (from the values set under Solution
    Initialization). For FMG, run a standard initialisation and then execute_tui."""
    init = _solver().settings.solution.initialization
    if method == "hybrid":
        _, console = _run(init.hybrid_initialize)
    elif method == "standard":
        _, console = _run(init.standard_initialize)
    else:
        raise ValueError("method must be 'hybrid' or 'standard'.")
    return _report(f"{method.capitalize()} initialisation done.", console)


@_tool
def iterate(count: int) -> str:
    """Run `count` iterations and return the end of the console output (the residual lines).

    Run at most about 500 at a time and check the monitors between calls: a call blocks until the
    iterations are done."""
    start = time.time()
    _, console = _run(lambda: _solver().settings.solution.run_calculation.iterate(iter_count=count))
    return _report(f"Ran {count} iterations in {time.time() - start:.0f} s.", console)


@_tool
def execute_tui(command: str) -> str:
    """Run one Fluent text-interface command, e.g. '/solve/monitors/residual/print? yes', and
    return its console output. Paths with spaces go in double quotes."""
    _, console = _run(lambda: _solver().execute_tui(command))
    return _report(f"Ran {command}", console)


@_tool
def scheme_eval(code: str) -> str:
    """Evaluate a Scheme expression in Fluent, e.g. "(rpgetvar 'flow-time)", and return its value
    and any console output."""
    value, console = _run(lambda: _solver().scheme.eval(code))
    return _report(f"Value: {value!r}", console)


@_tool
def get_case_summary() -> str:
    """Models, reference values, report definitions, boundary zones and mesh size of the case."""
    solver = _solver()
    s = solver.settings
    facts = {
        "Fluent": solver.get_fluent_version,
        "solver": s.setup.general.solver.type,
        "2D space": s.setup.general.solver.two_dim_space,
        "time": s.setup.general.solver.time,
        "viscous model": s.setup.models.viscous.model,
        "k-omega model": s.setup.models.viscous.k_omega_model,
        "energy": s.setup.models.energy.enabled,
        "operating pressure [Pa]": s.setup.general.operating_conditions.operating_pressure,
        "reference area [m2]": s.setup.reference_values.area,
        "reference length [m]": s.setup.reference_values.length,
    }
    lines = []
    for name, get in facts.items():
        try:
            lines.append(f"{name}: {get()}")
        except Exception:
            pass  # setting not active in this case, e.g. 2D space in a 3D case
    reports = s.solution.report_definitions
    for kind in ("drag", "lift", "force", "moment", "surface"):
        try:
            names = getattr(reports, kind).get_object_names()
        except Exception:
            continue
        if names:
            lines.append(f"{kind} reports: {', '.join(names)}")
    _, zones = _run(lambda: solver.execute_tui("/define/boundary-conditions/list-zones"))
    _, size = _run(lambda: solver.execute_tui("/mesh/size-info"))
    return "\n".join(lines) + f"\n\nZones:\n{zones}\n\nMesh:\n{size}"


@_tool
def get_residuals_summary(monitor_set: str = "residual", last: int = 10) -> str:
    """Last values of a monitor set: 'residual' (scaled residuals) or a report plot such as
    'cd-nose-rplot'. Gives the change over that window too, for a convergence check."""
    monitors = _solver().monitors
    if not monitors.is_streaming:
        monitors.start()
    names = monitors.get_monitor_set_names()
    if monitor_set not in names:
        return f"No monitor set {monitor_set!r}. Available: {', '.join(names) or 'none (initialise or iterate first)'}."
    iterations, series = monitors.get_monitor_set_data(monitor_set)
    if len(iterations) == 0:
        return f"{monitor_set}: no data yet."
    lines = [f"{monitor_set}, iterations {int(iterations[-last:][0])}-{int(iterations[-1])}:"]
    for name, values in series.items():
        window = values[-last:]
        change = abs(window[-1] - window[0]) / abs(window[-1]) if window[-1] else float("nan")
        lines.append(f"  {name}: last {window[-1]:.6g}, change over the window {100 * change:.3g} %")
    lines.append(f"Other monitor sets: {', '.join(n for n in names if n != monitor_set) or 'none'}")
    return "\n".join(lines)


@_tool
def write_data(path: str, with_case: bool = False) -> str:
    """Save the data file, or case and data together with with_case=true (give the case file name
    then)."""
    file = _solver().settings.file
    if with_case:
        _, console = _run(lambda: file.write_case_data(file_name=path))
    else:
        _, console = _run(lambda: file.write_data(file_name=path))
    return _report(f"Wrote {path}.", console)


@_tool
def screenshot(path: str) -> str:
    """Save the active graphics window as an image (.png). Needs the Fluent GUI."""
    _, console = _run(lambda: _solver().settings.results.graphics.picture.save_picture(file_name=path))
    return _report(f"Saved {path}.", console)


@_tool
def exit_fluent(close_fluent: bool = False) -> str:
    """Detach from Fluent. A Fluent started by launch_fluent is closed. One you connected to keeps
    running unless close_fluent is true, which loses unsaved data."""
    global _session
    if _session is None:
        return "Not attached."
    if _launched or close_fluent:
        _session.exit()
        message = "Fluent closed."
    else:
        _session.transcript.stop()
        message = "Detached; Fluent keeps running."
    _session = None
    return message


if __name__ == "__main__":
    server.run("stdio")
