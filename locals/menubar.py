"""locals in the macOS menu bar: the server's state at a glance, start and stop, and the
tools opened on it, without a terminal.

    locals menubar          (needs rumps: uv tool install --python 3.12 --force 'locals[menubar]')

The icon shows whether the server runs (●), is starting (◐) or is stopped (○); the menu
shows the model and the requests running, and opens OpenCode or the status page.
Starting and stopping go through the same commands as the terminal (locals start /
locals stop), so both stay in step.
"""

from __future__ import annotations

import subprocess
import sys
import threading
import webbrowser


def _osascript_terminal(command: str) -> None:
    """Open Terminal with ``command`` (the tools are terminal programs)."""
    quoted = command.replace("\\", "\\\\").replace('"', '\\"')  # an AppleScript string: a folder's name cannot end it
    script = f'tell application "Terminal" to do script "{quoted}"\ntell application "Terminal" to activate'
    subprocess.run(["osascript", "-e", script], check=False)


def main(argv=None) -> None:
    import argparse

    argparse.ArgumentParser(prog="locals menubar", description="A menu-bar item for the locals server: its status, "
                            "start and stop, the status page and the coding tools. It runs until you quit it from its "
                            "menu.").parse_args(sys.argv[2:] if argv is None else argv)  # (--help started the app)
    try:
        import rumps
    except ImportError:
        sys.exit("the menu-bar app needs rumps; add it to the locals install with:\n"
                 "  uv tool install --python 3.12 --force 'locals[menubar]'")
    from .app import DEFAULT_PORT, _load_saved, base_url, get_json, server_alive

    saved = _load_saved()
    port = saved.get("port", DEFAULT_PORT)
    locals = [sys.executable, "-m", "locals.app"]

    def _open_folder():
        """OpenCode on a folder the user picks (its agents may edit files and run commands there)."""
        import shlex

        p = subprocess.run(["osascript", "-e", 'POSIX path of (choose folder with prompt "Open OpenCode on which project?")'],
                           capture_output=True, text=True)
        folder = p.stdout.strip()
        if p.returncode == 0 and folder:
            # this Python, not a `locals` the new Terminal may not find on its PATH
            _osascript_terminal(f"cd {shlex.quote(folder)} && {' '.join(shlex.quote(a) for a in locals)} code .")

    class App(rumps.App):
        def __init__(self):
            super().__init__("locals", title="locals ○", quit_button="Quit")
            self.state = rumps.MenuItem("stopped")
            self.toggle = rumps.MenuItem("Start the server", callback=self.on_toggle)
            self.menu = [self.state, None, self.toggle,
                         rumps.MenuItem("Open OpenCode on a folder…", callback=lambda _: _open_folder()),
                         rumps.MenuItem("Status page", callback=lambda _: webbrowser.open(base_url(port) + "/")), None]
            self.busy = False
            rumps.Timer(self.refresh, 2).start()

        def refresh(self, _=None):
            if self.busy:
                return
            if getattr(self, "failed", ""):
                rumps.alert("locals", self.failed)
                self.failed = ""
            if not server_alive(port):
                self.title, self.state.title, self.toggle.title = "locals ○", "stopped", "Start the server"
                return
            try:
                s = get_json(base_url(port) + "/stats", timeout=2)
                model, active = s.get("model", "?").split("/")[-1], len(s.get("active", []))
                self.state.title = f"{model}: {active} request(s) running"
            except Exception:  # noqa: BLE001
                self.state.title = "running"
            self.title, self.toggle.title = "locals ●", "Stop the server"

        def on_toggle(self, _):
            running = server_alive(port) is not None
            self.busy = True
            self.title = "locals ◐"
            self.state.title = "stopping…" if running else "starting… (the first start downloads the model)"

            def work():  # the menu is updated by the timer, on the main thread (AppKit is not thread-safe)
                r = subprocess.run(locals + (["stop"] if running else ["start"]), capture_output=True, text=True)
                self.failed = "" if r.returncode == 0 else (r.stderr or r.stdout).strip()[-300:]
                self.busy = False

            threading.Thread(target=work, daemon=True).start()

    App().run()


if __name__ == "__main__":
    main()
