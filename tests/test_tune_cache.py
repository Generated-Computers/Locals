"""The kernel tuning cache when two processes write it (`locals tune` next to a server tuning while idle): each
save merges with the file as it is, so neither erases the other's results."""

import json

from locals import app, kernels


def test_a_save_keeps_what_another_process_wrote(tmp_path, monkeypatch):
    path = tmp_path / "tune.json"
    monkeypatch.setattr(kernels, "_TUNE_PATH", str(path))
    monkeypatch.setattr(kernels, "_TUNE", {"2560x4096x8": ("frag", 2, 2, 1), "__times__": {"2560x4096x8": {"frag": 10.0}},
                                           "__model_tuned_pipe__x": [2, 3]})
    path.write_text(json.dumps({"9216x2560x8": ["mlx"], "__times__": {"9216x2560x8": {"mlx": 20.0}},
                                "__model_tuned_pipe__x": [3, 4]}))  # another process saved meanwhile
    kernels._save_tune()
    data = json.loads(path.read_text())
    assert data["2560x4096x8"] == ["frag", 2, 2, 1] and data["9216x2560x8"] == ["mlx"]
    assert set(data["__times__"]) == {"2560x4096x8", "9216x2560x8"}
    assert data["__model_tuned_pipe__x"] == [3, 4, 2]
    assert kernels._TUNE["9216x2560x8"] == ("mlx",)  # and this process uses the other's results from now on


def test_tune_refuses_while_a_server_runs(monkeypatch):
    monkeypatch.setattr(app, "_locals_server_pids", lambda: [4242])
    try:
        app.cmd_tune(["--model", "mlx-community/Qwen3.5-4B-MLX-4bit"])
    except SystemExit as e:
        assert "4242" in str(e) and "locals stop" in str(e)
    else:
        raise AssertionError("tuned next to a running server")
