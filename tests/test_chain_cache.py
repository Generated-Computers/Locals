"""The prompt cache drops the stale copies of a conversation where the cache cannot be cut back:
earlier replies and checkpoints a newer entry extends, never system-prompt entries, and not the
checkpoint of the request whose reply just finished."""

import pytest

pytest.importorskip("mlx_lm")

from mlx_lm.models.cache import ArraysCache  # noqa: E402

from locals.hybrid_batch import chain_cache  # noqa: E402


def _state():
    import mlx.core as mx

    c = ArraysCache(1)
    c[0] = mx.zeros((1, 4))
    return [c]


def test_stale_copies_of_a_conversation_go_but_not_the_system_prompt_or_the_checkpoint():
    lru = chain_cache(max_size=100, max_bytes=1 << 40)
    lru.insert_cache("m", [1, 2], _state(), cache_type="system")
    lru.insert_cache("m", [1, 2, 3, 4], _state(), cache_type="user")  # the request's start of its last message
    lru.insert_cache("m", [1, 2, 3, 4, 5, 6], _state(), cache_type="assistant")  # its reply, finished
    assert len(lru) == 3  # the checkpoint stays: a client that reorders its last message continues from it
    assert lru.fetch_nearest_cache("m", [1, 2, 3, 4, 8])[1] == [8]
    lru.insert_cache("m", [1, 2, 3, 4, 5, 6, 7], _state(), cache_type="user")  # the next request's checkpoint
    assert len(lru) == 2  # the earlier checkpoint and reply are stale now
    lru.insert_cache("m", [1, 2, 9], _state(), cache_type="user")  # another conversation: kept
    assert len(lru) == 3
    assert lru.fetch_nearest_cache("m", [1, 2, 3, 4, 5, 6, 7, 8])[1] == [8]
    assert lru.fetch_nearest_cache("m", [1, 2, 3, 4, 8])[1] == [3, 4, 8]  # from the system entry
    assert lru.fetch_nearest_cache("m", [1, 2, 9, 1])[1] == [1]
    assert lru.nbytes == 3 * _state()[0].nbytes


def test_the_disk_cache_is_private_and_forgets_a_file_that_does_not_load(tmp_path):
    import json
    import stat

    from locals.hybrid_batch import DiskPromptCache

    d = tmp_path / "prompts"
    (d).mkdir()
    (d / "abc.tokens.json").write_text(json.dumps([1, 2, 3]))
    (d / "abc.safetensors").write_bytes(b"not a safetensors file")
    disk = DiskPromptCache(d, 1.0)
    assert stat.S_IMODE(d.stat().st_mode) == 0o700
    lru = chain_cache(max_size=10, max_bytes=1 << 30)
    assert disk.restore(lru, "m", [1, 2, 3, 4]) == 0
    assert "abc" not in disk.index and not (d / "abc.safetensors").exists()  # not tried again on every request
