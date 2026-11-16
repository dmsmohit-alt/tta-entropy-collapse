"""Tests for the CIFAR-10-C loader and stream builder, using tiny FAKE .npy files (numpy only)."""
import numpy as np
import pytest

from src.data.cifar10c import load_corruption
from src.data.stream import StreamSpec, iter_stream, spec_from_cfg, steps_per_cycle, total_steps

N = 100  # fake images per severity


def _make_fake(root, names=("gaussian_noise", "fog", "snow")):
    """Pixel [0,0,0] of each image encodes (corruption id, severity); [0,0,1] encodes the row within severity."""
    for ci, name in enumerate(names):
        x = np.zeros((5 * N, 32, 32, 3), dtype=np.uint8)
        for s in range(5):
            x[s * N:(s + 1) * N, 0, 0, 0] = 10 * ci + s + 1
            x[s * N:(s + 1) * N, 0, 0, 1] = np.arange(N)
        np.save(root / f"{name}.npy", x)
    np.save(root / "labels.npy", np.tile(np.arange(N) % 10, 5).astype(np.int64))
    return names


def _spec(root, names, **kw):
    base = dict(data_root=str(root), corruptions=tuple(names), severity=5, num_cycles=2, batch_size=20,
                seed=0, n_per_severity=N)
    base.update(kw)
    return StreamSpec(**base)


def test_loader_slices_the_right_severity_and_labels(tmp_path):
    names = _make_fake(tmp_path)
    x, y = load_corruption(tmp_path, "fog", 3, n_per_severity=N)
    assert x.shape == (N, 32, 32, 3) and (x[:, 0, 0, 0] == 10 * 1 + 3).all()
    assert y.shape == (N,) and y.dtype == np.int64 and (y == np.arange(N) % 10).all()


def test_loader_rejects_bad_shape_severity_and_missing(tmp_path):
    names = _make_fake(tmp_path)
    np.save(tmp_path / "bad.npy", np.zeros((77, 32, 32, 3), dtype=np.uint8))
    with pytest.raises(ValueError):
        load_corruption(tmp_path, "bad", 5, n_per_severity=N)
    with pytest.raises(ValueError):
        load_corruption(tmp_path, "fog", 6, n_per_severity=N)
    with pytest.raises(FileNotFoundError):
        load_corruption(tmp_path, "nope", 5, n_per_severity=N)


def test_stream_length_order_and_coverage(tmp_path):
    names = _make_fake(tmp_path)
    spec = _spec(tmp_path, names)
    batches = list(iter_stream(spec))
    assert steps_per_cycle(spec) == 3 * 5 and len(batches) == total_steps(spec) == 30
    assert [b.step for b in batches] == list(range(30))
    # order: cycle 0: gaussian_noise x5, fog x5, snow x5; then cycle 1 again
    assert [b.corruption for b in batches[:15:5]] == list(names) and batches[15].cycle == 1 and batches[15].corruption_idx == 0
    for c in range(2):                                           # every image exactly once per (cycle, corruption)
        for ci in range(3):
            seg = [b for b in batches if b.cycle == c and b.corruption_idx == ci]
            rows = np.concatenate([b.x[:, 0, 0, 1] for b in seg])
            assert sorted(rows.tolist()) == list(range(N))
            assert all((b.x[:, 0, 0, 0] == 10 * ci + 5).all() for b in seg)            # severity 5 only
            assert all((b.y == b.x[:, 0, 0, 1].astype(int) % 10).all() for b in seg)   # labels follow the images


def test_stream_reproducible_and_seed_and_cycle_dependent(tmp_path):
    names = _make_fake(tmp_path)
    rows = lambda spec: [tuple(b.x[:, 0, 0, 1]) for b in iter_stream(spec)]  # noqa: E731
    a, a2 = rows(_spec(tmp_path, names, seed=3)), rows(_spec(tmp_path, names, seed=3))
    b = rows(_spec(tmp_path, names, seed=4))
    assert a == a2 and a != b
    assert a[:5] != a[15:20]                                      # cycle 1 reshuffles the same corruption


def test_no_shuffle_is_test_set_order(tmp_path):
    names = _make_fake(tmp_path)
    b0 = next(iter_stream(_spec(tmp_path, names, shuffle=False)))
    assert b0.x[:, 0, 0, 1].tolist() == list(range(20))


def test_max_steps_and_drop_last(tmp_path):
    names = _make_fake(tmp_path)
    assert len(list(iter_stream(_spec(tmp_path, names, max_steps=7)))) == 7
    assert total_steps(_spec(tmp_path, names, max_steps=7)) == 7
    s = _spec(tmp_path, names, batch_size=30, num_cycles=1)        # 100 // 30 = 3 full batches per segment
    bs = list(iter_stream(s))
    assert len(bs) == 9 and all(b.x.shape[0] == 30 for b in bs)
    with pytest.raises(ValueError):
        list(iter_stream(_spec(tmp_path, names, batch_size=N + 1)))


def test_spec_from_cfg_num_corruptions_and_explicit_list():
    cfg = {"data": {"root": "x"}, "stream": {"batch_size": 50, "num_corruptions": 3, "severity": 5,
                                             "num_cycles": 2, "max_steps": None}}
    s = spec_from_cfg(cfg, seed=7)
    assert s.corruptions == ("gaussian_noise", "shot_noise", "impulse_noise") and s.seed == 7 and s.max_steps is None
    cfg["stream"]["corruptions"] = ["fog", "snow"]
    assert spec_from_cfg(cfg, 0).corruptions == ("fog", "snow")
