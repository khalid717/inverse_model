import os
import pytest
from unittest.mock import MagicMock, patch

from app.storage import LocalStore, S3Store


# ---------------------------------------------------------------------------
# LocalStore
# ---------------------------------------------------------------------------

def test_local_put_and_get_text(tmp_path):
    store = LocalStore(root=str(tmp_path))
    store.put_text("subdir/data.json", '{"x": 1}')
    assert store.get_text("subdir/data.json") == '{"x": 1}'


def test_local_put_creates_parent_dirs(tmp_path):
    store = LocalStore(root=str(tmp_path))
    store.put_text("a/b/c/file.txt", "hello")
    assert (tmp_path / "a" / "b" / "c" / "file.txt").exists()


def test_local_materialize_relative_path(tmp_path):
    store = LocalStore(root=str(tmp_path))
    (tmp_path / "raster.tif").write_bytes(b"fake")
    path = store.materialize("raster.tif", tmpdir=str(tmp_path / "work"))
    assert os.path.exists(path)


def test_local_materialize_absolute_path(tmp_path):
    store = LocalStore(root=str(tmp_path))
    abs_file = tmp_path / "abs.tif"
    abs_file.write_bytes(b"data")
    result = store.materialize(str(abs_file), tmpdir=str(tmp_path / "work"))
    assert result == str(abs_file)


# ---------------------------------------------------------------------------
# S3Store — filename-collision fix
# ---------------------------------------------------------------------------

def _make_s3_store(bucket="bucket", prefix="lib"):
    with patch("boto3.client") as mock_cls:
        mock_s3 = MagicMock()
        mock_cls.return_value = mock_s3
        store = S3Store(bucket=bucket, prefix=prefix)
        store.s3 = mock_s3
    return store


def test_s3_materialize_mirrors_key_structure(tmp_path):
    store = _make_s3_store(prefix="lib")

    downloaded = {}

    def fake_download(bucket, key, local_path):
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(f"data:{key}".encode())
        downloaded[key] = local_path

    store.s3.download_file.side_effect = fake_download

    p1 = store.materialize("domain_a/vel.asc", tmpdir=str(tmp_path))
    p2 = store.materialize("domain_b/vel.asc", tmpdir=str(tmp_path))

    assert p1 != p2, "Paths must differ for different domains"
    with open(p1, "rb") as f1, open(p2, "rb") as f2:
        assert f1.read() != f2.read(), "File contents must not collide"


def test_s3_materialize_reuses_cached_file(tmp_path):
    store = _make_s3_store(prefix="lib")

    call_count = {"n": 0}

    def fake_download(bucket, key, local_path):
        call_count["n"] += 1
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        with open(local_path, "wb") as f:
            f.write(b"cached")

    store.s3.download_file.side_effect = fake_download

    store.materialize("domain_a/vel.asc", tmpdir=str(tmp_path))
    store.materialize("domain_a/vel.asc", tmpdir=str(tmp_path))
    assert call_count["n"] == 1, "Second call should use cached file"


def test_s3_get_text(tmp_path):
    store = _make_s3_store()
    store.s3.get_object.return_value = {"Body": MagicMock(read=lambda: b'{"key":"val"}')}
    text = store.get_text("index/file.jsonl")
    assert text == '{"key":"val"}'


def test_s3_put_text(tmp_path):
    store = _make_s3_store()
    store.put_text("outputs/result.json", '{"ok": true}')
    store.s3.put_object.assert_called_once()
    call_kwargs = store.s3.put_object.call_args[1]
    assert call_kwargs["Body"] == b'{"ok": true}'
