from __future__ import annotations

import os
import pathlib
from abc import ABC, abstractmethod
from typing import Optional

import boto3


class Store(ABC):
    @abstractmethod
    def get_text(self, ref: str) -> str:
        ...

    @abstractmethod
    def put_text(self, ref: str, text: str) -> None:
        ...

    @abstractmethod
    def materialize(self, ref: str, tmpdir: str) -> str:
        """
        Ensure the referenced object exists as a local file path, returning that path.
        """
        ...


class S3Store(Store):
    def __init__(self, bucket: str, prefix: str = ""):
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.s3 = boto3.client("s3")

    def _key(self, ref: str) -> str:
        ref = ref.lstrip("/")
        if self.prefix:
            if ref.startswith(self.prefix + "/"):
                return ref
            return f"{self.prefix}/{ref}"
        return ref

    def get_text(self, ref: str) -> str:
        key = self._key(ref)
        obj = self.s3.get_object(Bucket=self.bucket, Key=key)
        return obj["Body"].read().decode("utf-8")

    def put_text(self, ref: str, text: str) -> None:
        key = self._key(ref)
        self.s3.put_object(Bucket=self.bucket, Key=key, Body=text.encode("utf-8"))

    def materialize(self, ref: str, tmpdir: str) -> str:
        key = self._key(ref)
        tmp = pathlib.Path(tmpdir)
        tmp.mkdir(parents=True, exist_ok=True)
        local_path = tmp / os.path.basename(key)
        if local_path.exists() and local_path.stat().st_size > 0:
            return str(local_path)
        self.s3.download_file(self.bucket, key, str(local_path))
        return str(local_path)


class LocalStore(Store):
    def __init__(self, root: str):
        self.root = pathlib.Path(root)

    def get_text(self, ref: str) -> str:
        p = self.root / ref
        return p.read_text(encoding="utf-8")

    def put_text(self, ref: str, text: str) -> None:
        p = self.root / ref
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, encoding="utf-8")

    def materialize(self, ref: str, tmpdir: str) -> str:
        # ref is a local relative path under root OR an absolute path already
        p = pathlib.Path(ref)
        if p.is_absolute() and p.exists():
            return str(p)
        p2 = self.root / ref
        return str(p2)
