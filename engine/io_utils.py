import os
import time
from pathlib import Path
from typing import Optional

import pandas as pd


class CSVWriteError(Exception):
    pass


def _acquire_lock(lock_path: Path, retries: int = 200, delay: float = 0.01) -> None:
    for _ in range(retries):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return
        except FileExistsError:
            time.sleep(delay)
    raise CSVWriteError(f"Could not acquire lock at {lock_path}")


def _release_lock(lock_path: Path) -> None:
    try:
        os.remove(lock_path)
    except FileNotFoundError:
        pass


def atomic_write_csv(df: pd.DataFrame, path: Path, append: bool = False, expected_columns: Optional[list] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = path.with_suffix(path.suffix + ".lock")
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    _acquire_lock(lock_path)
    try:
        if append and path.exists():
            existing = pd.read_csv(path, on_bad_lines="skip", engine="python")
            df = pd.concat([existing, df], ignore_index=True)
        df.to_csv(tmp_path, index=False)
        os.replace(tmp_path, path)
        # post-write sanity check
        reloaded = pd.read_csv(path, on_bad_lines="skip", engine="python")
        if reloaded.empty or (expected_columns and any(col not in reloaded.columns for col in expected_columns)):
            raise CSVWriteError(f"Sanity check failed for {path}; columns present: {list(reloaded.columns)}")
    finally:
        _release_lock(lock_path)
