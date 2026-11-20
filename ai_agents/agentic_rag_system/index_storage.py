"""Filesystem helpers for safely replacing the active vector-store directory."""

import os
import shutil
import tempfile
from collections.abc import Callable


def replace_directory(target_path: str, populate: Callable[[str], object]) -> str | None:
    """Build a replacement off to the side, then swap it in without losing the old data on failure.

    Returns the previous directory path only when the new directory was installed but
    cleanup of that old directory failed.
    """
    target_path = os.path.abspath(target_path)
    parent_path = os.path.dirname(target_path)
    directory_name = os.path.basename(target_path)
    os.makedirs(parent_path, exist_ok=True)

    staging_path = tempfile.mkdtemp(
        prefix=f".{directory_name}.staging-",
        dir=parent_path,
    )
    backup_path = None

    try:
        populate(staging_path)

        if os.path.exists(target_path):
            backup_path = tempfile.mkdtemp(
                prefix=f".{directory_name}.backup-",
                dir=parent_path,
            )
            os.rmdir(backup_path)
            os.replace(target_path, backup_path)

        try:
            os.replace(staging_path, target_path)
        except Exception:
            if backup_path and os.path.exists(backup_path):
                os.replace(backup_path, target_path)
                backup_path = None
            raise

        if backup_path:
            try:
                shutil.rmtree(backup_path)
            except OSError:
                return backup_path
            backup_path = None

        return None
    except Exception:
        if (
            backup_path
            and os.path.exists(backup_path)
            and not os.path.exists(target_path)
        ):
            os.replace(backup_path, target_path)
            backup_path = None
        raise
    finally:
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path, ignore_errors=True)
