from pathlib import Path
import os


def find_repository_root(start_point_absolute_path: str) -> str:
    repository_root = Path(start_point_absolute_path).resolve().parent
    while not (repository_root / ".git").exists():
        if repository_root == repository_root.parent:
            raise RuntimeError(
                "Root of file system is reached, probably there is no .git file on search tree"
            )

        repository_root = repository_root.parent
    return repository_root.__str__()


REPOSITORY_ROOT = find_repository_root(start_point_absolute_path=__file__)
