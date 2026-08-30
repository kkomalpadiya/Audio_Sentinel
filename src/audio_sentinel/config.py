from pathlib import Path

from pydantic import BaseModel


class Paths(BaseModel):
    root: Path
    raw_data: Path
    interim_data: Path
    processed_data: Path
    models: Path

    @classmethod
    def from_root(cls, root: Path) -> "Paths":
        return cls(
            root=root,
            raw_data=root / "data" / "raw",
            interim_data=root / "data" / "interim",
            processed_data=root / "data" / "processed",
            models=root / "models",
        )

