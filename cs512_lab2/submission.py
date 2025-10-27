import zipfile
from pathlib import Path

import fire


def create_archive(archive_path: str = "submission.zip"):
    script_dir = Path(__file__).resolve().parent

    file_paths = [
        script_dir / "prog_model_sol.py",
    ]

    for file_path in file_paths:
        if not file_path.exists():
            raise FileNotFoundError(f"File {file_path} not found")

    with zipfile.ZipFile(archive_path, "w") as z:
        for file_path in file_paths:
            z.write(file_path, file_path.name)

    print(f"Archive created: {archive_path}")


if __name__ == "__main__":
    fire.Fire(create_archive)
