#!/usr/bin/env python3

"""Application entry point for the sepsis survival project."""
import os
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


def get_venv_python(project_root: Path) -> Path:
    """Get the path to the Python executable in the virtual environment."""
    venv_python: Path
    if os.name == "nt":
        venv_python = project_root / ".venv" / "Scripts" / "python.exe"
    else:
        venv_python = project_root / ".venv" / "bin" / "python"
    return venv_python


def main() -> None:
    """Run the application."""
    venv_python: Path = get_venv_python(PROJECT_ROOT)

    try:
        subprocess.run(
            [
                str(venv_python),
                "-m",
                "streamlit",
                "run",
                str(SRC_DIR / "hello.py"),
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    except KeyboardInterrupt:
        print("\nApplication interrupted by user.")
    except FileNotFoundError:
        print(f"Virtual environment not found: {venv_python}")
    except subprocess.CalledProcessError as error:
        print(f"Streamlit exited with an error: {error}")
    finally:
        print("Application has exited.")
        if os.environ.get("VIRTUAL_ENV"):
            print("Run 'deactivate' in the terminal "
                  "to leave the virtual "
                  "environment.")


if __name__ == "__main__":
    main()
