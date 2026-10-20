"""Application entry point for the sepsis survival project."""
import os
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
MAX_SUPPORTED_PYTHON = (3, 13)


def get_venv_python(project_root: Path) -> Path:
    """Get the path to the Python executable in the virtual environment."""
    if os.name == "nt":
        return project_root / ".venv" / "Scripts" / "python.exe"
    return project_root / ".venv" / "bin" / "python"


def get_bootstrap_python() -> str:
    """Select a Python version compatible with the pinned scientific packages."""
    current_version = sys.version_info[:2]

    if current_version <= MAX_SUPPORTED_PYTHON:
        return sys.executable

    # If the current Python version is greater than the maximum supported version,
    if current_version > MAX_SUPPORTED_PYTHON:
        raise RuntimeError(
            f"Python {current_version[0]}.{current_version[1]} is not compatible with the pinned "
            f"scikit-learn 1.7.1 dependencies. Install Python 3.13 and run this command again."
        )

    for executable in ("python3.13", "python3.12", "python3.11"):
        compatible_python = shutil.which(executable)
        if compatible_python:
            return compatible_python

    raise RuntimeError(
        "Python 3.14 or newer is not compatible with the pinned "
        "scikit-learn 1.7.1 dependencies. Install Python 3.13 and run "
        "this command again."
    )


def setup_virtual_environment(venv_python: Path) -> None:
    """Create and install dependencies in the project virtual environment."""
    venv_dir = venv_python.parent.parent
    subprocess.run(
        [get_bootstrap_python(), "-m", "venv", str(venv_dir)],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=PROJECT_ROOT,
        check=True,
    )
    subprocess.run(
        [str(venv_python), "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=PROJECT_ROOT,
        check=True,
    )


def main() -> None:
    """Run the application."""
    venv_python: Path = get_venv_python(PROJECT_ROOT)

    try:
        dependencies_are_installed = venv_python.is_file() and (
            subprocess.run(
                [
                    str(venv_python),
                    "-c",
                    "import streamlit",
                ],
                cwd=PROJECT_ROOT,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            ).returncode
            == 0
        )

        if not venv_python.is_file() or not dependencies_are_installed:
            print("Setting up the project virtual environment...")
            setup_virtual_environment(venv_python)

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
    except FileNotFoundError as error:
        print(f"Required executable or file was not found: {error}")
    except RuntimeError as error:
        print(f"Environment setup failed: {error}")
    except subprocess.CalledProcessError as error:
        print(f"Command failed with exit code {error.returncode}: {error.cmd}")
    finally:
        print("Application has exited.")


if __name__ == "__main__":
    main()
