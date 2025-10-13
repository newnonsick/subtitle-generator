import os
import sys
import sysconfig
from pathlib import Path


def find_site_packages():
    try:
        purelib = sysconfig.get_paths().get("purelib")
    except Exception:
        purelib = None

    if purelib and Path(purelib).exists():
        return Path(purelib)

    for p in sys.path:
        if p and "site-packages" in p and Path(p).exists():
            return Path(p)

    venv_base = Path(sys.executable).parent.parent
    candidates = [
        venv_base / "Lib" / "site-packages",
        venv_base
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages",
    ]
    for c in candidates:
        if c.exists():
            return c

    return None


def build_paths(site_packages_path: Path):
    nvidia_base = site_packages_path / "nvidia"
    candidates = [
        nvidia_base / "cuda_runtime" / "bin",
        nvidia_base / "cublas" / "bin",
        nvidia_base / "cudnn" / "bin",
        nvidia_base / "nvrtc" / "bin",
    ]
    return [str(p) for p in candidates]


def shell_escape(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"


def print_posix_exports(paths_to_add):
    joined = ":".join(paths_to_add)
    for var in ("CUDA_PATH", "CUDA_PATH_V12_4"):
        print(f"export {var}={shell_escape(joined)}:'\"'\"'${{{var}:-}}'\"'\"'")

    print(f"export PATH={shell_escape(joined)}:'\"'\"'${{PATH:-}}'\"'\"'")


def main():
    site_packages = find_site_packages()
    if not site_packages:
        print(
            "# Could not determine site-packages for this Python executable",
            file=sys.stderr,
        )
        return

    paths = build_paths(site_packages)
    print_posix_exports(paths)


if __name__ == "__main__":
    main()
