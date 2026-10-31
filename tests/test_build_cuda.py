"""The nvcc build behind setup.py, driven through a fake nvcc so no GPU is needed."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
import time
import types
import warnings
from pathlib import Path

import pytest

from nesle import env as nesle_env

REPO = Path(__file__).resolve().parents[1]
FAKE_NVCC = """#!/bin/sh
if [ "$1" = "--version" ]; then
  echo "Cuda compiler driver, release ${FAKE_NVCC_RELEASE:-12.9}, V0"
  exit 0
fi
echo "$*" >> "$NVCC_LOG"
[ -n "$FAKE_NVCC_FAIL" ] && exit 3
while [ $# -gt 1 ]; do
  [ "$1" = "-o" ] && out="$2"
  shift
done
: > "$out"
[ -n "$FAKE_NVCC_FAIL_AFTER_WRITE" ] && exit 3
exit 0
"""

pytestmark = pytest.mark.skipif(os.name == "nt", reason="the fake nvcc and nvidia-smi are sh scripts")


def _load_build_script():
    spec = importlib.util.spec_from_file_location("build_cuda_extension", REPO / "scripts" / "build_cuda_extension.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _setup(checkout: Path, *args: str, command: str = "build_ext", **env: str) -> subprocess.CompletedProcess:
    clean = {k: v for k, v in os.environ.items() if not k.startswith("NESLE_") and k != "NVCC"}
    clean["NESLE_CUDA_ARCH"] = "sm_99"
    clean.update(env)
    return subprocess.run(
        [sys.executable, "setup.py", command, *args],
        cwd=checkout, env=clean, capture_output=True, text=True,
    )


def _built(checkout: Path, stem: str) -> list[Path]:
    return list(checkout.glob(f"build/lib*/nesle/{stem}*"))


@pytest.fixture(scope="module")
def cpu_checkout(tmp_path_factory) -> Path:
    """A copy of the repo with _core compiled once, since that is the slow part."""
    pytest.importorskip("pybind11")
    root = tmp_path_factory.mktemp("checkout")
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(REPO / name, root / name)
    for name in ("cpp", "scripts"):
        shutil.copytree(REPO / name, root / name)
    shutil.copytree(REPO / "src", root / "src", ignore=shutil.ignore_patterns("*.so", "*.pyd", "__pycache__"))
    result = _setup(root, NVCC="/nonexistent/nvcc")
    assert result.returncode == 0, result.stdout + result.stderr
    return root


@pytest.fixture
def checkout(cpu_checkout, tmp_path) -> Path:
    root = tmp_path / "checkout"
    shutil.copytree(cpu_checkout, root)
    return root


@pytest.fixture
def nvcc(tmp_path) -> dict[str, str]:
    """Env pointing setup.py at a fake nvcc, whose calls land in the NVCC_LOG file."""
    fake = tmp_path / "nvcc"
    fake.write_text(FAKE_NVCC)
    fake.chmod(0o755)
    return {"NVCC": str(fake), "NVCC_LOG": str(tmp_path / "nvcc.log")}


def _nvcc_calls(nvcc: dict[str, str]) -> int:
    log = Path(nvcc["NVCC_LOG"])
    return len(log.read_text().splitlines()) if log.exists() else 0


def test_missing_nvcc_still_builds_the_cpu_core(checkout):
    result = _setup(checkout, NVCC="/nonexistent/nvcc")
    assert result.returncode == 0, result.stdout + result.stderr
    assert _built(checkout, "_core")
    assert not _built(checkout, "_cuda_core")


def test_missing_nvcc_is_fatal_when_cuda_is_required(checkout):
    result = _setup(checkout, NVCC="/nonexistent/nvcc", NESLE_REQUIRE_CUDA="1")
    assert result.returncode != 0
    assert "nvcc" in result.stdout + result.stderr


def test_skipped_build_keeps_an_old_binary_out_of_the_wheel_tree(checkout, nvcc):
    assert _setup(checkout, **nvcc).returncode == 0
    assert _setup(checkout, NVCC="/nonexistent/nvcc").returncode == 0
    assert not _built(checkout, "_cuda_core")


def test_skipped_inplace_build_leaves_the_checkout_binary_alone(checkout, nvcc):
    assert _setup(checkout, "--inplace", **nvcc).returncode == 0
    assert _setup(checkout, "--inplace", NVCC="/nonexistent/nvcc").returncode == 0
    assert list((checkout / "src" / "nesle").glob("_cuda_core*"))


def test_strict_editable_install_works_without_nvcc(checkout, tmp_path):
    result = _setup(
        checkout, "--dist-dir", str(tmp_path / "dist"), "--mode", "strict",
        command="editable_wheel", NVCC="/nonexistent/nvcc",
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_too_old_nvcc_counts_as_missing(checkout, nvcc):
    result = _setup(checkout, FAKE_NVCC_RELEASE="11.8", **nvcc)
    assert result.returncode == 0, result.stdout + result.stderr
    assert not _built(checkout, "_cuda_core")
    assert _nvcc_calls(nvcc) == 0


def test_too_old_nvcc_is_fatal_when_cuda_is_required(checkout, nvcc):
    result = _setup(checkout, FAKE_NVCC_RELEASE="11.8", NESLE_REQUIRE_CUDA="1", **nvcc)
    assert result.returncode != 0
    assert "nvcc" in result.stdout + result.stderr


def test_bare_nvcc_name_is_found_on_path(checkout, nvcc):
    path = f"{Path(nvcc['NVCC']).parent}{os.pathsep}{os.environ['PATH']}"
    result = _setup(checkout, NVCC="nvcc", NVCC_LOG=nvcc["NVCC_LOG"], PATH=path)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _nvcc_calls(nvcc) == 1


def test_nvcc_gets_an_absolute_output_path(checkout, nvcc):
    # vcvarsall can change directory before nvcc runs, when VSCMD_START_DIR is set.
    assert _setup(checkout, **nvcc).returncode == 0
    words = Path(nvcc["NVCC_LOG"]).read_text().split()
    assert os.path.isabs(words[words.index("-o") + 1])


def test_shell_entry_point_runs_the_python_build(tmp_path):
    env = dict(os.environ, PYTHON=sys.executable, NVCC="/nonexistent/nvcc", NESLE_REQUIRE_CUDA="1")
    result = subprocess.run(
        ["sh", str(REPO / "scripts" / "build_cuda_extension.sh")],
        cwd=tmp_path, env=env, capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "nvcc is not available" in result.stdout


def test_nvcc_builds_the_extension_for_the_requested_arch(checkout, nvcc):
    result = _setup(checkout, **nvcc)
    assert result.returncode == 0, result.stdout + result.stderr
    assert _built(checkout, "_cuda_core")
    command = Path(nvcc["NVCC_LOG"]).read_text()
    assert "-arch=sm_99" in command
    assert "kernels.cu" in command and "cuda_module.cu" in command


def test_inplace_build_lands_next_to_the_python_package(checkout, nvcc):
    result = _setup(checkout, "--inplace", **nvcc)
    assert result.returncode == 0, result.stdout + result.stderr
    assert list((checkout / "src" / "nesle").glob("_cuda_core*"))


def test_a_failing_nvcc_fails_the_build_instead_of_keeping_a_stale_binary(checkout, nvcc):
    result = _setup(checkout, FAKE_NVCC_FAIL="1", **nvcc)
    assert result.returncode != 0
    assert "nvcc" in result.stdout + result.stderr


def test_unchanged_sources_do_not_recompile(checkout, nvcc):
    assert _setup(checkout, **nvcc).returncode == 0
    assert _setup(checkout, **nvcc).returncode == 0
    assert _nvcc_calls(nvcc) == 1


@pytest.mark.parametrize("touched", [
    "cpp/include/nesle/cuda/batch_bus.cuh",
    "cpp/src/cuda/kernels.cu",
    "cpp/bindings/cuda_module.cu",
    "cpp/src/rom.cpp",
    "scripts/build_cuda_extension.py",
])
def test_changed_source_recompiles(checkout, nvcc, touched):
    assert _setup(checkout, **nvcc).returncode == 0
    future = time.time() + 60
    os.utime(checkout / touched, (future, future))
    assert _setup(checkout, **nvcc).returncode == 0
    assert _nvcc_calls(nvcc) == 2


def test_changed_arch_recompiles(checkout, nvcc):
    assert _setup(checkout, **nvcc).returncode == 0
    assert _setup(checkout, **{**nvcc, "NESLE_CUDA_ARCH": "sm_61"}).returncode == 0
    assert _nvcc_calls(nvcc) == 2
    assert "-arch=sm_61" in Path(nvcc["NVCC_LOG"]).read_text().splitlines()[-1]


def test_changed_toolkit_recompiles(checkout, nvcc, tmp_path):
    assert _setup(checkout, **nvcc).returncode == 0
    other = tmp_path / "other-toolkit"
    other.mkdir()
    shutil.copy2(nvcc["NVCC"], other / "nvcc")
    assert _setup(checkout, **{**nvcc, "NVCC": str(other / "nvcc")}).returncode == 0
    assert _nvcc_calls(nvcc) == 2


def test_force_recompiles(checkout, nvcc):
    assert _setup(checkout, **nvcc).returncode == 0
    assert _setup(checkout, "--force", **nvcc).returncode == 0
    assert _nvcc_calls(nvcc) == 2


def test_failed_compile_is_not_mistaken_for_up_to_date(checkout, nvcc):
    assert _setup(checkout, **nvcc).returncode == 0
    future = time.time() + 60
    os.utime(checkout / "cpp/src/cuda/kernels.cu", (future, future))
    # A partial binary newer than its sources, left behind by the failed compile.
    assert _setup(checkout, FAKE_NVCC_FAIL_AFTER_WRITE="1", **nvcc).returncode != 0
    for partial in _built(checkout, "_cuda_core"):
        os.utime(partial, (future + 60, future + 60))
    assert _setup(checkout, **nvcc).returncode == 0
    assert _nvcc_calls(nvcc) == 3


def _fake_smi(tmp_path: Path, monkeypatch, body: str) -> None:
    smi = tmp_path / "nvidia-smi"
    smi.write_text(f"#!/bin/sh\n{body}\n")
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.delenv("NESLE_CUDA_ARCH", raising=False)
    # A pip build runs in an isolated env where torch cannot be imported.
    monkeypatch.setitem(sys.modules, "torch", None)


def test_detect_arch_asks_nvidia_smi_without_torch(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, 'echo "6.1"; echo "8.0"')
    assert _load_build_script().detect_arch() == "sm_61"


def test_detect_arch_falls_back_when_nvidia_smi_fails(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, "exit 9")
    assert _load_build_script().detect_arch() == "sm_80"


def test_detect_arch_env_override_wins(tmp_path, monkeypatch):
    _fake_smi(tmp_path, monkeypatch, 'echo "6.1"')
    monkeypatch.setenv("NESLE_CUDA_ARCH", "sm_90")
    assert _load_build_script().detect_arch() == "sm_90"


def _tree(root: Path) -> Path:
    """A source checkout whose built extension is older than every source."""
    built = root / "src" / "nesle" / "_cuda_core.so"
    built.parent.mkdir(parents=True)
    built.touch()
    os.utime(built, (1000, 1000))
    for source in ("cpp/include/nesle/cuda/state.cuh", "cpp/src/cuda/kernels.cu",
                   "cpp/bindings/cuda_module.cu", "cpp/src/rom.cpp", "cpp/src/smb.cpp"):
        path = root / source
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
        os.utime(path, (500, 500))
    return built


@pytest.mark.parametrize("source", ["cpp/include/nesle/cuda/state.cuh", "cpp/src/cuda/kernels.cu"])
def test_warns_when_a_cuda_source_is_newer_than_the_binary(tmp_path, source):
    built = _tree(tmp_path)
    os.utime(tmp_path / source, (2000, 2000))
    with pytest.warns(UserWarning, match="stale"):
        nesle_env._warn_if_cuda_core_stale(str(built))


def test_no_warning_for_a_fresh_binary(tmp_path):
    built = _tree(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nesle_env._warn_if_cuda_core_stale(str(built))


def test_no_warning_for_a_source_only_the_cpu_core_uses(tmp_path):
    built = _tree(tmp_path)
    os.utime(tmp_path / "cpp/src/smb.cpp", (2000, 2000))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nesle_env._warn_if_cuda_core_stale(str(built))


def test_no_warning_for_a_wheel_install_without_sources(tmp_path):
    built = tmp_path / "lib" / "site-packages" / "nesle" / "_cuda_core.so"
    built.parent.mkdir(parents=True)
    built.touch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nesle_env._warn_if_cuda_core_stale(str(built))



def test_no_warning_for_a_vendored_install_inside_a_source_checkout(tmp_path):
    _tree(tmp_path)
    os.utime(tmp_path / "cpp/src/cuda/kernels.cu", (2000, 2000))
    built = tmp_path / "vendor" / "nesle" / "_cuda_core.so"
    built.parent.mkdir(parents=True)
    built.touch()
    os.utime(built, (1000, 1000))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        nesle_env._warn_if_cuda_core_stale(str(built))


@pytest.mark.parametrize("path", ["/_cuda_core.so", "/nesle/_cuda_core.so"])
def test_stale_check_tolerates_a_shallow_path(path):
    nesle_env._warn_if_cuda_core_stale(path)


def test_stale_check_tolerates_an_unreadable_binary(tmp_path):
    built = _tree(tmp_path)
    built.unlink()
    nesle_env._warn_if_cuda_core_stale(str(built))


def test_creating_a_cuda_batch_runs_the_stale_check(tmp_path, monkeypatch):
    built = _tree(tmp_path)
    os.utime(tmp_path / "cpp/src/cuda/kernels.cu", (2000, 2000))
    fake = types.SimpleNamespace(__file__=str(built), CudaBatch=lambda *args: "batch")
    monkeypatch.setattr("nesle._cuda_core", fake, raising=False)
    with pytest.warns(UserWarning, match="stale"):
        assert nesle_env._make_cuda_batch(1, 1) == "batch"


def test_suite_skips_when_the_extension_is_built_but_no_gpu_is_visible():
    pytest.importorskip("nesle._cuda_core")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/test_critical_fixes.py", "-q", "-k", "auto_resets"],
        cwd=REPO, env=dict(os.environ, CUDA_VISIBLE_DEVICES=""), capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 skipped" in result.stdout
