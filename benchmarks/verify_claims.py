#!/usr/bin/env python3
"""Re-measure every performance claim in the README and report pass/fail.

Reuses the measurement functions in ``benchmarks/gpu_vs_cpu.py`` rather than
reimplementing the protocol, so results are directly comparable to the
published tables: frameskip 4, RIGHT held, 30 warmup + 200 timed steps,
``render_frame=False, copy_obs=False``.

Runs anywhere with a CUDA GPU. Designed to be driven either locally, from the
Colab CLI, or from ``notebooks/verify_claims.ipynb``:

    colab new -s nesle --gpu A100
    colab upload -s nesle "Super Mario Bros. (World).nes" /content/rom.nes
    colab exec   -s nesle -f benchmarks/verify_claims.py
    colab download -s nesle /content/verification.json ./docs/data/
    colab stop -s nesle

Configuration comes from flags, or the matching NESLE_* environment variables
when run through ``colab exec`` (which does not forward argv).
"""
from __future__ import annotations

import argparse
import gc
import gzip
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO_URL = "https://github.com/hbofz/NeSLE.git"
ROM_NAME = "Super Mario Bros. (World).nes"

# Every performance number the README asserts, keyed by device class.
CLAIMS = {
    "A100": {4096: 311_844, 16384: 1_050_710, 32768: 1_931_460,
             65536: 3_142_205, 131072: 2_816_144},
    "1050Ti": {4096: 180_437},
}
TOLERANCE = 0.20  # clocks, driver version and host load all move these


def sh(*cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def smi(query: str) -> list[str]:
    try:
        out = sh("nvidia-smi", f"--query-gpu={query}",
                 "--format=csv,noheader,nounits").stdout.strip()
    except FileNotFoundError:
        sys.exit("nvidia-smi not found. This script needs a CUDA GPU; run it on "
                 "a GPU machine, or via:  colab new --gpu A100")
    if not out:
        sys.exit("nvidia-smi returned nothing. Is a GPU visible to this process?")
    return out.split("\n")[0].split(", ")


def mem_used_mib() -> int:
    return int(smi("memory.used")[0])


def device_class(name: str) -> str:
    if "A100" in name:
        return "A100"
    if "1050" in name:
        return "1050Ti"
    return "other"


def bootstrap(repo: Path, build: bool) -> None:
    """Clone and build in place if we are not already inside a built checkout."""
    if not repo.exists():
        print(f"cloning {REPO_URL} -> {repo}")
        subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(repo)],
                       check=True)
    os.chdir(repo)
    if not build:
        return
    print("installing package ...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", ".[dev,rl]"],
                   check=True)
    print("building CUDA extension ...")
    subprocess.run([sys.executable, "scripts/build_cuda_extension.py"], check=True)


def run_tests() -> dict:
    r = sh(sys.executable, "-m", "pytest", "tests/", "-q")
    lines = [l for l in r.stdout.splitlines() if " passed" in l or " failed" in l]
    return {"returncode": r.returncode,
            "summary": lines[-1].strip() if lines else "no summary",
            "passed": r.returncode == 0}


def run_correctness() -> dict:
    r = sh(sys.executable, "benchmarks/verify_correctness.py")
    ok = "All three falsifiability checks passed" in r.stdout
    if not ok:
        print(r.stdout[-2000:], r.stderr[-1000:])
    return {"returncode": r.returncode, "passed": ok}


def sweep(counts: list[int], baseline: float) -> list[dict]:
    from benchmarks.gpu_vs_cpu import bench_gpu_batched
    rows = []
    for n in counts:
        try:
            r = bench_gpu_batched(n)
            r["vs_cpu"] = r["env_steps_per_s"] / baseline
            rows.append(r)
            print(f"  {n:>7,} envs  {r['env_steps_per_s']:>13,.0f} env-steps/s"
                  f"  {r['vs_cpu']:>9,.1f}x CPU")
        except Exception as exc:
            print(f"  {n:>7,} envs  FAILED: {type(exc).__name__}: {exc}")
            break
    return rows


def measure_memory(repo: Path, envs: int) -> dict:
    """Footprint at `envs`, sampled while the batch is still referenced.

    bench_gpu_batched frees its CudaBatch on return, so sampling after it would
    measure released memory. This holds the batch alive across the reading.
    """
    import numpy as np
    from nesle._cuda_core import CudaBatch

    rom = (repo / ROM_NAME).read_bytes()
    state = gzip.decompress((repo / "docs/data/smb_level1_1.state").read_bytes())

    gc.collect(); time.sleep(2)
    before = mem_used_mib()

    batch = CudaBatch(envs, 4, rom, state)
    batch.reset()
    acts = np.full(envs, 0x80, dtype=np.uint8)
    for _ in range(10):
        batch.step(acts, render_frame=False, copy_obs=False)
    live = mem_used_mib()          # batch is still alive here

    del batch, acts
    gc.collect(); time.sleep(2)

    return {"envs": envs, "before_mib": before, "live_mib": live,
            "after_free_mib": mem_used_mib(), "attributable_mib": live - before,
            "total_mib": int(smi("memory.total")[0])}


def nespy_baseline(venv: Path) -> dict:
    """nes-py needs an old gym/numpy combination, so it runs isolated."""
    pkgs = ["numpy<2", "gym>=0.25,<0.26", "nes-py>=8.2,<8.3",
            "gym-super-mario-bros>=7.4,<7.5"]
    try:
        if not venv.exists():
            # nes-py needs an old gym/numpy pair on Python <= 3.11. Colab's
            # python3-venv ships without ensurepip, so stdlib venv exits 1
            # there; uv brings its own interpreter and sidesteps both problems.
            if sh(sys.executable, "-m", "venv", str(venv)).returncode != 0:
                if not shutil.which("uv"):
                    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"],
                                   check=True)
                subprocess.run(["uv", "venv", "--python", "3.11", str(venv)], check=True)
                subprocess.run(["uv", "pip", "install", "--python",
                                str(venv / "bin" / "python"), *pkgs], check=True)
            else:
                subprocess.run([str(venv / "bin" / "pip"), "install", "-q", *pkgs],
                               check=True)
        r = sh(str(venv / "bin" / "python"), "benchmarks/nespy_baseline.py", timeout=900)
        m = re.search(r"([\d,.]+)\s*env-steps/s", r.stdout)
        if m:
            return {"status": "ok", "env_steps_per_s": float(m.group(1).replace(",", ""))}
        return {"status": "ran", "raw": r.stdout[-400:], "err": r.stderr[-400:]}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


MULTICORE_SRC = '''
import sys, time
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from nes_py.wrappers import JoypadSpace
from gym.vector import AsyncVectorEnv

N = int(sys.argv[1]); FRAMES = 600; RIGHT_B = 3
mk = lambda: JoypadSpace(gym_super_mario_bros.make("SuperMarioBros-v0"), RIGHT_ONLY)
envs = AsyncVectorEnv([mk for _ in range(N)])
envs.reset()
acts = [RIGHT_B] * N
for _ in range(60): envs.step(acts)
t0 = time.perf_counter()
for _ in range(FRAMES): envs.step(acts)
el = time.perf_counter() - t0
envs.close()
print(f"workers={N} env_steps_per_s={N*FRAMES/el/4:.1f}")
'''


def nespy_multicore(venv: Path, workers: int) -> dict:
    """The comparison a skeptical reader asks for: nes-py on every core."""
    if not venv.exists():
        return {"status": "skipped", "reason": "nes-py venv unavailable"}
    try:
        script = Path("/tmp/nespy_multi.py")
        script.write_text(MULTICORE_SRC)
        r = sh(str(venv / "bin" / "python"), str(script), str(workers), timeout=1800)
        m = re.search(r"env_steps_per_s=([\d.]+)", r.stdout)
        if m:
            return {"status": "ok", "workers": workers,
                    "env_steps_per_s": float(m.group(1))}
        return {"status": "ran", "raw": r.stdout[-400:], "err": r.stderr[-400:]}
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}


def report(dev: str, gpu: str, rows: list[dict], tests: dict, correct: dict,
           memory: dict, single: dict, multi: dict, crossover, peak) -> list[dict]:
    measured = {r["num_envs"]: r["env_steps_per_s"] for r in rows}
    verdicts = []
    print(f"\n{'='*68}\nCLAIMED vs MEASURED   ({gpu}, class={dev})\n{'='*68}")
    claims = CLAIMS.get(dev, {})
    if claims:
        print(f"{'envs':>9} {'claimed':>14} {'measured':>14} {'ratio':>8}  verdict")
        print("-" * 68)
        for envs, claimed in sorted(claims.items()):
            got = measured.get(envs)
            if got is None:
                verdict, ratio = "NOT RUN", "-"
            else:
                ratio_f = got / claimed
                verdict = "PASS" if abs(ratio_f - 1) <= TOLERANCE else f"FAIL ({ratio_f:.2f}x)"
                ratio = f"{ratio_f:.2f}x"
            verdicts.append({"envs": envs, "claimed": claimed, "measured": got,
                             "verdict": verdict})
            print(f"{envs:>9,} {claimed:>14,} "
                  f"{(f'{got:,.0f}' if got else '-'):>14} {ratio:>8}  {verdict}")
    else:
        print(f"No published claim set for this GPU. Sweep recorded, not checked.")

    print("\n--- other claims ---")
    print(f"test suite            : {tests['summary']}")
    print(f"falsifiability checks : {'PASS' if correct['passed'] else 'FAIL'}")
    print(f"crossover vs 1 CPU env: {crossover} envs")
    if memory:
        print(f"memory @ {memory['envs']:,} envs   : {memory['attributable_mib']:,} MiB "
              f"attributable, {memory['live_mib']:,} of {memory['total_mib']:,} MiB in use")
    if peak:
        print(f"peak                  : {peak['env_steps_per_s']:,.0f} env-steps/s "
              f"at {peak['num_envs']:,} envs "
              f"({peak['env_steps_per_s']*4/60:,.0f}x real time)")
    if single.get("env_steps_per_s") and peak:
        v = single["env_steps_per_s"]
        print(f"nes-py, 1 process     : {v:,.1f} env-steps/s -> peak is "
              f"{peak['env_steps_per_s']/v:,.0f}x")
    if multi.get("env_steps_per_s") and peak:
        v = multi["env_steps_per_s"]
        print(f"nes-py, {multi['workers']:>2} workers    : {v:,.1f} env-steps/s -> peak is "
              f"{peak['env_steps_per_s']/v:,.0f}x   <-- the fair comparison")
    return verdicts


def main() -> int:
    env = os.environ.get
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", default=env("NESLE_REPO", "/content/NeSLE"))
    p.add_argument("--rom", default=env("NESLE_ROM", "/content/rom.nes"))
    p.add_argument("--out", default=env("NESLE_OUT", "/content/verification.json"))
    p.add_argument("--no-build", action="store_true",
                   default=bool(env("NESLE_NO_BUILD")))
    p.add_argument("--skip-nespy", action="store_true",
                   default=bool(env("NESLE_SKIP_NESPY")))
    p.add_argument("--env-counts", default=env("NESLE_ENV_COUNTS", ""))
    args = p.parse_args()

    gpu, total_mib = smi("name,memory.total")
    total_mib = int(total_mib)
    dev = device_class(gpu)
    print(f"GPU: {gpu}  ({total_mib:,} MiB)  class={dev}")

    repo = Path(args.repo).resolve()
    bootstrap(repo, build=not args.no_build)

    rom_dest = repo / ROM_NAME
    if not rom_dest.exists():
        src = Path(args.rom)
        if not src.exists():
            print(f"ERROR: no ROM at {src} and none at {rom_dest}.\n"
                  f"Upload one:  colab upload '{ROM_NAME}' /content/rom.nes")
            return 2
        rom_dest.write_bytes(src.read_bytes())
    data = rom_dest.read_bytes()
    mapper = (data[6] >> 4) | (data[7] & 0xF0)
    assert data[:4] == b"NES\x1a", "not an iNES ROM"
    assert mapper == 0, f"NeSLE supports mapper 0 only, got {mapper}"
    print(f"ROM: {len(data):,} bytes, mapper {mapper}")

    # An editable install writes a .pth that site.py reads only at interpreter
    # startup, so a self-installing run cannot import nesle by that route. Point
    # at the src layout directly; the built _cuda_core lands in src/nesle/ too.
    sys.path.insert(0, str(repo))
    sys.path.insert(0, str(repo / "src"))
    import importlib
    importlib.invalidate_caches()

    print("\n[1/6] test suite ...")
    tests = run_tests(); print("  ", tests["summary"])

    print("\n[2/6] falsifiability checks ...")
    correct = run_correctness(); print("  ", "PASS" if correct["passed"] else "FAIL")

    print("\n[3/6] CPU baseline ...")
    from benchmarks.gpu_vs_cpu import bench_cpu_single
    cpu = bench_cpu_single()
    print(f"   native CPU, 1 env: {cpu['env_steps_per_s']:,.0f} env-steps/s")

    print("\n[4/6] GPU sweep ...")
    if args.env_counts:
        counts = [int(x) for x in args.env_counts.split(",")]
    else:
        counts = [1, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
        if total_mib > 30000:
            counts += [8192, 16384, 32768, 65536, 131072]
    rows = sweep(counts, cpu["env_steps_per_s"])
    crossover = next((r["num_envs"] for r in rows if r["vs_cpu"] >= 1.0), None)
    peak = max(rows, key=lambda r: r["env_steps_per_s"]) if rows else None

    print("\n[5/6] memory at peak batch ...")
    memory = measure_memory(repo, peak["num_envs"]) if peak else {}
    if memory:
        print(f"   {memory['attributable_mib']:,} MiB attributable "
              f"({memory['attributable_mib']/1024:.1f} GiB)")

    print("\n[6/6] nes-py baselines ...")
    venv = Path("/content/nespy-venv") if Path("/content").exists() else Path("/tmp/nespy-venv")
    single = {"status": "skipped"} if args.skip_nespy else nespy_baseline(venv)
    multi = {"status": "skipped"} if args.skip_nespy else nespy_multicore(venv, os.cpu_count() or 2)
    print("   single:", single.get("status"), "| multicore:", multi.get("status"))

    verdicts = report(dev, gpu, rows, tests, correct, memory, single, multi,
                      crossover, peak)

    out = Path(args.out)
    out.write_text(json.dumps({
        "environment": {"gpu": gpu, "memory_total_mib": total_mib,
                        "device_class": dev, "python": platform.python_version(),
                        "platform": platform.platform(),
                        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
        "commit": sh("git", "rev-parse", "HEAD").stdout.strip(),
        "protocol": {"frameskip": 4, "action": "RIGHT", "warmup_steps": 30,
                     "timed_steps": 200, "render_frame": False, "copy_obs": False},
        "cpu_single_env": cpu, "sweep": rows, "crossover_envs": crossover,
        "peak": peak, "memory": memory, "tests": tests, "correctness": correct,
        "nespy_single": single, "nespy_multicore": multi, "verdicts": verdicts,
    }, indent=2))
    print(f"\nwrote {out} ({out.stat().st_size:,} bytes)")

    failed = [v for v in verdicts if v["verdict"].startswith("FAIL")]
    return 1 if (failed or not tests["passed"] or not correct["passed"]) else 0


if __name__ == "__main__":
    sys.exit(main())
