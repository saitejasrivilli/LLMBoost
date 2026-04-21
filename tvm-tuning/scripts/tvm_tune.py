"""
tvm_tuning/scripts/tvm_tune.py
================================
Phase 6: Run MetaSchedule tuning on the TIR module produced by
build_relax_module.py, then export the best schedule as a .so.

Usage:
  python tvm_tune.py \
      --module relax_module.json \
      --trials 1000             \
      --num_gpus 4              \
      --target "cuda -arch=sm_86" \
      --log_dir tuning_logs/    \
      --output fused_rmsnorm_linear.so

  # Quick smoke-test (20 trials, no real GPU):
  python tvm_tune.py --trials 20 --dry_run

The script:
1.  Loads the lowered TIR module.
2.  Spawns one MetaSchedule RPCRunner per GPU (4× A30).
3.  Runs Bayesian optimisation for --trials trials.
4.  Compiles the winner to a .so shared library.
5.  Prints the winning tile configuration.
"""

import argparse
import json
import os
import time
from pathlib import Path

import tvm
from tvm import meta_schedule as ms
from tvm.target import Target


# ── Config keys written by MetaSchedule that we want to surface ───────────
TILE_CONFIG_KEYS = [
    "tile_x", "tile_y", "tile_k",
    "unroll_max_step",
    "vector_bytes",
    "thread_x", "block_x",
]


def load_module(path: str) -> tvm.IRModule:
    with open(path) as f:
        return tvm.load_json(f.read())


def get_target(target_str: str, num_gpus: int) -> list[Target]:
    """Return one Target per GPU.  All GPUs are A30 (sm_86)."""
    return [Target(target_str) for _ in range(num_gpus)]


def run_tuning(
    mod: tvm.IRModule,
    targets: list[Target],
    trials: int,
    log_dir: str,
    dry_run: bool,
) -> ms.database.JSONDatabase:
    """
    Run MetaSchedule tuning.  Returns the populated database.
    """
    os.makedirs(log_dir, exist_ok=True)
    database = ms.database.JSONDatabase(
        path_workload=os.path.join(log_dir, "workload.json"),
        path_tuning_record=os.path.join(log_dir, "tuning_record.json"),
    )

    if dry_run:
        print("[tvm_tune] dry_run=True — using LocalRunner (no real GPU timing)")
        runner = ms.runner.LocalRunner(timeout_sec=30)
    else:
        # RPC runner that distributes work across all 4 A30s.
        # Each GPU runs trials in parallel; MetaSchedule handles scheduling.
        runner = ms.runner.RPCRunner(
            rpc_config=ms.runner.RPCConfig(
                tracker_host=os.environ.get("TVM_TRACKER_HOST", "0.0.0.0"),
                tracker_port=int(os.environ.get("TVM_TRACKER_PORT", "9190")),
                tracker_key=os.environ.get("TVM_TRACKER_KEY", "a30"),
                session_timeout_sec=60,
            ),
            evaluator_config=ms.runner.EvaluatorConfig(
                number=5,           # 5 measurements per candidate
                repeat=1,
                min_repeat_ms=300,
                enable_cpu_cache_flush=False,
            ),
            max_workers=len(targets),  # 4 parallel workers
        )

    task = ms.Task(
        mod=mod,
        target=targets[0],  # All GPUs have the same arch; tune on one target
    )

    print(f"[tvm_tune] Starting MetaSchedule tuning — {trials} trials")
    print(f"  log_dir : {log_dir}")
    print(f"  num_gpus: {len(targets)}")
    t0 = time.time()

    ms.tune_tir(
        mod=mod,
        target=targets[0],
        max_trials_global=trials,
        num_trials_per_iter=64,
        runner=runner,
        database=database,
        cost_model=ms.cost_model.XGBModel(),
        space=ms.space_generator.PostOrderApply(),
        seed=42,
        work_dir=log_dir,
    )

    elapsed = time.time() - t0
    print(f"[tvm_tune] Tuning complete in {elapsed/60:.1f} min")
    return database


def compile_best(
    mod: tvm.IRModule,
    target: Target,
    database: ms.database.JSONDatabase,
    output_path: str,
) -> None:
    """Compile the best schedule in the database to a .so."""
    with ms.ApplyHistoryBest(database):
        with tvm.transform.PassContext(
            opt_level=3,
            config={"relay.backend.use_auto_scheduler": True}
        ):
            lib = tvm.build(mod, target=target)

    lib.export_library(output_path)
    print(f"[tvm_tune] Exported best schedule → {output_path}")


def print_winning_config(log_dir: str) -> None:
    """Parse the tuning log and print the winning tile configuration."""
    record_path = os.path.join(log_dir, "tuning_record.json")
    if not os.path.exists(record_path):
        print("[tvm_tune] No tuning_record.json found — skipping config dump")
        return

    best_latency = float("inf")
    best_config  = None

    with open(record_path) as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            lat = rec.get("run_secs")
            if lat and lat < best_latency:
                best_latency = lat
                best_config  = rec.get("args_info", {})

    print("\n── Winning tile configuration ─────────────────────────────────")
    print(f"  Best measured latency : {best_latency*1000:.3f} ms")
    if best_config:
        for k in TILE_CONFIG_KEYS:
            if k in best_config:
                print(f"  {k:20s}: {best_config[k]}")
    print("────────────────────────────────────────────────────────────────")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--module",    type=str, default="relax_module.json",
                    help="Path to lowered TIR module (from build_relax_module.py)")
    ap.add_argument("--trials",    type=int, default=1000)
    ap.add_argument("--num_gpus",  type=int, default=4,
                    help="Number of A30 GPUs in the cluster")
    ap.add_argument("--target",    type=str, default="cuda -arch=sm_86",
                    help="TVM target string for A30")
    ap.add_argument("--log_dir",   type=str, default="tuning_logs")
    ap.add_argument("--output",    type=str,
                    default="fused_rmsnorm_linear.so")
    ap.add_argument("--dry_run",   action="store_true",
                    help="Use LocalRunner — no real GPU required")
    args = ap.parse_args()

    mod     = load_module(args.module)
    targets = get_target(args.target, args.num_gpus)

    database = run_tuning(
        mod, targets, args.trials, args.log_dir, args.dry_run)

    compile_best(mod, targets[0], database, args.output)
    print_winning_config(args.log_dir)


if __name__ == "__main__":
    main()
