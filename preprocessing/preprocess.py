#!/usr/bin/env python3
"""
sample_and_preprocess.py

Each execution:
 - Reads the input Google-cluster CSV
 - Randomly samples 100 rows (or all rows if <100)
 - Preprocesses them into simplified columns:
     cpu cores, memory mb, latency sensitive, execution time,
     data size mb, cost_traditional, cost_serverless, cost_ratio, target platform
 - Writes the result to the output CSV

Enhancements:
 - Automatically detects and normalizes timestamp units (ns, µs, ms → seconds)
 - Estimates data size if missing, using memory × (execution_time / 100)
 - Estimates workload cost for traditional and serverless environments
 - Applies cost ratio rule (serverless only if cost_ratio < 5)
 - Defaults to 16 cores and 65536 MB (≈64 GB)

Usage:
  python sample_and_preprocess.py --input google_dataset.csv --output workload_dataset_sample.csv
  python sample_and_preprocess.py --input google_dataset.csv --output workload_dataset_sample.csv --cores 32 --mem_mb 131072
"""

import argparse
import pandas as pd
import numpy as np
import ast
import re

# -----------------------
# Helper Functions
# -----------------------
def safe_literal_eval(s):
    if pd.isna(s):
        return None
    try:
        return ast.literal_eval(s)
    except Exception:
        try:
            return ast.literal_eval(str(s).replace("null", "None").replace("'", '"'))
        except Exception:
            return None


def parse_resource_request(rr, machine_cores=16, machine_mem_mb=65536):
    """Parse resource_request (dict-like or string) into (cpu_cores, memory_mb)."""
    if pd.isna(rr):
        return (np.nan, np.nan)
    if isinstance(rr, dict):
        d = rr
    else:
        d = safe_literal_eval(str(rr).strip())

    cpu = np.nan
    mem = np.nan
    if isinstance(d, dict):
        cpu_raw = None
        mem_raw = None
        for k, v in d.items():
            lk = str(k).lower()
            if "cpu" in lk:
                try:
                    cpu_raw = float(v)
                except:
                    cpu_raw = None
            if "mem" in lk:
                try:
                    mem_raw = float(v)
                except:
                    mem_raw = None
        if cpu_raw is not None:
            cpu = cpu_raw * machine_cores if cpu_raw <= 1.0 else cpu_raw
        if mem_raw is not None:
            if mem_raw <= 1.0:
                mem = mem_raw * machine_mem_mb
            elif mem_raw < 256:
                mem = mem_raw * 1024
            else:
                mem = mem_raw
        return (cpu, mem)

    # Fallback for text or numeric values
    s = str(rr)
    parts = re.split(r"[,\s;/]+", s)
    nums = []
    for p in parts:
        try:
            nums.append(float(re.sub("[^0-9.eE+-]", "", p)))
        except:
            pass
    if len(nums) >= 2:
        cpu_cand, mem_cand = nums[0], nums[1]
        cpu = cpu_cand if cpu_cand > 1.0 else cpu_cand * machine_cores
        mem = mem_cand if mem_cand > 1.0 else mem_cand * machine_mem_mb
        return (cpu, mem)
    if len(nums) == 1:
        val = nums[0]
        if val <= 64:
            return (val, np.nan)
        else:
            return (np.nan, val)
    return (np.nan, np.nan)


def epoch_to_seconds(val):
    """Convert epoch timestamps (ns, µs, ms, s) → seconds."""
    if pd.isna(val):
        return np.nan
    try:
        v = float(val)
    except:
        return np.nan
    av = abs(v)
    if av > 1e17:  # nanoseconds
        return v / 1e9
    if av > 1e14:  # microseconds
        return v / 1e6
    if av > 1e11:  # milliseconds
        return v / 1e3
    return v


def derive_latency_sensitive(row):
    """Heuristic for latency_sensitive flag."""
    pri = row.get("priority", np.nan)
    sc = row.get("scheduling_class", np.nan)
    try:
        pri_n = float(pri)
    except:
        pri_n = np.nan
    try:
        sc_n = float(sc)
    except:
        sc_n = np.nan
    if (not np.isnan(pri_n) and pri_n <= 2) or (not np.isnan(sc_n) and sc_n <= 1):
        return 1
    combined = " ".join(
        [str(row.get("collection_type", "")).lower(), str(row.get("collection_name", "")).lower()]
    )
    if any(k in combined for k in ("api", "realtime", "interactive", "latency", "foreground")):
        return 1
    return 0


def normalize_failed(val):
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    return s in ("1", "true", "t", "yes", "y", "fail", "failed", "failure")


def heuristic_target(cpu, mem, exec_time, failed_flag, cost_ratio=np.nan):
    """
    Classify workload as serverless or traditional based on:
    - Execution time
    - CPU usage
    - Memory usage
    - Failure status
    - Cost ratio (new rule)
    """
    # If failed previously, prefer traditional
    if failed_flag:
        return "traditional"

    # Convert safely
    try:
        et = float(exec_time)
    except:
        et = np.nan
    try:
        cost_ratio = float(cost_ratio)
    except:
        cost_ratio = np.nan

    # Rules
    cond_et = (not np.isnan(et)) and (et <= 300)
    cond_cpu = np.isnan(cpu) or (cpu <= 2)
    cond_mem = np.isnan(mem) or (mem <= 2048)
    cond_cost = (not np.isnan(cost_ratio)) and (cost_ratio < 5)

    if cond_et and cond_cpu and cond_mem and cond_cost:
        return "serverless"
    return "traditional"


# -----------------------
# Preprocess DataFrame
# -----------------------
def preprocess_df(df_sample, machine_cores=16, machine_mem_mb=65536):
    df = df_sample.copy()

    parsed = df.get("resource_request", pd.Series([np.nan] * len(df))).apply(
        lambda x: parse_resource_request(x, machine_cores, machine_mem_mb)
    )
    df["cpu_cores"] = parsed.apply(lambda t: t[0])
    df["memory_mb"] = parsed.apply(lambda t: t[1])

    # Fill memory if missing using assigned_memory
    if "assigned_memory" in df.columns:
        def assign_mem(row):
            if not pd.isna(row["memory_mb"]):
                return row["memory_mb"]
            v = row.get("assigned_memory", np.nan)
            try:
                fv = float(v)
                if fv <= 1.0:
                    return fv * machine_mem_mb
                if fv < 10000:
                    return fv
                return fv / (1024 * 1024)
            except:
                return np.nan
        df["memory_mb"] = df.apply(assign_mem, axis=1)

    # execution time with auto unit detection
    if "start_time" in df.columns and "end_time" in df.columns:
        start = pd.to_numeric(df["start_time"], errors="coerce")
        end = pd.to_numeric(df["end_time"], errors="coerce")
        diff = end - start

        # Auto-detect time unit scale
        median_diff = diff.median(skipna=True)
        if pd.notna(median_diff):
            if median_diff > 1e9:
                print("Detected nanoseconds timestamps — converting to seconds (/1e9).")
                diff = diff / 1e9
            elif median_diff > 1e6:
                print("Detected microseconds timestamps — converting to seconds (/1e6).")
                diff = diff / 1e6
            elif median_diff > 1e3:
                print("Detected milliseconds timestamps — converting to seconds (/1e3).")
                diff = diff / 1e3
            else:
                print("Detected seconds timestamps — keeping as-is.")
        df["execution_time"] = diff
        df.loc[df["execution_time"] <= 0, "execution_time"] = np.nan
    else:
        df["execution_time"] = np.nan

    # latency sensitive
    df["latency_sensitive"] = df.apply(derive_latency_sensitive, axis=1)

    # data size estimation (if not present)
    if "data_size_mb" in df.columns:
        df["data_size_mb"] = pd.to_numeric(df["data_size_mb"], errors="coerce")
    elif "data_size" in df.columns:
        df["data_size_mb"] = pd.to_numeric(df["data_size"], errors="coerce")
    else:
        df["data_size_mb"] = (df["memory_mb"] * df["execution_time"] / 100).fillna(0)

    # ----------------------------
    # 💰 Cost Estimation Section
    # ----------------------------
    P_CPU = 0.0316          # $ per CPU core-hour
    P_MEM = 0.0045          # $ per GB-hour
    P_GB_SEC = 0.00001667   # $ per GB-second for serverless
    P_REQ = 0.0000002       # $ per request

    mem_gb = df["memory_mb"] / 1024
    runtime_hr = df["execution_time"] / 3600

    df["cost_traditional"] = ((df["cpu_cores"] * P_CPU) + (mem_gb * P_MEM)) * runtime_hr
    df["cost_serverless"] = P_REQ + (P_GB_SEC * mem_gb * df["execution_time"])
    df["cost_traditional"] = df["cost_traditional"].fillna(0)
    df["cost_serverless"] = df["cost_serverless"].fillna(0)
    df["cost_ratio"] = df["cost_serverless"] / (df["cost_traditional"] + 1e-9)

    # failed normalization
    df["failed_flag"] = df.get("failed", pd.Series([False] * len(df))).apply(normalize_failed)

    # target platform classification (updated with cost_ratio)
    df["target_platform"] = df.apply(
        lambda r: heuristic_target(
            r["cpu_cores"],
            r["memory_mb"],
            r["execution_time"],
            r["failed_flag"],
            r["cost_ratio"]
        ),
        axis=1,
    )

    # select final columns
    out = pd.DataFrame(
        {
            "cpu cores": pd.to_numeric(df["cpu_cores"], errors="coerce"),
            "memory mb": pd.to_numeric(df["memory_mb"], errors="coerce"),
            "latency sensitive": pd.to_numeric(df["latency_sensitive"], errors="coerce").fillna(0).astype(int),
            "execution time": pd.to_numeric(df["execution_time"], errors="coerce"),
            "data size mb": pd.to_numeric(df["data_size_mb"], errors="coerce"),
            "cost traditional": pd.to_numeric(df["cost_traditional"], errors="coerce"),
            "cost serverless": pd.to_numeric(df["cost_serverless"], errors="coerce"),
            "cost ratio": pd.to_numeric(df["cost_ratio"], errors="coerce"),
            "target platform": df["target_platform"],
        }
    )
    return out


# -----------------------
# Main: Sample + Process
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Sample 100 rows, preprocess, and save simplified CSV.")
    parser.add_argument("--input", "-i", required=True, help="Input Google dataset CSV")
    parser.add_argument("--output", "-o", default="workload_dataset_sample.csv", help="Output CSV path")
    parser.add_argument("--cores", type=int, default=16, help="Machine total cores (default: 16)")
    parser.add_argument("--mem_mb", type=int, default=65536, help="Machine total memory in MB (default: 65536 ≈ 64 GB)")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling")
    args = parser.parse_args()

    # Read dataset as strings to safely parse complex JSON-like fields
    df_raw = pd.read_csv(args.input, dtype=str)
    n_rows = len(df_raw)
    sample_n = min(100, n_rows)
    if args.seed is None:
        df_sample = df_raw.sample(n=sample_n)
    else:
        df_sample = df_raw.sample(n=sample_n, random_state=int(args.seed))

    processed = preprocess_df(df_sample, machine_cores=args.cores, machine_mem_mb=args.mem_mb)
    processed.to_csv(args.output, index=False)
    print(f"\n[OK] Sampled {len(processed)} rows (from {n_rows}) and saved to {args.output}")
    print(f"Machine config: {args.cores} cores, {args.mem_mb} MB memory")
    print("\nPreview:")
    print(processed.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
