from __future__ import annotations

import argparse
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Export features for unseen data using the main export_features_csv.py")
    parser.add_argument("--raw-root", default="data/raw/github_unseen", help="Raw unseen root")
    parser.add_argument("--out", default="data/processed/github_features_unseen.csv", help="Output CSV path")
    parser.add_argument("--refine", action="store_true", help="Pass --refine to exporter")
    parser.add_argument("--inject-noise", action="store_true", help="Pass --inject-noise to exporter")
    parser.add_argument("--noise-level", type=float, default=0.01, help="Noise level if injecting noise")
    parser.add_argument("--fillna0", action="store_true", help="Pass --fillna0 to exporter")
    args = parser.parse_args()

    cmd = [
        sys.executable,
        "code/scripts/export_features_csv.py",
        "--raw-root", args.raw_root,
        "--out", args.out,
    ]

    if args.refine:
        cmd.append("--refine")
    if args.inject_noise:
        cmd.extend(["--inject-noise", "--noise-level", str(args.noise_level)])
    if args.fillna0:
        cmd.append("--fillna0")

    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
