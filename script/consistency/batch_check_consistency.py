"""
Batch consistency evaluation across multiple checkpoints

This script automatically scans a base directory for all checkpoints
(e.g., checkpoint-10, checkpoint-20, ..., final), runs the consistency
check for each checkpoint, and records the results.

Usage:
    python batch_check_consistency.py --base-dir local/checkpoints/qwen3_0.6b+qwen3_32b_include --config rosetta_consistency_config.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime


def find_checkpoints(base_dir: str) -> List[tuple]:
    """
    Find all checkpoint directories and return a sorted list.
    
    Returns:
        List of (checkpoint_name, checkpoint_path) tuples, sorted by checkpoint step.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"Directory not found: {base_dir}")
    
    checkpoints = []
    
    # Find all checkpoint-* directories
    for item in base_path.iterdir():
        if item.is_dir():
            name = item.name
            # Match the "checkpoint-<number>" pattern
            match = re.match(r'checkpoint-(\d+)', name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, name, str(item)))
            elif name == "final":
                # Put "final" at the end
                checkpoints.append((float('inf'), name, str(item)))
    
    # Sort by step
    checkpoints.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in checkpoints]


def run_consistency_check(config_path: str, checkpoint_dir: str, output_file: str = None) -> Dict[str, Any]:
    """
    Run consistency check for a single checkpoint.
    
    Args:
        config_path: Path to the base config JSON.
        checkpoint_dir: Path to the checkpoint directory.
        output_file: Output log file path (append). Output is also streamed to stdout.
    
    Returns:
        Simple run info (success flag, return code).
    """
    # Base config
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # Update checkpoint path
    config['rosetta']['checkpoints_dir'] = checkpoint_dir

    # Create a temporary config file
    import tempfile
    script_dir = Path(__file__).parent
    check_script = script_dir / "check_rosetta_consistency.py"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(config, tmp, indent=2, ensure_ascii=False)
        tmp_config_path = tmp.name

    # Run the check script: merge stdout/stderr and stream line-by-line to stdout and file
    import subprocess
    cmd = [sys.executable, str(check_script), "--config", tmp_config_path]

    log_fh: Optional[Any] = None
    try:
        if output_file:
            log_fh = open(output_file, "a", encoding="utf-8")

        proc = subprocess.Popen(
            cmd,
            cwd=str(script_dir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        assert proc.stdout is not None
        for line in proc.stdout:
            # Stream to stdout
            print(line, end="")
            # Also write to file
            if log_fh:
                log_fh.write(line)
                log_fh.flush()

        returncode = proc.wait()
        return {
            "checkpoint": checkpoint_dir,
            "success": returncode == 0,
            "returncode": returncode,
        }
    finally:
        if log_fh:
            log_fh.write("\n")
            log_fh.flush()
            log_fh.close()
        if os.path.exists(tmp_config_path):
            os.unlink(tmp_config_path)


def main():
    parser = argparse.ArgumentParser(description="Batch consistency evaluation across multiple checkpoints")
    parser.add_argument("--base-dir", type=str, default="local/checkpoints/include_response_proj_zero")
    parser.add_argument("--config", type=str, default="rosetta_consistency_config.json",
                       help="Path to the base config JSON")
    parser.add_argument("--output", type=str, default=None,
                       help="Output log file path (append as-is). Default: base_dir/consistency_output.log")
    
    args = parser.parse_args()
    
    # Find all checkpoints
    checkpoints = find_checkpoints(args.base_dir)
    
    if not checkpoints:
        print(f"❌ No checkpoints found in {args.base_dir}")
        return
    
    print(f"📋 Found {len(checkpoints)} checkpoints:")
    for name, path in checkpoints:
        print(f"  - {name}")
    
    # Set output path
    if args.output is None:
        args.output = os.path.join(args.base_dir, "consistency_output.log")
    
    # Run batch tests
    start_time = datetime.now()
    
    print(f"\n🚀 Starting batch evaluation...")
    print(f"📁 Output will be printed to stdout and appended to: {args.output}\n")
    
    # Write file header
    with open(args.output, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"[batch_check_consistency] start_time={start_time.isoformat()}\n")
        f.write(f"base_dir={args.base_dir}\n")
        f.write(f"config={args.config}\n")
        f.write("=" * 80 + "\n\n")

    for idx, (name, path) in enumerate(checkpoints, 1):
        header = (
            "\n" + "#" * 80 + "\n"
            f"[{idx}/{len(checkpoints)}] checkpoint={name}\n"
            f"path={path}\n"
            f"time={datetime.now().isoformat()}\n"
            + "#" * 80 + "\n"
        )
        print(header, end="")
        with open(args.output, "a", encoding="utf-8") as f:
            f.write(header)

        run_consistency_check(args.config, path, args.output)
    
    end_time = datetime.now()
    duration = (end_time - start_time).total_seconds()
    
    footer = (
        "\n" + "=" * 80 + "\n"
        f"[batch_check_consistency] end_time={end_time.isoformat()} duration_sec={duration:.1f}\n"
        "=" * 80 + "\n"
    )
    print(footer, end="")
    with open(args.output, "a", encoding="utf-8") as f:
        f.write(footer)


if __name__ == "__main__":
    main()

