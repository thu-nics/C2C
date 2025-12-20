"""
批量测试多个checkpoint的一致性

自动扫描指定目录下的所有checkpoint（checkpoint-10, checkpoint-20, ..., final），
对每个checkpoint运行一致性检查，并记录结果。

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
    查找所有checkpoint目录，返回排序后的列表。
    
    Returns:
        List of (checkpoint_name, checkpoint_path) tuples, sorted by checkpoint number
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        raise FileNotFoundError(f"目录不存在: {base_dir}")
    
    checkpoints = []
    
    # 查找所有 checkpoint-* 目录
    for item in base_path.iterdir():
        if item.is_dir():
            name = item.name
            # 匹配 checkpoint-数字 格式
            match = re.match(r'checkpoint-(\d+)', name)
            if match:
                step = int(match.group(1))
                checkpoints.append((step, name, str(item)))
            elif name == "final":
                # final 放在最后
                checkpoints.append((float('inf'), name, str(item)))
    
    # 按 step 排序
    checkpoints.sort(key=lambda x: x[0])
    return [(name, path) for _, name, path in checkpoints]


def run_consistency_check(config_path: str, checkpoint_dir: str, output_file: str = None) -> Dict[str, Any]:
    """
    运行单个checkpoint的一致性检查。
    
    Args:
        config_path: 基础配置文件路径
        checkpoint_dir: checkpoint目录路径
        output_file: 输出日志文件路径（追加写入）。会同时实时打印到终端。
    
    Returns:
        简单的运行信息（是否成功、返回码）
    """
    # 基础配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    # 更新checkpoint路径
    config['rosetta']['checkpoints_dir'] = checkpoint_dir

    # 创建临时配置文件
    import tempfile
    script_dir = Path(__file__).parent
    check_script = script_dir / "check_rosetta_consistency.py"

    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, encoding='utf-8') as tmp:
        json.dump(config, tmp, indent=2, ensure_ascii=False)
        tmp_config_path = tmp.name

    # 运行检查脚本：将 stdout/stderr 合并后逐行实时转发到终端与文件
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
            # 实时打印到终端
            print(line, end="")
            # 同时写到文件
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
    parser = argparse.ArgumentParser(description="批量测试多个checkpoint的一致性")
    parser.add_argument("--base-dir", type=str, default="local/checkpoints/include_response_proj_zero")
    parser.add_argument("--config", type=str, default="rosetta_consistency_config.json",
                       help="基础配置文件路径")
    parser.add_argument("--output", type=str, default=None,
                       help="输出日志文件路径（原样追加写入），默认：base_dir/consistency_output.log")
    
    args = parser.parse_args()
    
    # 查找所有checkpoint
    checkpoints = find_checkpoints(args.base_dir)
    
    if not checkpoints:
        print(f"❌ 在 {args.base_dir} 中未找到任何checkpoint")
        return
    
    print(f"📋 找到 {len(checkpoints)} 个checkpoint:")
    for name, path in checkpoints:
        print(f"  - {name}")
    
    # 设置输出路径
    if args.output is None:
        args.output = os.path.join(args.base_dir, "consistency_output.log")
    
    # 运行测试
    start_time = datetime.now()
    
    print(f"\n🚀 开始批量测试...")
    print(f"📁 输出将同时打印到终端并追加写入: {args.output}\n")
    
    # 写入文件头
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

