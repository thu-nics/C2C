import argparse
import json
import os
from typing import Set, Dict, Any

import matplotlib.pyplot as plt
from matplotlib_venn import venn3


def load_regions(json_path: str) -> Dict[str, Set[int]]:
    with open(json_path, 'r') as f:
        data: Dict[str, Any] = json.load(f)
    regions: Dict[str, Set[int]] = {}
    for k in [
        "rosetta_only", "slm_only", "llm_only",
        "rosetta_slm", "rosetta_llm", "slm_llm",
        "all_three"
    ]:
        regions[k] = set(data.get(k, []))
    return regions


def reconstruct_sets(regions: Dict[str, Set[int]]) -> tuple[Set[int], Set[int], Set[int]]:
    A_only = regions.get("rosetta_only", set())
    B_only = regions.get("slm_only", set())
    C_only = regions.get("llm_only", set())
    AB = regions.get("rosetta_slm", set())
    AC = regions.get("rosetta_llm", set())
    BC = regions.get("slm_llm", set())
    ABC = regions.get("all_three", set())

    A = set().union(A_only, AB, AC, ABC)
    B = set().union(B_only, AB, BC, ABC)
    C = set().union(C_only, AC, BC, ABC)
    return A, B, C


def plot_venn_from_json(json_path: str, out_path: str,
                        label_a: str = "C2C", label_b: str = "Receiver", label_c: str = "Sharer",
                        title: str | None = None):
    regions = load_regions(json_path)
    A, B, C = reconstruct_sets(regions)

    FIGSIZE = (9, 9)
    DPI = 300

    fig = plt.figure(figsize=FIGSIZE, constrained_layout=True)
    ax = fig.add_subplot(111)
    v = venn3([A, B, C], set_labels=(label_a, label_b, label_c))

    # Improve visibility and reduce empty margins
    if v is not None:
        if hasattr(v, 'set_labels') and v.set_labels is not None:
            for txt in v.set_labels:
                if txt is not None:
                    txt.set_fontsize(30)
        for sid in ("100","010","001","110","101","011","111"):
            lbl = v.get_label_by_id(sid)
            if lbl is not None:
                lbl.set_fontsize(30)
        for sid in ("100","010","001","110","101","011","111"):
            patch = v.get_patch_by_id(sid)
            if patch is not None:
                patch.set_linewidth(1.5)

    # if title is None:
        # title = f"Correct Answer Overlap: {label_a} vs {label_b}(Qwen3-0.6B) vs {label_c}(Qwen3-4B)"
    ax.set_position([0.03, 0.03, 0.96, 0.96])
    plt.title(title, fontsize=16, pad=8)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=DPI, bbox_inches=None)
    plt.close()
    print(f"Saved Venn diagram to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Plot a Venn diagram directly from precomputed region JSON")
    parser.add_argument("--json", type=str, required=True, help="Path to regions JSON file")
    parser.add_argument("--out", type=str, required=True, help="Output image path (e.g., .png)")
    parser.add_argument("--label_a", type=str, default="C2C", help="Label for set A (default: C2C)")
    parser.add_argument("--label_b", type=str, default="Receiver", help="Label for set B (default: Receiver)")
    parser.add_argument("--label_c", type=str, default="Sharer", help="Label for set C (default: Sharer)")
    parser.add_argument("--title", type=str, default=None, help="Optional custom title")
    args = parser.parse_args()

    plot_venn_from_json(args.json, args.out, args.label_a, args.label_b, args.label_c, args.title)


if __name__ == "__main__":
    main()


