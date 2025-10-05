import os
import re
import json
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _convert_dict_keys_to_ints(obj: Any) -> Any:
    """
    Recursively convert dictionary keys that look like integers back to int.
    Mirrors the helper in `rosetta.model.wrapper.RosettaModel` but duplicated here
    to avoid import-time coupling for analysis scripts.
    """
    if isinstance(obj, dict):
        new_obj = {}
        for key, value in obj.items():
            if isinstance(key, str) and key.lstrip('-').isdigit():
                new_key = int(key)
            else:
                new_key = key
            new_obj[new_key] = _convert_dict_keys_to_ints(value)
        return new_obj
    if isinstance(obj, list):
        return [_convert_dict_keys_to_ints(v) for v in obj]
    return obj


def load_aggregator_meta(checkpoint_dir: str) -> Dict[int, Dict[str, Any]]:
    """
    Load per-aggregator metadata: temperature, num_options, logits, and probs.

    Returns a dict keyed by aggregator_id with fields:
      - temperature: float
      - logits: np.ndarray [num_options]
      - probs: np.ndarray [num_options]
    """
    meta: Dict[int, Dict[str, Any]] = {}

    pt_files = [fn for fn in os.listdir(checkpoint_dir) if re.match(r'^aggregator_\d+\.pt$', fn)]
    pt_files = sorted(pt_files, key=lambda x: int(re.findall(r'\d+', x)[0]))

    for fn in pt_files:
        agg_id = int(re.findall(r'\d+', fn)[0])
        pt_path = os.path.join(checkpoint_dir, fn)
        json_path = os.path.join(checkpoint_dir, f'aggregator_{agg_id}.json')

        temperature = 1.0
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    j = json.load(f)
                init_args = j.get('init_args', {})
                if isinstance(init_args, dict) and 'final_temperature' in init_args:
                    temperature = float(init_args['final_temperature'])
            except Exception:
                pass

        state = torch.load(pt_path, map_location='cpu')
        logits = state['gate_logits'].detach().float().cpu().numpy()

        scaled = logits / max(temperature, 1e-6)
        e = np.exp(scaled - np.max(scaled))
        probs = e / e.sum()

        meta[agg_id] = {
            'temperature': float(temperature),
            'logits': logits,
            'probs': probs,
            'file': fn,
        }

    return meta


def load_configs(checkpoint_dir: str) -> Tuple[Dict[int, Dict[int, Dict[int, int]]], Dict[int, Dict[int, Dict[int, List[Tuple[int, int]]]]]]:
    """
    Load aggregator_config and projector_config, converting stringified int keys.

    Returns:
      aggregator_cfg: {target_model_idx: {source_model_idx: {target_layer_idx: aggregator_idx}}}
      projector_cfg: {target_model_idx: {source_model_idx: {target_layer_idx: [(source_layer_idx, projector_idx), ...]}}}
    """
    with open(os.path.join(checkpoint_dir, 'aggregator_config.json'), 'r') as f:
        aggregator_cfg = _convert_dict_keys_to_ints(json.load(f))

    with open(os.path.join(checkpoint_dir, 'projector_config.json'), 'r') as f:
        projector_cfg = _convert_dict_keys_to_ints(json.load(f))

    return aggregator_cfg, projector_cfg


def plot_overall_heatmap(aggregator_meta: Dict[int, Dict[str, Any]], out_path: str) -> None:
    if not aggregator_meta:
        return
    ids = sorted(aggregator_meta.keys())
    probs = np.stack([aggregator_meta[i]['probs'] for i in ids], axis=0)

    fig, ax = plt.subplots(figsize=(max(6, probs.shape[1] * 1.0), max(4, probs.shape[0] * 0.25)))
    im = ax.imshow(probs, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_xlabel('Option index')
    ax.set_ylabel('Aggregator id')
    ax.set_title('WeightedAggregator softmax probabilities')
    ax.set_yticks(range(len(ids)))
    ax.set_yticklabels(ids, fontsize=6)
    ax.set_xticks(range(probs.shape[1]))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='Probability')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_prob_bar(aggregator_meta: Dict[int, Dict[str, Any]], out_path: str) -> None:
    if not aggregator_meta:
        return
    ids = sorted(aggregator_meta.keys())
    probs = np.stack([aggregator_meta[i]['probs'] for i in ids], axis=0)
    mean_probs = probs.mean(axis=0)

    fig, ax = plt.subplots(figsize=(max(6, probs.shape[1] * 0.8), 4))
    ax.bar(range(mean_probs.shape[0]), mean_probs)
    ax.set_xlabel('Option index')
    ax.set_ylabel('Mean probability across aggregators')
    ax.set_title('Mean softmax probability across all aggregators')
    for i, v in enumerate(mean_probs):
        ax.text(i, float(v) + 0.01, f'{float(v):.2f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylim(0, min(1.05, max(1.0, float(mean_probs.max()) + 0.1)))
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bipartite_for_layer(
    source_model_idx: int,
    target_model_idx: int,
    target_layer_idx: int,
    pair_list: List[Tuple[int, int]],
    probs: np.ndarray,
    out_path: str,
) -> None:
    """
    Draw a bipartite graph for a specific target layer.

    Left nodes: source layers (S{source_model_idx}:L{src_layer})
    Right nodes: single target layer (T{target_model_idx}:L{target_layer_idx})
    Edges colored by weight (probability), width proportional to weight.
    """
    num_sources = len(pair_list)
    if probs.shape[0] != num_sources:
        # Align by trunc/pad
        if probs.shape[0] > num_sources:
            probs = probs[:num_sources]
        else:
            probs = np.pad(probs, (0, num_sources - probs.shape[0]))

    # Layout: left column for sources, right column for the single target node
    fig, ax = plt.subplots(figsize=(6, max(4, num_sources * 0.4)))
    ax.axis('off')

    # Positions
    left_x = 0.1
    right_x = 0.9
    ys = np.linspace(0.1, 0.9, num_sources) if num_sources > 1 else np.array([0.5])

    # Draw source nodes
    for i, (src_layer, _proj_idx) in enumerate(pair_list):
        ax.plot([left_x], [ys[i]], 'o', color='tab:blue')
        ax.text(left_x - 0.02, ys[i], f'S{source_model_idx}:L{src_layer}', ha='right', va='center', fontsize=8)

    # Draw target node
    ax.plot([right_x], [0.5], 'o', color='tab:orange')
    ax.text(right_x + 0.02, 0.5, f'T{target_model_idx}:L{target_layer_idx}', ha='left', va='center', fontsize=9)

    # Edges with color/width by probs
    cmap = plt.get_cmap('viridis')
    for i, p in enumerate(probs):
        color = cmap(float(p))
        linewidth = max(0.5, 6.0 * float(p))
        ax.plot([left_x, right_x], [ys[i], 0.5], color=color, linewidth=linewidth, alpha=0.9)
        ax.text((left_x + right_x) / 2.0, (ys[i] + 0.5) / 2.0, f'{float(p):.2f}', fontsize=7, ha='center', va='center')

    # Colorbar reference
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Probability')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bipartite_grid(
    source_model_idx: int,
    target_model_idx: int,
    layer_to_agg: Dict[int, int],
    projector_cfg_for_pair: Dict[int, List[Tuple[int, int]]],
    aggregator_meta: Dict[int, Dict[str, Any]],
    out_path: str,
    cols: int = 7,
) -> None:
    """
    Draw a grid figure where each subplot shows the bipartite mapping for one target layer.
    Color and width of edges indicate the aggregator's softmax probability for each source.
    """
    layers = sorted(layer_to_agg.keys())
    if not layers:
        return

    # Grid layout
    num_layers = len(layers)
    rows = int(np.ceil(num_layers / float(cols)))
    cell_w, cell_h = 3.0, 2.4
    fig_w = max(6.0, cols * cell_w)
    fig_h = max(4.0, rows * cell_h)
    fig, axes = plt.subplots(rows, cols, figsize=(fig_w, fig_h))
    axes = np.array(axes).reshape(rows, cols)

    cmap = plt.get_cmap('viridis')
    vmin, vmax = 0.0, 1.0

    for idx, tgt_layer in enumerate(layers):
        r, c = divmod(idx, cols)
        ax = axes[r, c]
        ax.axis('off')

        aggregator_idx = layer_to_agg[tgt_layer]
        agg = aggregator_meta.get(aggregator_idx)
        pair_list = projector_cfg_for_pair.get(tgt_layer)
        if agg is None or not pair_list:
            ax.text(0.5, 0.5, f'L{tgt_layer}: (no data)', ha='center', va='center', fontsize=8)
            continue

        probs = agg['probs']
        # Align length
        if probs.shape[0] != len(pair_list):
            if probs.shape[0] > len(pair_list):
                probs = probs[:len(pair_list)]
            else:
                probs = np.pad(probs, (0, len(pair_list) - probs.shape[0]))

        # Node positions within subplot
        left_x, right_x = 0.1, 0.9
        n_src = len(pair_list)
        ys = np.linspace(0.1, 0.9, n_src) if n_src > 1 else np.array([0.5])

        # Draw source nodes
        for i, (src_layer, _proj_idx) in enumerate(pair_list):
            ax.plot([left_x], [ys[i]], 'o', color='tab:blue', markersize=3)
            ax.text(left_x - 0.02, ys[i], f'S{source_model_idx}:L{src_layer}', ha='right', va='center', fontsize=6)

        # Draw target node
        ax.plot([right_x], [0.5], 'o', color='tab:orange', markersize=4)
        ax.text(right_x + 0.02, 0.5, f'T{target_model_idx}:L{tgt_layer}', ha='left', va='center', fontsize=7)

        # Draw edges
        for i, p in enumerate(probs):
            color = cmap(float(p))
            lw = max(0.4, 5.0 * float(p))
            ax.plot([left_x, right_x], [ys[i], 0.5], color=color, linewidth=lw, alpha=0.9)
        # Title for each cell
        ax.set_title(f'Layer {tgt_layer} (agg {aggregator_idx})', fontsize=8)

    # Hide any remaining empty subplots
    for idx in range(num_layers, rows * cols):
        r, c = divmod(idx, cols)
        axes[r, c].axis('off')

    # Shared colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
    cbar.set_label('Probability')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_bipartite_all(
    source_model_idx: int,
    target_model_idx: int,
    layer_to_agg: Dict[int, int],
    projector_cfg_for_pair: Dict[int, List[Tuple[int, int]]],
    aggregator_meta: Dict[int, Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Draw a single bipartite figure with ALL source layers on the left and ALL target layers on the right.
    Each edge connects a source layer to a target layer with color/width proportional to the aggregator weight.
    """
    # Determine valid target layers (those that have projector pairs and aggregator meta)
    valid_targets: List[int] = []
    all_source_layers: List[int] = []
    edges: List[Tuple[int, int, float]] = []  # (src_layer, tgt_layer, weight)

    for tgt_layer, agg_idx in layer_to_agg.items():
        pair_list = projector_cfg_for_pair.get(tgt_layer)
        agg = aggregator_meta.get(agg_idx)
        if not pair_list or agg is None:
            continue
        probs = agg['probs']
        # align
        if probs.shape[0] != len(pair_list):
            if probs.shape[0] > len(pair_list):
                probs = probs[:len(pair_list)]
            else:
                probs = np.pad(probs, (0, len(pair_list) - probs.shape[0]))
        valid_targets.append(tgt_layer)
        for i, (src_layer, _proj_idx) in enumerate(pair_list):
            w = float(probs[i])
            all_source_layers.append(src_layer)
            edges.append((src_layer, tgt_layer, w))

    if not edges:
        return

    # Unique sorted nodes
    left_nodes = sorted(sorted(set(all_source_layers)))
    right_nodes = sorted(sorted(set(valid_targets)))

    # Layout positions
    n_left = len(left_nodes)
    n_right = len(right_nodes)
    left_y = np.linspace(0.05, 0.95, n_left) if n_left > 1 else np.array([0.5])
    right_y = np.linspace(0.05, 0.95, n_right) if n_right > 1 else np.array([0.5])
    left_x, right_x = 0.05, 0.95

    # Figure sizing: scale height with max(n_left, n_right)
    height = max(5.0, 0.25 * max(n_left, n_right) + 2.0)
    fig, ax = plt.subplots(figsize=(14, height))
    ax.axis('off')

    # Draw nodes
    for i, src in enumerate(left_nodes):
        ax.plot([left_x], [left_y[i]], 'o', color='tab:blue', markersize=5)
        ax.text(left_x + 0.01, left_y[i], f'S{source_model_idx}:L{src}', ha='left', va='center', fontsize=7)
    for j, tgt in enumerate(right_nodes):
        ax.plot([right_x], [right_y[j]], 'o', color='tab:orange', markersize=6)
        ax.text(right_x - 0.01, right_y[j], f'T{target_model_idx}:L{tgt}', ha='right', va='center', fontsize=8)

    # Build quick index to y positions
    left_pos = {src: left_y[i] for i, src in enumerate(left_nodes)}
    right_pos = {tgt: right_y[j] for j, tgt in enumerate(right_nodes)}

    # Draw edges
    cmap = plt.get_cmap('viridis')
    for src, tgt, w in edges:
        color = cmap(w)
        lw = max(0.4, 6.0 * w)
        ax.plot([left_x, right_x], [left_pos[src], right_pos[tgt]], color=color, linewidth=lw, alpha=0.9)

    # Title and colorbar
    ax.set_title(f'Bipartite map: Source model {source_model_idx} layers to Target model {target_model_idx} layers', fontsize=12)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0.0, vmax=1.0))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Aggregator probability (edge weight)')

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mapping_heatmap_all(
    source_model_idx: int,
    target_model_idx: int,
    layer_to_agg: Dict[int, int],
    projector_cfg_for_pair: Dict[int, List[Tuple[int, int]]],
    aggregator_meta: Dict[int, Dict[str, Any]],
    out_path: str,
) -> None:
    """
    Build an M x N matrix heatmap where rows are all source layers and columns are all target layers.
    Each column corresponds to a target layer; entries are aggregator probabilities for each source layer.
    """
    # Collect nodes and weights
    all_source_layers: List[int] = []
    valid_targets: List[int] = []
    column_entries: Dict[int, List[Tuple[int, float]]] = {}  # tgt_layer -> list of (src_layer, weight)

    for tgt_layer, agg_idx in layer_to_agg.items():
        pair_list = projector_cfg_for_pair.get(tgt_layer)
        agg = aggregator_meta.get(agg_idx)
        if not pair_list or agg is None:
            continue
        probs = agg['probs']
        if probs.shape[0] != len(pair_list):
            if probs.shape[0] > len(pair_list):
                probs = probs[:len(pair_list)]
            else:
                probs = np.pad(probs, (0, len(pair_list) - probs.shape[0]))
        valid_targets.append(tgt_layer)
        entries = []
        for i, (src_layer, _proj_idx) in enumerate(pair_list):
            w = float(probs[i])
            all_source_layers.append(src_layer)
            entries.append((src_layer, w))
        column_entries[tgt_layer] = entries

    if not column_entries:
        return

    left_nodes = sorted(sorted(set(all_source_layers)))
    right_nodes = sorted(sorted(set(valid_targets)))

    # Build matrix
    W = np.zeros((len(left_nodes), len(right_nodes)), dtype=np.float32)
    src_index = {s: i for i, s in enumerate(left_nodes)}
    tgt_index = {t: j for j, t in enumerate(right_nodes)}
    for tgt_layer, entries in column_entries.items():
        j = tgt_index[tgt_layer]
        for src_layer, w in entries:
            i = src_index[src_layer]
            W[i, j] = w

    # Plot heatmap
    fig_w = max(8.0, 0.3 * len(right_nodes) + 4.0)
    fig_h = max(6.0, 0.25 * len(left_nodes) + 3.0)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(W, aspect='auto', cmap='viridis', vmin=0.0, vmax=1.0)
    ax.set_title(f'M×N mapping heatmap (S{source_model_idx} → T{target_model_idx})')
    ax.set_xlabel('Target layer index')
    ax.set_ylabel('Source layer index')
    ax.set_xticks(range(len(right_nodes)))
    ax.set_xticklabels(right_nodes, rotation=90, fontsize=7)
    ax.set_yticks(range(len(left_nodes)))
    ax.set_yticklabels(left_nodes, fontsize=7)
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    cbar.set_label('Probability')
    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def visualize(checkpoint_dir: str, out_subdir: str = 'aggregator_viz') -> None:
    os.makedirs(checkpoint_dir, exist_ok=True)
    out_dir = os.path.join(checkpoint_dir, out_subdir)
    os.makedirs(out_dir, exist_ok=True)

    aggregator_meta = load_aggregator_meta(checkpoint_dir)
    aggregator_cfg, projector_cfg = load_configs(checkpoint_dir)

    # Global summaries
    if aggregator_meta:
        plot_overall_heatmap(aggregator_meta, os.path.join(out_dir, 'weights_heatmap.png'))
        plot_mean_prob_bar(aggregator_meta, os.path.join(out_dir, 'weights_mean.png'))

        # Save JSON summary
        ids = sorted(aggregator_meta.keys())
        summary = {
            'num_aggregators': len(ids),
            'num_options': int(aggregator_meta[ids[0]]['probs'].shape[0]) if ids else 0,
            'aggregators': [
                {
                    'id': int(i),
                    'file': aggregator_meta[i]['file'],
                    'temperature': float(aggregator_meta[i]['temperature']),
                    'logits': [float(x) for x in aggregator_meta[i]['logits'].tolist()],
                    'probs': [float(x) for x in aggregator_meta[i]['probs'].tolist()],
                    'argmax_option': int(int(np.argmax(aggregator_meta[i]['probs'])))
                }
                for i in ids
            ]
        }
        with open(os.path.join(out_dir, 'weights_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2)

    # Combined bipartite visualizations per (target_model_idx, source_model_idx)
    # Option A: grid of subplots per-layer (kept for reference)
    for target_model_idx, src_map in aggregator_cfg.items():
        if target_model_idx not in projector_cfg:
            continue
        for source_model_idx, layer_to_agg in src_map.items():
            if source_model_idx not in projector_cfg[target_model_idx]:
                continue
            # Option B: single figure with ALL nodes and edges
            all_out = os.path.join(out_dir, f'bipartite_all_T{target_model_idx}_S{source_model_idx}.png')
            plot_bipartite_all(
                source_model_idx=source_model_idx,
                target_model_idx=target_model_idx,
                layer_to_agg=layer_to_agg,
                projector_cfg_for_pair=projector_cfg[target_model_idx][source_model_idx],
                aggregator_meta=aggregator_meta,
                out_path=all_out,
            )
            # M x N heatmap (rows: source layers, cols: target layers)
            heat_out = os.path.join(out_dir, f'mapping_heatmap_T{target_model_idx}_S{source_model_idx}.png')
            plot_mapping_heatmap_all(
                source_model_idx=source_model_idx,
                target_model_idx=target_model_idx,
                layer_to_agg=layer_to_agg,
                projector_cfg_for_pair=projector_cfg[target_model_idx][source_model_idx],
                aggregator_meta=aggregator_meta,
                out_path=heat_out,
            )


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Visualize WeightedAggregator distributions and bipartite layer mappings')
    parser.add_argument('--checkpoint-dir', type=str, required=False,
                        default='local/checkpoints/20250819_205607/final',
                        help='Directory containing aggregator_*.pt/json and *_config.json files')
    parser.add_argument('--out-subdir', type=str, default='aggregator_viz', help='Subdirectory under checkpoint-dir to write outputs')
    args = parser.parse_args()

    visualize(args.checkpoint_dir, args.out_subdir)


