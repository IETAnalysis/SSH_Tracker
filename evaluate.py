import csv
import os
from collections import defaultdict
import numpy as np
import torch
from scipy.signal import find_peaks
import config

def _edit_distance(a: list, b: list) -> int:
    m, n = len(a), len(b)
    dp = np.arange(n + 1, dtype=np.int32)
    for i in range(1, m + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, n + 1):
            tmp = dp[j]
            dp[j] = prev if a[i - 1] == b[j - 1] else 1 + min(prev, dp[j], dp[j - 1])
            prev = tmp
    return int(dp[n])

def _ned(pred: list, true: list) -> float:
    if not pred and not true: return 1.0
    return 1.0 - _edit_distance(pred, true) / max(len(pred), len(true))

def _lcs_recall(pred: list, true: list) -> float:
    if not true: return 1.0
    m, n = len(pred), len(true)
    dp = np.zeros((m + 1, n + 1), dtype=np.int32)
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if pred[i - 1] == true[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return int(dp[m][n]) / len(true)

def _position_accuracy(pred: list, true: list) -> float:
    if not true: return 1.0
    hits = sum(1 for i in range(min(len(pred), len(true))) if pred[i] == true[i])
    return hits / len(true)

def _set_prf(pred: list, true: list) -> tuple:
    sp, st = set(pred), set(true)
    inter = len(sp & st)
    p = inter / len(sp) if sp else float('nan')
    r = inter / len(st) if st else float('nan')
    f1 = 2 * p * r / (p + r) if not (np.isnan(p) or np.isnan(r)) and p + r > 0 else float('nan')
    return p, r, f1

def _exact_match(pred: list, true: list) -> int:
    return int(set(pred) == set(true))

def _jaccard(pred: list, true: list) -> float:
    sp, st = set(pred), set(true)
    if not sp and not st:
        return 1.0
    union = len(sp | st)
    if union == 0:
        return 1.0
    return len(sp & st) / union

def _split_prf(pred_splits, true_splits, seg_lengths):
    intervals = []
    tolerance_ratio = getattr(config, 'SPLIT_TOLERANCE_RATIO')

    for k, ts in enumerate(true_splits):
        bl = seg_lengths[k] if k < len(seg_lengths) else 1
        al = seg_lengths[k + 1] if k + 1 < len(seg_lengths) else 1
        intervals.append((ts - max(1, int(tolerance_ratio * bl)),
                          ts + max(1, int(tolerance_ratio * al))))

    matched, tp = set(), 0
    signed_offsets = []

    for pp in pred_splits:
        best_k, best_d = -1, float('inf')
        for k, (lo, hi) in enumerate(intervals):
            if k not in matched and lo <= pp <= hi:
                d = abs(pp - true_splits[k])
                if d < best_d:
                    best_d, best_k = d, k
        if best_k >= 0:
            matched.add(best_k)
            tp += 1
            signed_offsets.append(pp - true_splits[best_k])

    fp, fn = len(pred_splits) - tp, len(true_splits) - tp
    p = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
    r = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
    f1 = 2 * p * r / (p + r) if not (np.isnan(p) or np.isnan(r)) and p + r > 0 else float('nan')
    all_hit = int(len(true_splits) > 0 and tp == len(true_splits))

    return {
        'sp_tp': tp, 'sp_fp': fp, 'sp_fn': fn,
        'sp_precision': p, 'sp_recall': r, 'sp_f1': f1,
        'sp_all_hit': all_hit, 'sp_n_true': len(true_splits),
        'sp_signed_offsets': signed_offsets,
    }

def _aggregate(flow_results):
    n = len(flow_results)
    if n == 0:
        return _empty_metrics()

    def _mean(lst):
        return float(np.mean(lst)) if lst else float('nan')

    ned_vals = [r['ned'] for r in flow_results]
    lcs_vals = [r['lcs'] for r in flow_results]
    pos_vals = [r['pos_acc'] for r in flow_results]
    emr_vals = [r['emr'] for r in flow_results]
    jaccard_vals = [r['jaccard'] for r in flow_results]

    set_p_vals = [r['set_p'] for r in flow_results if not np.isnan(r['set_p'])]
    set_r_vals = [r['set_r'] for r in flow_results if not np.isnan(r['set_r'])]
    set_f1_vals = [r['set_f1'] for r in flow_results if not np.isnan(r['set_f1'])]

    sp_tp = sum(r['sp_tp'] for r in flow_results)
    sp_fp = sum(r['sp_fp'] for r in flow_results)
    sp_fn = sum(r['sp_fn'] for r in flow_results)
    p = sp_tp / (sp_tp + sp_fp) if (sp_tp + sp_fp) > 0 else float('nan')
    r = sp_tp / (sp_tp + sp_fn) if (sp_tp + sp_fn) > 0 else float('nan')
    f1 = 2 * p * r / (p + r) if not (np.isnan(p) or np.isnan(r)) and p + r > 0 else float('nan')

    all_offsets = []
    for res in flow_results:
        all_offsets.extend(res['sp_signed_offsets'])

    if all_offsets:
        offsets_arr = np.array(all_offsets, dtype=np.float64)
        offset_mean = float(np.mean(offsets_arr))
        offset_mae = float(np.mean(np.abs(offsets_arr)))
        offset_std = float(np.std(offsets_arr))
    else:
        offset_mean = offset_mae = offset_std = float('nan')

    ce_list = [r['ce'] for r in flow_results]

    ce_dist = {
        label: sum(1 for ce in ce_list if lo <= ce <= hi)
        for label, lo, hi in [
            ('0', 0, 0), ('+1', 1, 1), ('-1', -1, -1),
            ('+2', 2, 2), ('-2', -2, -2), ('+3', 3, 3),
            ('-3', -3, -3), ('+4', 4, 4), ('-4', -4, -4),
            ('>+4', 5, 9999), ('<-4', -9999, -5)
        ]
    }

    return {
        'ned': _mean(ned_vals),
        'lcs': _mean(lcs_vals),
        'pos_acc': _mean(pos_vals),
        'emr': _mean(emr_vals),
        'jaccard': _mean(jaccard_vals),
        'set_p': _mean(set_p_vals),
        'set_r': _mean(set_r_vals),
        'set_f1': _mean(set_f1_vals),
        'sp_precision': p,
        'sp_recall': r,
        'sp_f1': f1,
        'sp_all_hit': _mean([r['sp_all_hit'] for r in flow_results if r['sp_n_true'] > 0]),
        'ce_mean': float(np.mean(ce_list)),
        'ce_std': float(np.std(ce_list)),
        'ce_dist': ce_dist,
        'offset_mean': offset_mean,
        'offset_mae': offset_mae,
        'offset_std': offset_std,
        '_n': n,
        '_sp_tp': sp_tp,
        '_sp_fp': sp_fp,
        '_sp_fn': sp_fn,
        '_n_with_splits': sum(1 for r in flow_results if r['sp_n_true'] > 0),
        '_n_offset_pts': len(all_offsets),
    }

def _empty_metrics():
    return {
        'ned': float('nan'), 'lcs': float('nan'), 'pos_acc': float('nan'),
        'emr': float('nan'), 'jaccard': float('nan'),
        'set_p': float('nan'), 'set_r': float('nan'), 'set_f1': float('nan'),
        'sp_precision': float('nan'), 'sp_recall': float('nan'), 'sp_f1': float('nan'),
        'sp_all_hit': float('nan'),
        'ce_mean': float('nan'), 'ce_std': float('nan'), 'ce_dist': {},
        'offset_mean': float('nan'), 'offset_mae': float('nan'), 'offset_std': float('nan'),
        '_n': 0, '_sp_tp': 0, '_sp_fp': 0, '_sp_fn': 0,
        '_n_with_splits': 0, '_n_offset_pts': 0,
    }

def print_metrics(m, tag='SSH-Tracker'):
    print(f"\n[{tag}] Total Flows: {m['_n']} | TP={m['_sp_tp']} FP={m['_sp_fp']} FN={m['_sp_fn']}")
    print(f"  NED:     {m['ned'] * 100:.2f}%  |  LCS: {m['lcs'] * 100:.2f}%  |  PosAcc: {m['pos_acc'] * 100:.2f}%")
    print(f"  EMR:     {m['emr'] * 100:.2f}%  |  Jaccard: {m['jaccard'] * 100:.2f}%")
    print(f"  Set-P:   {m['set_p'] * 100:.2f}%  |  Set-R: {m['set_r'] * 100:.2f}%  |  Set-F1: {m['set_f1'] * 100:.2f}%")
    print(f"  SP-P:    {m['sp_precision'] * 100:.2f}%  |  SP-R:  {m['sp_recall'] * 100:.2f}%  |  SP-F1: {m['sp_f1'] * 100:.2f}%")
    print(f"  Offset({m['_n_offset_pts']}pts): Mean={m['offset_mean']:+.1f}  MAE={m['offset_mae']:.1f}  Std={m['offset_std']:.1f}")

def print_metrics_grouped(overall, per_tab):
    def _fmt(v):
        return f"{v * 100:6.2f}%" if isinstance(v, float) and not np.isnan(v) else "   N/A"

    def _fmtf(v, dec=2):
        return f"{v:.{dec}f}" if isinstance(v, float) and not np.isnan(v) else "N/A"

    print(f"\n{'=' * 80}")
    print("SSH-Tracker Evaluation Summary".center(80))
    print(f"{'=' * 80}")

    print(f"\n--- Overall (Total {overall['_n']} flows) ---")
    _print_one_group(overall, _fmt, _fmtf)

    for tc in sorted(per_tab.keys()):
        m = per_tab[tc]
        print(f"\n--- {tc}-tab (Total {m['_n']} flows) ---")
        if m['_n'] == 0:
            print("  (No samples in this group)")
            continue
        _print_one_group(m, _fmt, _fmtf)

    _print_comparison_table(overall, per_tab)

    print(f"\n--- Count Error Distribution ---")
    ce_labels = ['0', '+1', '-1', '+2', '-2', '+3', '-3', '+4', '-4', '>+4', '<-4']

    header = f"  {'Group':<8} |"
    for l in ce_labels:
        header += f" {l:>5} |"
    print(header)
    print(f"  {'-' * 100}")

    def _print_ce_row(label, m):
        ce_d = m.get('ce_dist', {})
        row_str = f"  {label:<8} |"
        for l in ce_labels:
            cnt = ce_d.get(l, 0)
            row_str += f" {cnt:>5} |"
        print(row_str)

    _print_ce_row('Overall', overall)
    for tc in sorted(per_tab.keys()):
        _print_ce_row(f'{tc}-tab', per_tab[tc])
    print(f"{'=' * 104}\n")

def _print_one_group(m, _fmt, _fmtf):
    print(f"  NED:     {_fmt(m.get('ned'))}  |  LCS:     {_fmt(m.get('lcs'))}  |  PosAcc:  {_fmt(m.get('pos_acc'))}")
    print(f"  EMR:     {_fmt(m.get('emr'))}  |  Jaccard: {_fmt(m.get('jaccard'))}")
    print(f"  Set-P:   {_fmt(m.get('set_p'))}  |  Set-R:   {_fmt(m.get('set_r'))}  |  Set-F1:  {_fmt(m.get('set_f1'))}")
    print(f"  SP-P:    {_fmt(m.get('sp_precision'))}  |  SP-R:    {_fmt(m.get('sp_recall'))}  |  SP-F1:   {_fmt(m.get('sp_f1'))}")
    print(f"  AllHit:  {_fmt(m.get('sp_all_hit'))}  (TP={m.get('_sp_tp', 0)} FP={m.get('_sp_fp', 0)} FN={m.get('_sp_fn', 0)})")
    print(f"  Offset({m.get('_n_offset_pts', 0)}pts): Mean={_fmtf(m.get('offset_mean', float('nan')), 2)}  MAE={_fmtf(m.get('offset_mae', float('nan')), 2)}  Std={_fmtf(m.get('offset_std', float('nan')), 2)}")
    print(f"  CE: {_fmtf(m.get('ce_mean', float('nan')))} ± {_fmtf(m.get('ce_std', float('nan')))}")

def _print_comparison_table(overall, per_tab):
    print(f"\n" + "=" * 122)
    print(f" {'':<12} | {'Behavior Classification':^68} | {'Boundary Detection':^35}")
    print(f" {'':<12} | {'Ordered Sequence':^23} | {'Unordered Set':^42} | {'(Split Metrics)':^35}")
    print(f" {'Group':<6} {'N':>5} | {'NED':>7} {'LCS':>7} {'PosAcc':>7} | {'EMR':>7} {'Set-P':>7} {'Set-R':>7} {'Set-F1':>7} {'Jaccard':>7} | {'SP-P':>7} {'SP-R':>7} {'SP-F1':>7} {'MAE ± Std':>13}")
    print("-" * 122)

    def _row(label, m):
        n = m.get('_n', 0)

        def _f(v):
            if np.isnan(v): return "    N/A"
            return f"{v * 100:>6.2f}%"

        mae = m.get('offset_mae', float('nan'))
        std = m.get('offset_std', float('nan'))
        if np.isnan(mae) or np.isnan(std):
            mae_std_str = "      N/A"
        else:
            mae_std_str = f"{mae:.2f} ± {std:.2f}"

        print(f" {label:<6} {n:>5} | "
              f"{_f(m.get('ned'))} {_f(m.get('lcs'))} {_f(m.get('pos_acc'))} | "
              f"{_f(m.get('emr'))} {_f(m.get('set_p'))} {_f(m.get('set_r'))} {_f(m.get('set_f1'))} {_f(m.get('jaccard'))} | "
              f"{_f(m.get('sp_precision'))} {_f(m.get('sp_recall'))} {_f(m.get('sp_f1'))} {mae_std_str:>13}")

    _row('Overall', overall)
    for tc in sorted(per_tab.keys()):
        _row(f'{tc}-tab', per_tab[tc])
    print("=" * 122)

@torch.no_grad()
def evaluate_cascaded(model, class_seg_stats, test_loader, device, csv_path=None):
    model.eval()
    flow_results = []
    writer = None

    if csv_path:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        f_csv = open(csv_path, 'w', newline='', encoding='utf-8')
        writer = csv.writer(f_csv)
        writer.writerow(['Tab_Count', 'True_Labels', 'Predicted_Labels'])

    for feats_list, _, metas_list in test_loader:
        for f, meta in zip(feats_list, metas_list):
            h_fused = model.encoder(f.unsqueeze(0).to(device))
            L = h_fused.size(1)

            logits_p1 = model.phase1(h_fused).squeeze(0)
            probs = torch.sigmoid(logits_p1).cpu().numpy()

            peaks, _ = find_peaks(
                probs,
                height=getattr(config, 'PEAK_HEIGHT'),
                prominence=getattr(config, 'PEAK_PROMINENCE'),
                distance=getattr(config, 'PEAK_DISTANCE')
            )
            pred_splits = list(peaks)

            boundaries = [0] + pred_splits + [L]
            pred_beh = []

            for i in range(len(boundaries) - 1):
                start, end = boundaries[i], boundaries[i + 1]
                if end - start < getattr(config, 'MIN_SEGMENT_LENGTH'):
                    pred_beh.append(pred_beh[-1] if pred_beh else 0)
                    continue
                chunk_tensor = h_fused[0, start:end, :]
                logits_p2 = model.phase2(chunk_tensor)
                pred_beh.append(logits_p2.argmax(dim=-1).item())

            true_beh = meta['flow_label']
            tab_count = meta.get('tab_count', len(true_beh))

            if writer:
                writer.writerow([tab_count, str(true_beh), str(pred_beh)])

            set_p, set_r, set_f1 = _set_prf(pred_beh, true_beh)
            sp_prf = _split_prf(pred_splits, meta['split_points'], meta['seg_lengths'])

            flow_results.append({
                'tab_count': tab_count,
                'ned': _ned(pred_beh, true_beh),
                'lcs': _lcs_recall(pred_beh, true_beh),
                'pos_acc': _position_accuracy(pred_beh, true_beh),
                'emr': _exact_match(pred_beh, true_beh),
                'jaccard': _jaccard(pred_beh, true_beh),
                'set_p': set_p,
                'set_r': set_r,
                'set_f1': set_f1,
                'ce': len(pred_splits) - len(meta['split_points']),
                **sp_prf,
            })

    if csv_path:
        f_csv.close()

    overall = _aggregate(flow_results)

    grouped = defaultdict(list)
    for r in flow_results:
        grouped[r['tab_count']].append(r)

    per_tab = {}
    for tc in sorted(grouped.keys()):
        per_tab[tc] = _aggregate(grouped[tc])

    return overall, per_tab