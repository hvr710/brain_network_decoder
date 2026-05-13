import argparse
import html
import json
from pathlib import Path
from typing import Dict, List, Optional

from abide_lcm_utils import ensure_dir
from abide_lcm_utils import metric_value, summarize_runs


METRICS = [
    ("accuracy", "ACC"),
    ("weighted_f1", "F1"),
]


def parse_args():
    parser = argparse.ArgumentParser("Render ABIDE dx AAL116 result tables")
    parser.add_argument("--probe_dir", type=str, default="outputs/abide_dx_aal116_lcm_probe")
    parser.add_argument("--finetune_dir", type=str, default="outputs/abide_dx_aal116_lcm_finetune")
    parser.add_argument("--feature_source", type=str, default="lcm_frozen")
    parser.add_argument("--out_dir", type=str, default="outputs/abide_dx_aal116_tables")
    return parser.parse_args()


def load_json(path: Path) -> Optional[Dict]:
    if not path.exists():
        print(f"Skip missing summary: {path}")
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_summaries(args) -> List[Dict]:
    probe_dir = Path(args.probe_dir) / args.feature_source
    candidates = [
        ("LCM frozen + LP", probe_dir / "lp" / "summary.json"),
        ("LCM frozen + MLP", probe_dir / "mlp" / "summary.json"),
        ("LCM full finetune", Path(args.finetune_dir) / "summary.json"),
    ]
    rows = []
    for method, path in candidates:
        summary = load_json(path)
        if summary is None:
            continue
        summary["method"] = summary.get("method", method)
        summary = summarize_runs(summary.get("runs", []))
        summary["method"] = method
        rows.append(summary)
    if not rows:
        raise FileNotFoundError("No result summaries found. Run probe and/or finetune first.")
    return rows


def pct(mean, std) -> str:
    if mean is None:
        return "-"
    if std is None:
        std = 0.0
    return f"{100.0 * float(mean):.2f}&plusmn;{100.0 * float(std):.2f}"


def best_second(rows: List[Dict], metric: str):
    values = []
    for idx, row in enumerate(rows):
        mean = row.get("test_mean", {}).get(metric)
        if mean is not None:
            values.append((float(mean), idx))
    values.sort(reverse=True)
    best = values[0][1] if values else None
    second = values[1][1] if len(values) > 1 else None
    return best, second


def page_head(title: str) -> str:
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>{html.escape(title)}</title>
  <style>
    body {{
      margin: 24px 40px;
      color: #111;
      background: #fff;
      font-family: "Times New Roman", Times, serif;
    }}
    .paper-title {{
      text-align: center;
      font-weight: 700;
      font-size: 30px;
      border-bottom: 4px solid #111;
      padding-bottom: 8px;
      margin-bottom: 24px;
    }}
    .caption {{
      font-size: 24px;
      line-height: 1.3;
      margin: 0 0 24px;
    }}
    .caption em {{
      font-style: italic;
      font-weight: 700;
    }}
    .best {{
      color: #c85f70;
      font-weight: 700;
    }}
    .second {{
      text-decoration: underline;
      text-underline-offset: 3px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 23px;
      table-layout: fixed;
    }}
    th, td {{
      padding: 8px 10px;
      text-align: center;
      vertical-align: middle;
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{
      text-align: left;
      width: 24%;
    }}
    thead tr:first-child th {{
      border-top: 3px solid #111;
    }}
    thead tr:last-child th {{
      border-bottom: 2px solid #555;
    }}
    tbody tr:last-child td {{
      border-bottom: 4px double #111;
    }}
    tbody tr.group-start td {{
      border-top: 2px solid #777;
    }}
    .subhead {{
      font-style: italic;
      font-weight: 400;
    }}
  </style>
</head>
<body>
"""


def page_tail() -> str:
    return "</body>\n</html>\n"


def render_summary(rows: List[Dict], out_path: Path) -> None:
    ranks = {metric: best_second(rows, metric) for metric, _ in METRICS}
    parts = [
        page_head("ABIDE dx AAL116 LCM Summary"),
        '<div class="paper-title">Omni-fMRI: ABIDE AAL116 LCM Diagnosis</div>',
        (
            '<p class="caption"><em>Table.</em> Performance on ABIDE diagnosis with '
            "Accuracy/F1. <span class=\"best\">Red</span> indicates the best performance "
            "and <span class=\"second\">Underline</span> indicates the second performance.</p>"
        ),
        "<table>",
        "<thead>",
        "<tr><th>Model</th><th colspan=\"4\">ABIDE<br><span class=\"subhead\">Diagnosis</span></th></tr>",
        "<tr><th></th>",
    ]
    for _, label in METRICS:
        parts.append(f"<th>{html.escape(label)}&uarr;</th>")
    parts.extend(["</tr>", "</thead>", "<tbody>"])
    for row_idx, row in enumerate(rows):
        parts.append("<tr>")
        parts.append(f"<td><strong>{html.escape(row['method'])}</strong></td>")
        for metric, _ in METRICS:
            mean = row.get("test_mean", {}).get(metric)
            std = row.get("test_std", {}).get(metric)
            css = ""
            if row_idx == ranks[metric][0]:
                css = ' class="best"'
            elif row_idx == ranks[metric][1]:
                css = ' class="second"'
            parts.append(f"<td{css}>{pct(mean, std)}</td>")
        parts.append("</tr>")
    parts.extend(["</tbody>", "</table>", page_tail()])
    out_path.write_text("\n".join(parts), encoding="utf-8")


def render_per_run(rows: List[Dict], out_path: Path) -> None:
    parts = [
        page_head("ABIDE dx AAL116 LCM Per Run"),
        '<div class="paper-title">Omni-fMRI: ABIDE AAL116 LCM Diagnosis</div>',
        (
            '<p class="caption"><em>Table.</em> Per-run validation and test metrics. '
            "Probe rows use five LP/MLP seeds; finetune rows use three full-finetune seeds.</p>"
        ),
        "<table>",
        "<thead>",
        (
            "<tr><th>Model</th><th>Seed</th><th>Best Epoch</th>"
            "<th colspan=\"2\">Validation</th><th colspan=\"2\">Test</th></tr>"
        ),
        (
            "<tr><th></th><th></th><th></th><th>ACC&uarr;</th><th>F1&uarr;</th>"
            "<th>ACC&uarr;</th><th>F1&uarr;</th></tr>"
        ),
        "</thead>",
        "<tbody>",
    ]
    first = True
    for summary in rows:
        method = summary["method"]
        for run in summary.get("runs", []):
            cls = ' class="group-start"' if not first else ""
            first = False
            best_epoch = run.get("best_epoch")
            if best_epoch is None:
                best_epoch = "-"
            parts.append(f"<tr{cls}>")
            parts.append(f"<td><strong>{html.escape(method)}</strong></td>")
            parts.append(f"<td>{html.escape(str(run.get('seed', '-')))}</td>")
            parts.append(f"<td>{html.escape(str(best_epoch))}</td>")
            parts.append(f"<td>{pct(run['val'].get('accuracy'), 0)}</td>")
            parts.append(f"<td>{pct(metric_value(run['val'], 'weighted_f1'), 0)}</td>")
            parts.append(f"<td>{pct(run['test'].get('accuracy'), 0)}</td>")
            parts.append(f"<td>{pct(metric_value(run['test'], 'weighted_f1'), 0)}</td>")
            parts.append("</tr>")
    parts.extend(["</tbody>", "</table>", page_tail()])
    out_path.write_text("\n".join(parts), encoding="utf-8")


def main():
    args = parse_args()
    out_dir = ensure_dir(args.out_dir)
    rows = collect_summaries(args)
    render_summary(rows, out_dir / "summary_table.html")
    render_per_run(rows, out_dir / "per_run_table.html")
    print(f"Wrote {out_dir / 'summary_table.html'}")
    print(f"Wrote {out_dir / 'per_run_table.html'}")


if __name__ == "__main__":
    main()
