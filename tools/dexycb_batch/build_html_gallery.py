#!/usr/bin/env python3
"""Build every preprocessed DexYCB HTML plus a compact gallery index."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from visualize_preprocessed_interactive import make_html


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preprocessed-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "preprocessed" / "dexycb",
    )
    parser.add_argument(
        "--visualization-root",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "visualizations" / "dexycb",
    )
    args = parser.parse_args()
    source_root = args.preprocessed_root.expanduser().resolve()
    destination_root = args.visualization_root.expanduser().resolve()
    inputs = sorted(source_root.glob("*/dexycb_right_hand_preprocessed.npz"))
    if not inputs:
        raise FileNotFoundError(f"no preprocessed trajectories under {source_root}")
    summaries = []
    for input_path in inputs:
        output_path = destination_root / input_path.parent.name / "preprocessed_interactive.html"
        summary = make_html(input_path, output_path)
        summaries.append(summary)
        print(f"[{summary['sequence']}] {summary['frames']} frames -> {output_path}")

    rows = []
    for item in summaries:
        relative = Path(item["html"]).relative_to(destination_root).as_posix()
        rows.append(
            "<tr>"
            f"<td><a href='{html.escape(relative)}'>{html.escape(item['sequence'])}</a></td>"
            f"<td>{item['frames']}</td>"
            f"<td>{html.escape(item['source_side'])} → right</td>"
            f"<td>{html.escape(item['object'])}</td>"
            "</tr>"
        )
    page = """<!doctype html>
<html lang="ko"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>DexYCB preprocessed sequences</title>
<style>
body{font-family:system-ui,sans-serif;max-width:1000px;margin:40px auto;padding:0 18px;color:#202124}
table{border-collapse:collapse;width:100%;box-shadow:0 2px 12px #0001}
th,td{padding:13px 15px;border-bottom:1px solid #ddd;text-align:left}th{background:#f4f7fb}
a{color:#1769aa;font-weight:650;text-decoration:none}a:hover{text-decoration:underline}
.note{color:#5f6368;margin-bottom:24px}.ok{color:#188038;font-weight:650}
</style></head><body>
<h1>DexYCB 전체 전처리 결과</h1>
<p class="note"><span class="ok">완료</span> — 각 링크는 Plotly가 내장된 독립 실행형 HTML입니다.
마우스로 회전·확대하고 슬라이더/재생 버튼으로 전체 프레임을 확인할 수 있습니다.</p>
<table><thead><tr><th>Sequence</th><th>Valid frames</th><th>Hand conversion</th><th>Grasped object</th></tr></thead>
<tbody>""" + "\n".join(rows) + """</tbody></table></body></html>\n"""
    destination_root.mkdir(parents=True, exist_ok=True)
    index_path = destination_root / "index.html"
    index_path.write_text(page, encoding="utf-8")
    (destination_root / "manifest.json").write_text(
        json.dumps(summaries, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved gallery: {index_path}")


if __name__ == "__main__":
    main()
