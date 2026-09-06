"""Full-roster vs half-A vs half-B setup models: curves, ceilings, and seen/unseen opponent split."""
import json, sys, statistics as st, math
from pathlib import Path
import numpy as np
FS = Path(sys.argv[1]); H = Path(sys.argv[2]); out_html = Path(sys.argv[3]); out_json = Path(sys.argv[4])
HALVES = {"A": ("basic_heuristic", "tactical_rule_based", "stress_miner_rush", "stress_information_miser"),
          "B": ("strategic_rule_based", "stress_scout_rush", "stress_berserker", "stress_chaos")}
runs = {"full": json.load(open(FS / "eval_analysis.json"))}
for h in ("A", "B"):
    p = H / h / "eval_analysis.json"
    if p.exists(): runs[f"half_{h}"] = json.load(open(p))
S = list(runs["full"]["ewr"]); OPP = list(runs["full"]["pooled"]["ckpt_1024-ckpt_0"]["by_opponent"])
pool = lambda A, arm: st.mean(A["ewr"][s][arm] for s in S)
lib = pool(runs["full"], "library")
cks = lambda A: sorted(int(k.split("_")[1]) for k in A["ewr"][S[0]] if k.startswith("ckpt_"))
summary = {"library": lib, "runs": {}}
# per-opponent absolute EWR at the final checkpoint needs per-opponent scores; reconstruct from rows
def per_opponent_ewr(run_dir, arm):
    import collections
    from types import SimpleNamespace
    sys.path.insert(0, "/Users/brandonwashington/Dev/Github/stratego/gpt_agent/output/phase18/worktrees/g3-stage6b")
    from stratego.training.phase18 import g3_evaluation as ev
    cases = ev.build_cases(ev.load_evaluation_bases()); opp_of = {c.case_index: c.opponent_id for c in cases}
    tot = collections.defaultdict(list)
    for s in S:
        for line in (run_dir / "eval" / f"rows_{s}_{arm}.jsonl").read_text().splitlines():
            if line.strip():
                o = json.loads(line); tot[opp_of[o["setup_pair_id"]]].append(o["candidate_score"])
    return {o: float(np.mean(v)) for o, v in tot.items()}
dirs = {"full": FS, "half_A": H / "A", "half_B": H / "B"}
for name, A in runs.items():
    c = cks(A); last = c[-1]
    curve = [(k, pool(A, f"ckpt_{k}")) for k in c]
    v = A["pooled"].get(f"ckpt_{last}-library", {})
    po = per_opponent_ewr(dirs[name], f"ckpt_{last}"); po_lib = per_opponent_ewr(dirs[name], "library")
    seen = set(HALVES.get(name[-1], ())) if name.startswith("half") else set(OPP)
    summary["runs"][name] = {"checkpoints": curve, "final": pool(A, f"ckpt_{last}"), "init": pool(A, "ckpt_0"),
        "vs_library": {k: v.get(k) for k in ("point", "lower", "upper")}, "seen_opponents": sorted(seen),
        "final_ewr_by_opponent": po, "library_ewr_by_opponent": po_lib,
        "final_minus_library_seen": float(np.mean([po[o] - po_lib[o] for o in OPP if o in seen])),
        "final_minus_library_unseen": float(np.mean([po[o] - po_lib[o] for o in OPP if o not in seen])) if len(seen) < len(OPP) else None}
out_json.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
# ---- chart ----
W_, H1, L, R_, T = 880, 360, 70, 30, 44
cols = {"full": "#2a78d6", "half_A": "#eb6834", "half_B": "#1baf7a"}; names = {"full": "full roster (8 bots)", "half_A": "half A (4 bots)", "half_B": "half B (4 bots)"}
ys = [y for r in summary["runs"].values() for _, y in r["checkpoints"]] + [lib]; lo_y, hi_y = min(ys) - 0.02, max(ys) + 0.03
sx = lambda p: L + (W_ - L - R_) * p / 1024; sy = lambda v: T + (H1 - T - 40) * (1 - (v - lo_y) / (hi_y - lo_y))
g = [f'<text x="{L}" y="22" class="t1">Absolute EWR by checkpoint — trained on all 8 opponents vs 4 of 8 (same init, same games per period, same draws)</text>']
for v in np.arange(math.ceil(lo_y * 20) / 20, hi_y, 0.05): g.append(f'<line x1="{L}" x2="{W_-R_}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="#e1e0d9"/><text x="{L-8}" y="{sy(v)+4:.1f}" class="ax" text-anchor="end">{v:.2f}</text>')
for p in range(0, 1025, 128): g.append(f'<text x="{sx(p):.1f}" y="{H1-16}" class="ax" text-anchor="middle">{p}</text>')
g.append(f'<line x1="{L}" x2="{W_-R_}" y1="{sy(lib):.1f}" y2="{sy(lib):.1f}" stroke="#898781" stroke-dasharray="6 4"/><text x="{W_-R_}" y="{sy(lib)-6:.1f}" class="lab" text-anchor="end">handcrafted library {lib:.3f}</text>')
lx = L
for name, r in summary["runs"].items():
    col = cols[name]; pts = r["checkpoints"]
    g.append('<polyline points="' + " ".join(f"{sx(k):.1f},{sy(v):.1f}" for k, v in pts) + f'" fill="none" stroke="{col}" stroke-width="2"/>')
    for k, v in pts: g.append(f'<circle cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="4" fill="{col}" stroke="#fcfcfb" stroke-width="2"/><circle class="hit" cx="{sx(k):.1f}" cy="{sy(v):.1f}" r="13" fill="transparent" data-tip="{names[name]} period {k}: {v:.4f}"/>')
    k, v = pts[-1]; g.append(f'<text x="{sx(k)+8:.1f}" y="{sy(v)+4:.1f}" class="lab">{v:.3f}</text>')
    g.append(f'<rect x="{lx}" y="{T-4}" width="10" height="10" fill="{col}" rx="2"/><text x="{lx+16}" y="{T+5}" class="lab">{names[name]}</text>'); lx += 200
rows = ""
for name, r in summary["runs"].items():
    u = r["final_minus_library_unseen"]
    rows += f"<tr><td>{names[name]}</td><td>{r['init']:.4f}</td><td>{r['final']:.4f}</td><td>{r['final']-lib:+.4f}</td><td>{r['final_minus_library_seen']:+.4f}</td><td>{(f'{u:+.4f}' if u is not None else '—')}</td></tr>"
html = f"""<title>Roster Diversity Test</title><style>:root{{--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9}}@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a}}}}:root[data-theme="dark"]{{--surf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a}}
body{{background:var(--surf);color:var(--ink);font:14px/1.45 -apple-system,Inter,Segoe UI,sans-serif;margin:0;padding:20px}}.wrap{{max-width:960px;margin:0 auto}}svg{{width:100%;height:auto;display:block}}.t1{{font-size:14px;font-weight:600;fill:var(--ink)}}.ax{{font-size:11px;fill:#898781}}.lab{{font-size:12px;fill:var(--ink2);font-variant-numeric:tabular-nums}}
#tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--surf);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0}}table{{border-collapse:collapse;margin-top:16px;font-size:13px;font-variant-numeric:tabular-nums}}th,td{{padding:6px 12px;text-align:right;border-bottom:1px solid var(--grid)}}th{{color:var(--ink2)}}td:first-child,th:first-child{{text-align:left}}</style>
<div class="wrap"><svg viewBox="0 0 {W_} {H1}" xmlns="http://www.w3.org/2000/svg">{''.join(g)}</svg>
<table><tr><th>trained on</th><th>init</th><th>final (1,024)</th><th>final − library, all 8 opponents</th><th>… seen opponents</th><th>… unseen opponents</th></tr>{rows}</table>
<p class="lab" style="max-width:72ch">"Seen" = the opponents the model watched during training; "unseen" = the other four. All values are EWR pooled over 8 handcrafted movers on the frozen 2,560-game schedule, minus the same mover's EWR with a handcrafted library formation on the same cases.</p></div><div id="tip"></div>
<script>const tip=document.getElementById('tip');document.querySelectorAll('.hit').forEach(h=>{{h.addEventListener('mousemove',e=>{{tip.textContent=h.dataset.tip;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px';tip.style.opacity=1}});h.addEventListener('mouseleave',()=>tip.style.opacity=0)}});</script>"""
out_html.write_text(html); print("comparison built for:", list(summary["runs"]), "| library", round(lib, 4))
for n, r in summary["runs"].items(): print("  %-7s final %.4f  vs lib all %+.4f  seen %+.4f  unseen %s" % (n, r["final"], r["final"] - lib, r["final_minus_library_seen"], ("%+.4f" % r["final_minus_library_unseen"]) if r["final_minus_library_unseen"] is not None else "—"))
