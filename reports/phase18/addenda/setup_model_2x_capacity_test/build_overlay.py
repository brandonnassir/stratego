import json, sys, math, statistics as st
import numpy as np
A1, A2 = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
R1 = [json.loads(l) for l in open(sys.argv[3])]; R2 = [json.loads(l) for l in open(sys.argv[4])]; out = sys.argv[5]
S = list(A1["ewr"]); lib = st.mean(A1["ewr"][s]["library"] for s in S)
def series(A):
    cks = sorted(int(k.split("_")[1]) for k in A["ewr"][S[0]] if k.startswith("ckpt_"))
    pts = []
    for c in cks:
        abs_ = st.mean(A["ewr"][s][f"ckpt_{c}"] for s in S)
        if c == 0: pts.append((c, abs_, abs_, abs_)); continue
        v = A["pooled"][f"ckpt_{c}-ckpt_0"]; i0 = st.mean(A["ewr"][s]["ckpt_0"] for s in S)
        pts.append((c, abs_, i0 + v["lower"], i0 + v["upper"]))
    return pts
P1, P2 = series(A1), series(A2)
def sm(R, k=32):
    e = np.array([r["mean_prefix_entropy_nats"] for r in R]); return np.arange(k, len(e) + 1), np.convolve(e, np.ones(k) / k, mode="valid")
E1, E2 = sm(R1), sm(R2)
W_, H1, H2, L, R_, T, GAP = 880, 340, 200, 70, 30, 40, 56
xmax = 1024
def sx(p): return L + (W_ - L - R_) * p / xmax
lo_y = min(p[1] for p in P1 + P2) - 0.02; hi_y = max(max(p[3] for p in P1 + P2), lib) + 0.03
def sy(v): return T + (H1 - T - 40) * (1 - (v - lo_y) / (hi_y - lo_y))
def sy2(v, lo=0.3, hi=1.9): return H1 + GAP + (H2 - 40) * (1 - (v - lo) / (hi - lo))
ink, ink2, mute, grid, base = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
c1, c2 = "#2a78d6", "#eb6834"   # categorical slots 1 and 2, fixed order: 1x, 2x
g = [f'<text x="{L}" y="22" class="t1">Absolute EWR by checkpoint — 1× (802k) vs 2× (1.6M) setup model, pooled over 8 handcrafted movers</text>']
for v in np.arange(math.ceil(lo_y * 20) / 20, hi_y, 0.05):
    g.append(f'<line x1="{L}" x2="{W_-R_}" y1="{sy(v):.1f}" y2="{sy(v):.1f}" stroke="{grid}"/><text x="{L-8}" y="{sy(v)+4:.1f}" class="ax" text-anchor="end">{v:.2f}</text>')
for p in range(0, 1025, 128): g.append(f'<text x="{sx(p):.1f}" y="{H1-16}" class="ax" text-anchor="middle">{p}</text>')
g.append(f'<line x1="{L}" x2="{W_-R_}" y1="{sy(lib):.1f}" y2="{sy(lib):.1f}" stroke="{mute}" stroke-dasharray="6 4"/><text x="{W_-R_}" y="{sy(lib)-6:.1f}" class="lab" text-anchor="end">handcrafted library {lib:.3f}</text>')
for P, col, name in ((P1, c1, "1×"), (P2, c2, "2×")):
    g.append('<polyline points="' + " ".join(f"{sx(c):.1f},{sy(v):.1f}" for c, v, _, _ in P) + f'" fill="none" stroke="{col}" stroke-width="2"/>')
    for c, v, lo, hi in P:
        g.append(f'<line x1="{sx(c):.1f}" x2="{sx(c):.1f}" y1="{sy(lo):.1f}" y2="{sy(hi):.1f}" stroke="{col}" stroke-width="1.5"/><circle cx="{sx(c):.1f}" cy="{sy(v):.1f}" r="4.5" fill="{col}" stroke="#fcfcfb" stroke-width="2"/>')
        g.append(f'<circle class="hit" cx="{sx(c):.1f}" cy="{sy(v):.1f}" r="13" fill="transparent" data-tip="{name} period {c}: {v:.4f}  [{lo:.4f}, {hi:.4f}]"/>')
    c, v, *_ = P[-1]; g.append(f'<text x="{sx(c)+8:.1f}" y="{sy(v)+4:.1f}" class="lab" text-anchor="start">{name} {v:.3f}</text>')
# legend (2 series -> required)
g.append(f'<rect x="{L}" y="{T-2}" width="10" height="10" fill="{c1}" rx="2"/><text x="{L+16}" y="{T+7}" class="lab">1× model, 4 blocks, 802k params</text>')
g.append(f'<rect x="{L+240}" y="{T-2}" width="10" height="10" fill="{c2}" rx="2"/><text x="{L+256}" y="{T+7}" class="lab">2× model, 8 blocks, 1.6M params</text>')
g.append(f'<text x="{L}" y="{H1+GAP-18}" class="t1">Policy entropy per period (nats/prefix, 32-period mean)</text>')
for v in (0.5, 1.0, 1.5):
    g.append(f'<line x1="{L}" x2="{W_-R_}" y1="{sy2(v):.1f}" y2="{sy2(v):.1f}" stroke="{grid}"/><text x="{L-8}" y="{sy2(v)+4:.1f}" class="ax" text-anchor="end">{v:.1f}</text>')
for (X, Y), col in ((E1, c1), (E2, c2)):
    g.append('<polyline points="' + " ".join(f"{sx(int(x)):.1f},{sy2(float(y)):.1f}" for x, y in zip(X, Y)) + f'" fill="none" stroke="{col}" stroke-width="2"/>')
for p in range(0, 1025, 128): g.append(f'<text x="{sx(p):.1f}" y="{H1+GAP+H2-20}" class="ax" text-anchor="middle">{p}</text>')
rows = ""
d2 = dict((c, (v, lo, hi)) for c, v, lo, hi in P2)
for c, v, lo, hi in P1:
    v2 = d2.get(c); rows += f"<tr><td>{c}</td><td>{v:.4f}</td><td>{(f'{v2[0]:.4f}' if v2 else '—')}</td><td>{(f'{v2[0]-v:+.4f}' if v2 else '—')}</td></tr>"
html = f"""<title>1x vs 2x Setup Model Curves</title>
<style>:root{{--surf:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--grid:#e1e0d9}}@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--surf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a}}}}:root[data-theme="dark"]{{--surf:#1a1a19;--ink:#fff;--ink2:#c3c2b7;--grid:#2c2c2a}}
body{{background:var(--surf);color:var(--ink);font:14px/1.45 -apple-system,Inter,Segoe UI,sans-serif;margin:0;padding:20px}}.wrap{{max-width:960px;margin:0 auto}}svg{{width:100%;height:auto;display:block}}
.t1{{font-size:15px;font-weight:600;fill:var(--ink)}}.ax{{font-size:11px;fill:#898781;font-variant-numeric:tabular-nums}}.lab{{font-size:12px;fill:var(--ink2);font-variant-numeric:tabular-nums}}
#tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--surf);padding:6px 9px;border-radius:6px;font-size:12px;opacity:0}}
table{{border-collapse:collapse;margin-top:16px;font-variant-numeric:tabular-nums;font-size:13px}}th,td{{padding:6px 12px;text-align:right;border-bottom:1px solid var(--grid)}}th{{color:var(--ink2)}}</style>
<div class="wrap"><svg viewBox="0 0 {W_} {H1+GAP+H2}" xmlns="http://www.w3.org/2000/svg">{''.join(g)}</svg>
<table><tr><th>period</th><th>1× EWR</th><th>2× EWR</th><th>2× − 1×</th></tr>{rows}</table></div><div id="tip"></div>
<script>const tip=document.getElementById('tip');document.querySelectorAll('.hit').forEach(h=>{{h.addEventListener('mousemove',e=>{{tip.textContent=h.dataset.tip;tip.style.left=(e.clientX+12)+'px';tip.style.top=(e.clientY-10)+'px';tip.style.opacity=1}});h.addEventListener('mouseleave',()=>tip.style.opacity=0)}});</script>"""
open(out, "w").write(html); print("overlay written:", out, "| 2x points:", [c for c, *_ in P2])
