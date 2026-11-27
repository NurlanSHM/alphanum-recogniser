import torch

# 1. Load your trained weights directly (no extra libraries needed)
state_dict = torch.load("charnet32.pth", map_location="cpu")
linear_weights = [v for v in state_dict.values() if v.ndim == 2]
w1, w2 = linear_weights[0], linear_weights[1]  # (128, 1024) and (47, 128)

# 2. Sample nodes so the web is readable instead of 130,000 overlapping lines
idx0 = torch.linspace(0, w1.shape[1] - 1, 32).long()  # 32 of 1024 flattened features
idx1 = torch.linspace(0, w1.shape[0] - 1, 24).long()  # 24 of 128 hidden neurons
idx2 = torch.arange(w2.shape[0])                      # All 47 output characters

W1_sub = w1[idx1][:, idx0]
W2_sub = w2[idx2][:, idx1]

W, H = 1200, 800
xs = [180, 600, 1020]
layers = [len(idx0), len(idx1), len(idx2)]

def node_coords(layer_idx, count):
    step = (H - 100) / max(count - 1, 1)
    return [(xs[layer_idx], 50 + i * step) for i in range(count)]

pts = [node_coords(i, c) for i, c in enumerate(layers)]
svg = [f'<svg xmlns="http://w3.org" width="{W}" height="{H}" style="background:#0f172a">']

def draw_edges(w_mat, src_pts, dst_pts):
    max_w = w_mat.abs().max().item()
    for d_i, (x2, y2) in enumerate(dst_pts):
        for s_i, (x1, y1) in enumerate(src_pts):
            val = w_mat[d_i, s_i].item()
            strength = abs(val) / max_w
            if strength < 0.15:  # hide weak noise connections
                continue
            color = "#38bdf8" if val > 0 else "#f87171"  # blue = positive, red = negative
            svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
                       f'stroke="{color}" stroke-opacity="{strength * 0.7:.2f}" stroke-width="{strength * 2:.2f}"/>')

draw_edges(W1_sub, pts[0], pts[1])
draw_edges(W2_sub, pts[1], pts[2])

for layer in pts:
    for x, y in layer:
        svg.append(f'<circle cx="{x}" cy="{y}" r="5" fill="#f8fafc" stroke="#334155" stroke-width="1.5"/>')

svg.append("</svg>")
with open("network_web.svg", "w") as f:
    f.write("\n".join(svg))

print("Saved network_web.svg! Double-click it to open in your browser.")

