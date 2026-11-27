import tkinter as tk
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from main import (
    BRUSH_WIDTH,
    CHAR_CLASSES,
    GLYPH_BOX,
    MODEL_PATH,
    MODEL_SIZE,
    NORM_MEAN,
    NORM_STD,
    CharNet32,
)

CLASSES = CHAR_CLASSES

model = CharNet32()
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True)
)
model.eval()

linears = [m for m in model.modules() if isinstance(m, torch.nn.Linear)]
W1, B1 = linears[0].weight.detach(), linears[0].bias.detach()
W2, B2 = linears[1].weight.detach(), linears[1].bias.detach()

idx0 = torch.linspace(0, W1.shape[1] - 1, 32).long()
idx1 = torch.linspace(0, W1.shape[0] - 1, 24).long()
idx2 = torch.arange(W2.shape[0])

captured = {}
def hook_fn(module, inp, out):
    captured["feat"] = inp[0][0].detach()
linears[0].register_forward_hook(hook_fn)

root = tk.Tk()
root.title("Live CharNet32 Neural Web")
root.configure(bg="#ffffff")

left_frame = tk.Frame(root, bg="#ffffff", padx=25, pady=25)
left_frame.pack(side=tk.LEFT, fill=tk.Y)

tk.Label(left_frame, text="Draw Here (280x280)", fg="#000000", bg="#ffffff", font=("Helvetica", 14, "bold")).pack(pady=(0, 10))

canvas_size = 280
draw_canvas = tk.Canvas(left_frame, width=canvas_size, height=canvas_size, bg="#000000", highlightthickness=2, highlightbackground="#000000")
draw_canvas.pack()

pil_img = Image.new("L", (canvas_size, canvas_size), 0)
pil_draw = ImageDraw.Draw(pil_img)

pred_label = tk.Label(left_frame, text="Prediction: —", fg="#000000", bg="#ffffff", font=("Helvetica", 20, "bold"))
pred_label.pack(pady=15)

right_frame = tk.Frame(root, bg="#ffffff", padx=25, pady=25)
right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

W, H = 760, 720
net_canvas = tk.Canvas(right_frame, width=W, height=H, bg="#000000", highlightthickness=2, highlightbackground="#000000")
net_canvas.pack()

xs = [60, 380, 690]
def get_coords(col, count):
    step = (H - 60) / max(count - 1, 1)
    return [(xs[col], 30 + i * step) for i in range(count)]

pts = [get_coords(0, len(idx0)), get_coords(1, len(idx1)), get_coords(2, len(idx2))]

edge_items = []
# Show only the two strongest displayed links entering each shown neuron.
# The full model still uses every weight; this keeps the live diagram readable.
for d_i, (x2, y2) in enumerate(pts[1]):
    row = W1[idx1[d_i], idx0]
    strongest_sources = torch.topk(row.abs(), k=min(2, row.numel())).indices.tolist()
    for s_i in strongest_sources:
        x1, y1 = pts[0][s_i]
        w = row[s_i].item()
        item = net_canvas.create_line(x1, y1, x2, y2, fill="#303030", width=1.25)
        edge_items.append((item, 0, idx0[s_i].item(), idx1[d_i].item(), w))

for d_i, (x2, y2) in enumerate(pts[2]):
    row = W2[idx2[d_i], idx1]
    strongest_sources = torch.topk(row.abs(), k=min(2, row.numel())).indices.tolist()
    for s_i in strongest_sources:
        x1, y1 = pts[1][s_i]
        w = row[s_i].item()
        item = net_canvas.create_line(x1, y1, x2, y2, fill="#303030", width=1.25)
        edge_items.append((item, 1, idx1[s_i].item(), idx2[d_i].item(), w))

node_items = [[], [], []]
for col_i, col in enumerate(pts):
    for row_i, (x, y) in enumerate(col):
        r = 5 if col_i < 2 else 4.5
        n = net_canvas.create_oval(x-r, y-r, x+r, y+r, fill="#333333", outline="#666666")
        node_items[col_i].append(n)
        if col_i == 2:
            ch = CLASSES[row_i] if row_i < len(CLASSES) else str(row_i)
            net_canvas.create_text(x + 18, y, text=ch, fill="#cccccc", font=("Menlo", 10, "bold"))

def preprocess_canvas(image):
    """Match DrawingApp.preprocess() in main.py for a 280x280 ink canvas."""
    bbox = image.getbbox()
    if bbox is None:
        return None

    glyph = image.crop(bbox)
    width, height = glyph.size
    scale = GLYPH_BOX / max(width, height)
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))

    resample = (
        Image.Resampling.LANCZOS
        if scale > 1
        else Image.Resampling.BOX
    )
    glyph = glyph.resize((new_width, new_height), resample)
    glyph = glyph.point(lambda p: 0 if p < 20 else p)

    data = torch.frombuffer(
        bytearray(glyph.tobytes()), dtype=torch.uint8
    ).clone().reshape(new_height, new_width).float()
    total_mass = data.sum().item()

    if total_mass > 0:
        xs = torch.arange(new_width, dtype=torch.float32)
        ys = torch.arange(new_height, dtype=torch.float32)
        centre_x = float((data.sum(dim=0) * xs).sum() / total_mass)
        centre_y = float((data.sum(dim=1) * ys).sum() / total_mass)
    else:
        centre_x = new_width / 2
        centre_y = new_height / 2

    offset_x = int(round(13.5 - centre_x))
    offset_y = int(round(13.5 - centre_y))
    offset_x = max(0, min(28 - new_width, offset_x))
    offset_y = max(0, min(28 - new_height, offset_y))

    centred = Image.new("L", (28, 28), 0)
    centred.paste(glyph, (offset_x, offset_y))
    padded = Image.new("L", (MODEL_SIZE, MODEL_SIZE), 0)
    padded.paste(centred, (2, 2))

    tensor = torch.frombuffer(
        bytearray(padded.tobytes()), dtype=torch.uint8
    ).clone().float()
    tensor = tensor.reshape(1, 1, MODEL_SIZE, MODEL_SIZE) / 255.0
    return (tensor - NORM_MEAN) / NORM_STD


def update_network():
    tensor = preprocess_canvas(pil_img)
    if tensor is None:
        return

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]
        feat0 = captured["feat"]
        hidden1 = F.relu(F.linear(feat0, W1, B1))

    top_idx = torch.argmax(probs).item()
    conf = probs[top_idx].item() * 100
    pred_char = CLASSES[top_idx] if top_idx < len(CLASSES) else str(top_idx)
    pred_label.config(text="Prediction: " + str(pred_char) + " (" + str(round(conf, 1)) + "%)")

    a0 = feat0 / (feat0.max().item() + 1e-5)
    a1 = hidden1 / (hidden1.max().item() + 1e-5)

    for i, idx in enumerate(idx0):
        v = min(1.0, a0[idx].item())
        color = "#38bdf8" if v > 0.3 else "#333333"
        net_canvas.itemconfig(node_items[0][i], fill=color)

    for i, idx in enumerate(idx1):
        v = min(1.0, a1[idx].item())
        color = "#38bdf8" if v > 0.3 else "#333333"
        net_canvas.itemconfig(node_items[1][i], fill=color)

    for i in range(len(idx2)):
        p = probs[i].item()
        if i == top_idx:
            color = "#22c55e"
        elif p > 0.05:
            color = "#38bdf8"
        else:
            color = "#333333"
        net_canvas.itemconfig(node_items[2][i], fill=color)

    for item, layer_id, s_idx, d_idx, w in edge_items:
        src_act = a0[s_idx].item() if layer_id == 0 else a1[s_idx].item()
        signal = src_act * w
        if signal > 0.15:
            net_canvas.itemconfig(item, fill="#42d7ff", width=min(2.6, 1.5 + signal))
        elif signal < -0.25:
            net_canvas.itemconfig(item, fill="#ffffff", width=2.0)
        else:
            net_canvas.itemconfig(item, fill="#303030", width=1.25)

last_x, last_y = None, None

def paint_dot(x, y):
    radius = BRUSH_WIDTH / 2
    draw_canvas.create_oval(
        x - radius, y - radius, x + radius, y + radius,
        fill="white", outline="white"
    )
    pil_draw.ellipse(
        [x - radius, y - radius, x + radius, y + radius], fill=255
    )

def start_paint(event):
    global last_x, last_y
    last_x, last_y = event.x, event.y
    paint_dot(event.x, event.y)
    update_network()

def paint(event):
    global last_x, last_y
    if last_x is not None:
        draw_canvas.create_line(last_x, last_y, event.x, event.y, fill="white", width=BRUSH_WIDTH, capstyle=tk.ROUND, smooth=True)
        pil_draw.line([last_x, last_y, event.x, event.y], fill=255, width=BRUSH_WIDTH)
    else:
        paint_dot(event.x, event.y)
    last_x, last_y = event.x, event.y
    update_network()

def reset_pos(event):
    global last_x, last_y
    last_x, last_y = None, None

def clear_canvas():
    draw_canvas.delete("all")
    pil_draw.rectangle([0, 0, canvas_size, canvas_size], fill=0)
    pred_label.config(text="Prediction: —")
    for item, *_ in edge_items:
        net_canvas.itemconfig(item, fill="#303030", width=1.25)
    for col in node_items:
        for n in col:
            net_canvas.itemconfig(n, fill="#333333")

draw_canvas.bind("<ButtonPress-1>", start_paint)
draw_canvas.bind("<B1-Motion>", paint)
draw_canvas.bind("<ButtonRelease-1>", reset_pos)

tk.Button(left_frame, text="Clear Canvas", command=clear_canvas, bg="#000000", fg="#ffffff", highlightbackground="#ffffff", font=("Helvetica", 12, "bold"), padx=15, pady=6).pack(pady=10)

root.mainloop()
