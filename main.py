import tkinter as tk
from tkinter import ttk
import os
import shutil
import string
import gzip

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader
from PIL import Image, ImageDraw
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# ==========================================
# CONFIGURATION
# ==========================================

MODEL_PATH = "charnet32.pth"
OLD_MODEL_PATH = "digitnet32.pth"
CORRECTION_PATH = "charnet32_corrections.pt"

DATA_DIR = "./data"
EMNIST_SPLIT = "balanced"

# NIST's download server is frequently slow/flaky.  This mirror hosts the
# exact same IDX files split individually, so we can fetch just "balanced".
EMNIST_MIRROR = "https://huggingface.co/datasets/Heliosoph/EMNIST/resolve/main"
EMNIST_RAW_FILES = (
    "emnist-balanced-train-images-idx3-ubyte",
    "emnist-balanced-train-labels-idx1-ubyte",
    "emnist-balanced-test-images-idx3-ubyte",
    "emnist-balanced-test-labels-idx1-ubyte",
)

# EMNIST "balanced" contains 10 digits + 37 letters.  Letters that are
# visually identical in upper/lower case (c, o, s, ...) are merged, which
# is what a single-character drawing app can realistically distinguish.
CHAR_CLASSES = sorted(
    set(string.digits + string.ascii_letters)
    - {"c", "i", "j", "k", "l", "m", "o", "p", "s", "u", "v", "w", "x", "y", "z"}
)
NUM_CLASSES = len(CHAR_CLASSES)

# The EMNIST IDX files store images transposed relative to MNIST.
EMNIST_TRANSPOSE = True

BATCH_SIZE = 128
LEARNING_RATE = 0.001
AUTO_TRAIN_EPOCHS = 15

NORM_MEAN = 0.1307
NORM_STD = 0.3081
BLACK_NORM = (0.0 - NORM_MEAN) / NORM_STD

CANVAS_SIZE = 280
MODEL_SIZE = 32
GLYPH_BOX = 20
BRUSH_WIDTH = 18

TOP_K = 5

# Online correction settings.  Corrections are stored as feature prototypes
# and blended with the frozen model; network weights are never modified, so a
# correction cannot erase classes the model already knows.
CORRECTION_LIMIT = 200
CORRECTION_THRESHOLD = 0.90
CORRECTION_BLEND = 0.90
CORRECTION_TEMPERATURE = 0.03
CORRECTION_DUPLICATE = 0.995


# ==========================================
# 1. NEURAL NETWORK
# ==========================================

class CharNet32(nn.Module):
    """Small CNN that recognises 47 characters (digits and letters)."""

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 16 x 16

            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 8 x 8

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),                       # 4 x 4
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.30),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.40),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):

        x = self.features(x)
        x = self.classifier(x)

        return x


# ==========================================
# 2. DATASET HELPERS
# ==========================================

def _upright(img):
    return img.transpose(Image.Transpose.TRANSPOSE)


def _training_transform():

    steps = []

    if EMNIST_TRANSPOSE:
        steps.append(transforms.Lambda(_upright))

    steps.extend([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.08, 0.08),
            scale=(0.90, 1.10),
            shear=7,
            interpolation=transforms.InterpolationMode.BILINEAR,
            fill=0
        ),
        transforms.Pad(2),                  # 28 x 28 -> 32 x 32
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])

    return transforms.Compose(steps)


def _evaluation_transform():

    steps = []

    if EMNIST_TRANSPOSE:
        steps.append(transforms.Lambda(_upright))

    steps.extend([
        transforms.Pad(2),
        transforms.ToTensor(),
        transforms.Normalize((NORM_MEAN,), (NORM_STD,)),
    ])

    return transforms.Compose(steps)


def _ensure_emnist_files():
    """Fetch the balanced IDX files from the mirror if they are missing."""

    raw_folder = os.path.join(DATA_DIR, "EMNIST", "raw")

    os.makedirs(raw_folder, exist_ok=True)

    missing = [
        name for name in EMNIST_RAW_FILES
        if not os.path.exists(os.path.join(raw_folder, name))
    ]

    if not missing:
        return

    print("Downloading EMNIST-balanced from mirror...")

    for name in missing:

        url = f"{EMNIST_MIRROR}/{name}.gz"
        gz_path = os.path.join(raw_folder, name + ".gz")
        out_path = os.path.join(raw_folder, name)

        print("  ", name, flush=True)

        torch.hub.download_url_to_file(url, gz_path)

        with gzip.open(gz_path, "rb") as source:
            with open(out_path, "wb") as destination:
                shutil.copyfileobj(source, destination)

        os.remove(gz_path)


def _load_emnist(train, transform):
    """Load an EMNIST split and make the targets zero-based."""

    try:
        _ensure_emnist_files()
    except Exception as error:
        print("Mirror download failed:", error)
        print("Falling back to torchvision's downloader...\n")

    dataset = torchvision.datasets.EMNIST(
        root=DATA_DIR,
        split=EMNIST_SPLIT,
        train=train,
        download=True,
        transform=transform
    )

    # The raw EMNIST labels are 1-based (1..47).  torchvision keeps them
    # that way, so shift them to 0..46 for CrossEntropyLoss.
    targets = torch.as_tensor(dataset.targets)

    if int(targets.min()) == 1:
        dataset.targets = targets - 1

    return dataset


def build_emnist_datasets():

    train_dataset = _load_emnist(True, _training_transform())
    test_dataset = _load_emnist(False, _evaluation_transform())

    return train_dataset, test_dataset


# ==========================================
# 3. TRAINING
# ==========================================

def evaluate(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            predictions = model(images).argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    return 100 * correct / total


def train_model(
    device,
    epochs=AUTO_TRAIN_EPOCHS,
    batch_size=BATCH_SIZE,
    lr=LEARNING_RATE,
    save_path=MODEL_PATH
):

    save_path = os.fspath(save_path)

    print(
        f"\nTraining CharNet32 on EMNIST-{EMNIST_SPLIT} "
        f"({NUM_CLASSES} classes) for {epochs} epoch(s)...\n"
    )

    train_dataset, test_dataset = build_emnist_datasets()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0
    )

    model = CharNet32().to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=1e-4
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, epochs)
    )

    for epoch in range(epochs):

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        for batch_index, (images, labels) in enumerate(train_loader, start=1):

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            if batch_index % 100 == 0:
                print(
                    f"\r  Epoch {epoch + 1}/{epochs} | "
                    f"batch {batch_index}/{len(train_loader)}",
                    end="",
                    flush=True
                )

        scheduler.step()

        train_accuracy = 100 * correct / total
        average_loss = total_loss / len(train_loader)
        test_accuracy = evaluate(model, test_loader, device)

        print(
            f"\rEpoch {epoch + 1}/{epochs} | "
            f"Loss: {average_loss:.4f} | "
            f"Train: {train_accuracy:.2f}% | "
            f"Test: {test_accuracy:.2f}%"
        )

    save_directory = os.path.dirname(os.path.abspath(save_path))
    os.makedirs(save_directory, exist_ok=True)
    temporary_path = save_path + ".tmp"
    try:
        torch.save(model.state_dict(), temporary_path)
        os.replace(temporary_path, save_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print("\nModel saved successfully!")
    print("File:", save_path)

    return model


def load_or_train(device):

    if os.path.exists(MODEL_PATH):

        print("\nExisting trained model found.")
        print("Loading weights...\n")

        model = CharNet32().to(device)

        try:

            model.load_state_dict(
                torch.load(
                    MODEL_PATH,
                    map_location=device,
                    weights_only=True
                )
            )

            model.eval()

            return model

        except Exception as error:

            print("Could not load the existing model:", error)
            print("Retraining from scratch...\n")

    else:

        print("\nNo trained character model found.")

        if os.path.exists(OLD_MODEL_PATH):
            print("(The old digit-only model is not compatible; retraining.)")

        print("Starting training...\n")

    return train_model(device)


# ==========================================
# 4. DRAWING APPLICATION
# ==========================================

class DrawingApp:

    def __init__(self, root, model, device):

        self.root = root
        self.model = model
        self.device = device

        self.image = Image.new(
            "L",
            (CANVAS_SIZE, CANVAS_SIZE),
            0
        )

        self.draw = ImageDraw.Draw(self.image)

        self.last_x = None
        self.last_y = None

        self.last_tensor = None
        self.last_prediction = None

        self.corrections = self.load_corrections()

        self._predict_job = None

        self.bar_labels = []
        self.bars = []
        self.percentages = []

        self.build_ui()

    def build_ui(self):

        self.root.title("CharNet32 - Handwritten Digit & Letter Recogniser")
        self.root.resizable(False, False)

        main = ttk.Frame(self.root, padding=15)
        main.grid()

        # Drawing canvas
        self.canvas = tk.Canvas(
            main,
            width=CANVAS_SIZE,
            height=CANVAS_SIZE,
            bg="black",
            highlightthickness=1
        )

        self.canvas.grid(row=0, column=0, rowspan=2)

        self.canvas.bind("<ButtonPress-1>", self.start_drawing)
        self.canvas.bind("<B1-Motion>", self.draw_line)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        # Buttons under the canvas
        buttons = ttk.Frame(main)
        buttons.grid(row=2, column=0, pady=10)

        ttk.Button(
            buttons,
            text="Predict",
            command=self.predict
        ).grid(row=0, column=0, padx=5)

        ttk.Button(
            buttons,
            text="Clear",
            command=self.clear
        ).grid(row=0, column=1, padx=5)

        self.auto_predict = tk.BooleanVar(value=True)

        ttk.Checkbutton(
            buttons,
            text="Auto predict",
            variable=self.auto_predict
        ).grid(row=0, column=2, padx=5)

        ttk.Label(
            main,
            text="Tip: draw the character large, filling most of the canvas.",
            foreground="#666666"
        ).grid(row=3, column=0, pady=(0, 4))

        # Prediction panel
        right = ttk.Frame(main, padding=(20, 0))
        right.grid(row=0, column=1, rowspan=4, sticky="n")

        ttk.Label(
            right,
            text="Prediction"
        ).grid(row=0, column=0, columnspan=3)

        self.prediction = ttk.Label(
            right,
            text="-",
            font=("Helvetica", 64, "bold"),
            width=3,
            anchor="center"
        )

        self.prediction.grid(row=1, column=0, columnspan=3)

        self.status = ttk.Label(
            right,
            text="Draw a digit or letter!",
            wraplength=250
        )

        self.status.grid(row=2, column=0, columnspan=3, pady=(0, 8))

        ttk.Separator(right, orient="horizontal").grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew"
        )

        ttk.Label(
            right,
            text="Top predictions"
        ).grid(row=4, column=0, columnspan=3, pady=(8, 4))

        # Confidence rows (top 5 instead of all 47 classes)
        for row in range(TOP_K):

            label = ttk.Label(
                right,
                text="-",
                width=3,
                anchor="center",
                font=("Helvetica", 14, "bold")
            )

            label.grid(row=5 + row, column=0)

            bar = ttk.Progressbar(
                right,
                length=170,
                maximum=100
            )

            bar.grid(row=5 + row, column=1, padx=5, pady=2)

            percentage = ttk.Label(
                right,
                text="0.0%",
                width=7,
                anchor="e"
            )

            percentage.grid(row=5 + row, column=2)

            self.bar_labels.append(label)
            self.bars.append(bar)
            self.percentages.append(percentage)

        feedback_row = 5 + TOP_K

        ttk.Separator(right, orient="horizontal").grid(
            row=feedback_row,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(8, 4)
        )

        ttk.Label(
            right,
            text="Was this right?"
        ).grid(row=feedback_row + 1, column=0, columnspan=3)

        feedback_buttons = ttk.Frame(right)
        feedback_buttons.grid(
            row=feedback_row + 2,
            column=0,
            columnspan=3,
            pady=(4, 0)
        )

        ttk.Button(
            feedback_buttons,
            text="\u2713 Correct",
            command=self.mark_correct
        ).grid(row=0, column=0, padx=4)

        ttk.Button(
            feedback_buttons,
            text="\u2717 Wrong",
            command=self.mark_wrong
        ).grid(row=0, column=1, padx=4)

        ttk.Button(
            feedback_buttons,
            text="Reset",
            command=self.reset_corrections
        ).grid(row=0, column=2, padx=4)

        ttk.Label(
            right,
            text="Corrections are remembered safely and never overwrite the trained network.",
            foreground="#666666",
            wraplength=250
        ).grid(row=feedback_row + 3, column=0, columnspan=3, pady=(6, 0))

        self.root.bind("<Return>", lambda e: self.predict())
        self.root.bind("<Escape>", lambda e: self.clear())

    # --------------------------------------
    # Drawing functions
    # --------------------------------------

    def start_drawing(self, event):

        self._cancel_auto_predict()

        # The previous prediction no longer matches the canvas.
        self.last_tensor = None
        self.last_prediction = None

        self.last_x = event.x
        self.last_y = event.y

        self.paint_dot(event.x, event.y)

    def draw_line(self, event):

        if self.last_x is not None:

            self.paint_line(
                self.last_x,
                self.last_y,
                event.x,
                event.y
            )

        self.last_x = event.x
        self.last_y = event.y

    def stop_drawing(self, event):

        self.last_x = None
        self.last_y = None

        if self.auto_predict.get() and self.image.getbbox() is not None:

            self._cancel_auto_predict()

            self._predict_job = self.root.after(500, self.predict)

    def _cancel_auto_predict(self):

        if self._predict_job is not None:

            self.root.after_cancel(self._predict_job)
            self._predict_job = None

    def paint_dot(self, x, y):

        r = BRUSH_WIDTH / 2

        self.canvas.create_oval(
            x - r, y - r, x + r, y + r,
            fill="white",
            outline="white"
        )

        self.draw.ellipse(
            [x - r, y - r, x + r, y + r],
            fill=255
        )

    def paint_line(self, x1, y1, x2, y2):

        r = BRUSH_WIDTH / 2

        self.canvas.create_line(
            x1, y1, x2, y2,
            fill="white",
            width=BRUSH_WIDTH,
            capstyle=tk.ROUND,
            joinstyle=tk.ROUND
        )

        self.draw.line(
            [(x1, y1), (x2, y2)],
            fill=255,
            width=BRUSH_WIDTH
        )

        self.draw.ellipse(
            [x2 - r, y2 - r, x2 + r, y2 + r],
            fill=255
        )

    # --------------------------------------
    # Image preprocessing
    # --------------------------------------

    def preprocess(self):
        """Turn the canvas into a centred, MNIST-style 32x32 tensor.

        The old version resized the whole 280x280 canvas to 32x32.  That
        made the character tiny and dependent on where/how large it was
        drawn.  Here we crop to the ink, scale the glyph to a 20x20 box
        (like EMNIST/MNIST), and centre it by its centre of mass.
        """

        bbox = self.image.getbbox()

        if bbox is None:
            return None

        glyph = self.image.crop(bbox)

        width, height = glyph.size

        scale = GLYPH_BOX / max(width, height)

        new_width = max(1, int(round(width * scale)))
        new_height = max(1, int(round(height * scale)))

        # BOX = area averaging, which avoids the ringing/halos that
        # LANCZOS can create when shrinking a hard-edged drawing.
        resample = (
            Image.Resampling.LANCZOS
            if scale > 1
            else Image.Resampling.BOX
        )

        glyph = glyph.resize((new_width, new_height), resample)

        # Drop very faint anti-aliasing fringe.
        glyph = glyph.point(lambda p: 0 if p < 20 else p)

        # Centre of mass (same convention as MNIST/EMNIST).
        data = torch.frombuffer(
            bytearray(glyph.tobytes()),
            dtype=torch.uint8
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
            bytearray(padded.tobytes()),
            dtype=torch.uint8
        ).clone().float()

        tensor = tensor.reshape(1, 1, MODEL_SIZE, MODEL_SIZE) / 255.0
        tensor = (tensor - NORM_MEAN) / NORM_STD

        return tensor

    # --------------------------------------
    # Prediction
    # --------------------------------------

    def predict(self, prefix=None):

        self._predict_job = None

        if self.image.getbbox() is None:

            self.status.config(text="Draw something first!")
            return

        tensor = self.preprocess()

        if tensor is None:
            return

        self.last_tensor = tensor.detach().cpu()

        self.model.eval()

        with torch.no_grad():

            x = tensor.to(self.device)

            flat = self.model.classifier[:-1](self.model.features(x))
            features = F.normalize(flat, dim=1)
            logits = self.model.classifier[-1](flat)

            probabilities = torch.softmax(logits, dim=1)
            probabilities = self.blend_corrections(probabilities, features)
            probabilities = probabilities[0].cpu()

        top_probabilities, top_indices = probabilities.topk(TOP_K)

        best = int(top_indices[0].item())

        self.last_prediction = best

        confidence = top_probabilities[0].item() * 100

        self.prediction.config(text=CHAR_CLASSES[best])

        kind = "digit" if CHAR_CLASSES[best].isdigit() else "letter"

        status = (
            f"{kind.capitalize()} {CHAR_CLASSES[best]} \u00b7 "
            f"{confidence:.1f}% confidence"
        )

        if prefix is not None:
            status = f"{prefix} \u00b7 {status}"

        self.status.config(text=status)

        for row in range(TOP_K):

            index = int(top_indices[row].item())
            percentage = top_probabilities[row].item() * 100

            self.bar_labels[row].config(text=CHAR_CLASSES[index])

            self.bars[row]["value"] = percentage

            self.percentages[row].config(
                text=f"{percentage:.1f}%"
            )

    # --------------------------------------
    # Feedback / online learning
    # --------------------------------------

    def mark_correct(self):

        if self.last_tensor is None or self.last_prediction is None:

            self.status.config(text="Make a prediction first!")
            return

        self.teach(
            self.last_prediction,
            prefix="Correct \u2713 reinforced"
        )

    def mark_wrong(self):

        if self.last_tensor is None or self.last_prediction is None:

            self.status.config(text="Make a prediction first!")
            return

        self.open_correction_dialog()

    def open_correction_dialog(self):

        dialog = tk.Toplevel(self.root)

        dialog.title("Correct the prediction")
        dialog.transient(self.root)
        dialog.resizable(False, False)
        dialog.grab_set()

        frame = ttk.Frame(dialog, padding=15)
        frame.grid()

        ttk.Label(
            frame,
            text="What did you actually draw?"
        ).grid(row=0, column=0, columnspan=2, pady=(0, 8))

        value = tk.StringVar(
            value=CHAR_CLASSES[self.last_prediction]
        )

        combo = ttk.Combobox(
            frame,
            textvariable=value,
            values=CHAR_CLASSES,
            state="readonly",
            width=8
        )

        combo.grid(row=1, column=0, columnspan=2, pady=(0, 12))
        combo.focus_set()

        def apply():

            choice = value.get()

            if choice not in CHAR_CLASSES:
                return

            label = CHAR_CLASSES.index(choice)

            dialog.destroy()

            self.teach(
                label,
                prefix=f"Corrected to {CHAR_CLASSES[label]} \u2713"
            )

        ttk.Button(
            frame,
            text="Remember",
            command=apply
        ).grid(row=2, column=0, padx=4)

        ttk.Button(
            frame,
            text="Cancel",
            command=dialog.destroy
        ).grid(row=2, column=1, padx=4)

        dialog.bind("<Return>", lambda e: apply())
        dialog.bind("<Escape>", lambda e: dialog.destroy())

        dialog.update_idletasks()

        x = self.root.winfo_rootx() + (
            self.root.winfo_width() - dialog.winfo_width()
        ) // 2

        y = self.root.winfo_rooty() + (
            self.root.winfo_height() - dialog.winfo_height()
        ) // 2

        dialog.geometry(f"+{max(0, x)}+{max(0, y)}")

    def teach(self, label, prefix):

        if self.last_tensor is None:
            return

        self.model.eval()

        try:

            with torch.no_grad():

                x = self.last_tensor.to(self.device)
                flat = self.model.classifier[:-1](self.model.features(x))
                feature = F.normalize(flat, dim=1)[0].cpu()

            # Very similar prototypes are replaced so the newest correction
            # wins if the same drawing is labelled differently twice.
            if self.corrections["features"]:

                memory = torch.stack(self.corrections["features"])
                similarity = memory @ feature
                keep = similarity < CORRECTION_DUPLICATE

                self.corrections["features"] = [
                    item for item, keep_item in zip(
                        self.corrections["features"], keep.tolist()
                    ) if keep_item
                ]

                self.corrections["labels"] = [
                    item for item, keep_item in zip(
                        self.corrections["labels"], keep.tolist()
                    ) if keep_item
                ]

            self.corrections["features"].append(feature)
            self.corrections["labels"].append(int(label))

            if len(self.corrections["features"]) > CORRECTION_LIMIT:

                self.corrections["features"] = self.corrections["features"][-CORRECTION_LIMIT:]
                self.corrections["labels"] = self.corrections["labels"][-CORRECTION_LIMIT:]

            self.save_corrections()

        except Exception as error:

            self.status.config(text=f"Could not store correction: {error}")
            return

        self.predict(prefix=prefix)

    def load_corrections(self):

        empty = {"features": [], "labels": []}

        if not os.path.exists(CORRECTION_PATH):
            return empty

        try:

            data = torch.load(
                CORRECTION_PATH,
                map_location="cpu",
                weights_only=False
            )

            features = data.get("features", [])
            labels = data.get("labels", [])

            if isinstance(features, torch.Tensor):
                features = [features[index] for index in range(features.size(0))]

            if isinstance(labels, torch.Tensor):
                labels = labels.tolist()

            stored_features = []
            stored_labels = []

            for feature, label in zip(features, labels):

                label = int(label)

                if not 0 <= label < len(CHAR_CLASSES):
                    continue

                vector = torch.as_tensor(feature, dtype=torch.float32).reshape(-1)

                if vector.numel() != 128:
                    continue

                stored_features.append(F.normalize(vector, dim=0))
                stored_labels.append(label)

            stored_features = stored_features[-CORRECTION_LIMIT:]
            stored_labels = stored_labels[-CORRECTION_LIMIT:]

            return {"features": stored_features, "labels": stored_labels}

        except Exception as error:

            print("Could not load corrections:", error)
            return empty

    def save_corrections(self):

        if self.corrections["features"]:

            features = torch.stack(self.corrections["features"]).cpu()
            labels = torch.tensor(self.corrections["labels"], dtype=torch.long)

        else:

            features = torch.empty(0, 128)
            labels = torch.empty(0, dtype=torch.long)

        temporary = CORRECTION_PATH + ".tmp"

        torch.save({"features": features, "labels": labels}, temporary)

        os.replace(temporary, CORRECTION_PATH)

    def reset_corrections(self):

        self.corrections = {"features": [], "labels": []}

        try:

            if os.path.exists(CORRECTION_PATH):
                os.remove(CORRECTION_PATH)

        except OSError as error:

            self.status.config(text=f"Could not clear corrections: {error}")
            return

        if self.last_tensor is not None:
            self.predict(prefix="Corrections reset")
        else:
            self.status.config(text="Corrections reset.")

    def blend_corrections(self, probabilities, features):
        """Blend frozen-model probabilities with stored correction prototypes."""

        if not self.corrections["features"]:
            return probabilities

        device = probabilities.device

        memory = torch.stack(
            [feature.to(device) for feature in self.corrections["features"]]
        )

        labels = torch.tensor(
            self.corrections["labels"],
            dtype=torch.long,
            device=device
        )

        similarity = features @ memory.T

        class_similarity = torch.full(
            (features.size(0), len(CHAR_CLASSES)),
            -1.0,
            device=device
        )

        for label in torch.unique(labels).tolist():

            mask = labels == label

            class_similarity[:, label] = similarity[:, mask].max(dim=1).values

        best_similarity = class_similarity.max(dim=1).values

        neighbour_probabilities = torch.softmax(
            class_similarity / CORRECTION_TEMPERATURE,
            dim=1
        )

        strength = CORRECTION_BLEND * (
            (best_similarity - CORRECTION_THRESHOLD)
            / (1 - CORRECTION_THRESHOLD)
        ).clamp(0, 1)

        return (
            (1 - strength.unsqueeze(1)) * probabilities
            + strength.unsqueeze(1) * neighbour_probabilities
        )

    # --------------------------------------
    # Clear canvas
    # --------------------------------------

    def clear(self):

        self._cancel_auto_predict()

        self.canvas.delete("all")

        self.draw.rectangle(
            [0, 0, CANVAS_SIZE, CANVAS_SIZE],
            fill=0
        )

        self.last_tensor = None
        self.last_prediction = None

        self.prediction.config(text="-")
        self.status.config(text="Draw a digit or letter!")

        for row in range(TOP_K):

            self.bar_labels[row].config(text="-")

            self.bars[row]["value"] = 0

            self.percentages[row].config(text="0.0%")


# ==========================================
# 5. MAIN PROGRAM
# ==========================================

def get_device():

    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def main():

    print("================================")
    print("          CHARNET32")
    print("   digits + letters recogniser")
    print("================================")

    device = get_device()

    print("Training device:", device)

    model = load_or_train(device)

    model.eval()

    print("\nOpening drawing application...")

    root = tk.Tk()

    DrawingApp(root, model, device)

    root.mainloop()


if __name__ == "__main__":
    main()
