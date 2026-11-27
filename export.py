import torch

# 1. Import your model class (likely defined in train.py or main.py)
# Check train.py for the exact class name, e.g., CharNet32 or DigitNet32
from main import CharNet32

# 2. Initialize and load your chosen weights (e.g., charnet32.pth)
model = CharNet32()
model.load_state_dict(torch.load("charnet32.pth", map_location="cpu"))
model.eval()

# 3. Create a 32x32 grayscale dummy input based on the '32' in your filenames
dummy_input = torch.randn(1, 1, 32, 32)

# 4. Export to ONNX for Netron
torch.onnx.export(model, dummy_input, "charnet32.onnx")
print("Exported charnet32.onnx successfully!")
