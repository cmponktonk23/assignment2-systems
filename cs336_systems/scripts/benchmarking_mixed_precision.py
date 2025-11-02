import torch
from torch import nn

class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


device = torch.device("cuda")
model = ToyModel(32, 32).to(device)
criterion = nn.MSELoss()

x = torch.randn(4, 32, device=device, dtype=torch.float32)
target = torch.zeros_like(x)

model.zero_grad(set_to_none=True)

with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    print(f"fc1.weight.dtype={model.fc1.weight.dtype}, ln.weight.dtype={model.ln.weight.dtype}, fc2.weight.dtype={model.fc2.weight.dtype}")
    h1 = model.fc1(x)
    print(f"fc1 output dtype={h1.dtype}")
    h1 = model.relu(h1)
    h2 = model.ln(h1)
    print(f"ln output dtype={h2.dtype}")
    logits = model.fc2(h2)
    print(f"logits dtype={logits.dtype}")
    loss = criterion(logits, target)
    print("loss dtype=", loss.dtype)

loss.backward()
print("fc1.weight.grad dtype=", model.fc1.weight.grad.dtype)