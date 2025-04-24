import torch.nn as nn
import torch

from model_application.deep_learning.model import GRU


class DebugLSTM(nn.Module):
    def __init__(
        self, 
        input_size: int,
        hidden_size: int = 1024,
        num_layers: int = 12,
        dropout_prob: float = 0.3,
        n_steps: int = 1
    ):
        super().__init__()
        
        self.model = GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout_prob=dropout_prob,
            n_steps=n_steps
        )

    def forward(self, input, target):
        print(f"🔹 [STEP 0] Input input.shape: {input.shape}")  # (batch_size, features)
        print(f"🔹 [STEP 0] Input target.shape: {target.shape}")  # (batch_size, )
        
        output = self.model(input)
        print(f"🔹 [STEP 1] Output output.shape: {output.shape}")
        
        return output

# Cấu hình LSTM
input_size = 58
n_steps = 1

device = "cuda" if torch.cuda.is_available() else "cpu"

# Khởi tạo mô hình
model = DebugLSTM(input_size=input_size, n_steps=n_steps).to(device)

# ====== Dữ liệu giả lập mới ======
batch_size = 4
time_steps = 30
features = 58

# Tạo input ngẫu nhiên (batch_size, time_steps, num_features)
input = torch.randn(batch_size, time_steps, features).to(device)

# Tạo target ngẫu nhiên (batch_size, n_steps)
target = torch.randint(100000, 40000000, (batch_size, n_steps)).to(device)

# ====== Chạy thử model ======
output = model(input, target)

# ====== In kết quả ======
print(f"✅ Output shapes: {[o.shape if o is not None else 'None' for o in output]}")  
