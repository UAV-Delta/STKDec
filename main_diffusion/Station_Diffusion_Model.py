import torch
import torch.nn as nn
import numpy as np
import csv

# ========== 读取数据 ==========
def read_csv(file_path, column_index):
    data = []
    with open(file_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            data.append([float(row[column_index])])
    return data

def read_txt(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data.append([float(x) for x in line.strip().split()])
    return data

# ========== 文件路径 ==========
data_file = r'2750sorted_filtered_成都市station_data.csv'
kg_file = r'E:\wyf\STKDec-STKDec\Format_Setting\成都知识图谱扩展2750.txt'
condition_file = r'E:\wyf\STKDec-STKDec\Format_Setting\成都spatial_time.txt'
weight_file = r'无权重.txt'

data_lu = read_csv(data_file, 3)
kg_lu = read_txt(kg_file)
condition_lu = read_txt(condition_file)
weight_lu = read_txt(weight_file)

dataset = torch.Tensor(data_lu).float()
kg = torch.Tensor(kg_lu).float()
conditions = torch.Tensor(condition_lu).float()
weights = torch.Tensor(weight_lu).float().squeeze()

# ========== Diffusion 参数 ==========
num_steps = 200
betas = torch.linspace(-6, 6, num_steps)
betas = torch.sigmoid(betas) * (0.5e-2 - 1e-5) + 1e-5
alphas = 1 - betas
alphas_prod = torch.cumprod(alphas, 0)
alphas_bar_sqrt = torch.sqrt(alphas_prod)
one_minus_alphas_bar_sqrt = torch.sqrt(1 - alphas_prod)

def q_x(x_0, t):
    noise = torch.randn_like(x_0)
    return alphas_bar_sqrt[t] * x_0 + one_minus_alphas_bar_sqrt[t] * noise


# ========== MLP Diffusion Model ==========
class MLPDiffusion(nn.Module):
    def __init__(self, n_steps, num_units=128):
        super(MLPDiffusion, self).__init__()
        self.kg_proj = nn.Linear(4, 8)  # kg升维 4 -> 8

        self.mha = nn.MultiheadAttention(embed_dim=8, num_heads=1, batch_first=True)

        self.linears = nn.ModuleList([
            nn.Linear(9, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, num_units), nn.ReLU(),
            nn.Linear(num_units, 1)
        ])

        self.step_embeddings = nn.ModuleList([
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
            nn.Embedding(n_steps, num_units),
        ])

    def attention(self, cond1, cond2, kg):
        # 直接拼接 cond1 和 cond2，形状为 (B, 8)
        cond = torch.cat([cond1, cond2], dim=-1)  # (B, 4) + (B, 4) = (B, 8)

        kg_proj = self.kg_proj(kg)  # (B, 8)

        query = kg_proj.unsqueeze(1)  # (B, 1, 8)
        key = cond.unsqueeze(1)  # (B, 1, 8)
        value = cond.unsqueeze(1)  # (B, 1, 8)

        out, _ = self.mha(query, key, value)
        return out.squeeze(1)  # (B, 8)

    def forward(self, x, t, condition1, condition2, kg):
        # 处理 attention 输出并拼接到输入 x
        attention_output = self.attention(condition1, condition2, kg)  # (B, 8)
        x = torch.cat([x, attention_output], dim=1)  # (B, 1+8=9)

        for idx, emb in enumerate(self.step_embeddings):
            t_emb = emb(t)
            x = self.linears[2 * idx](x) + t_emb
            x = self.linears[2 * idx + 1](x)

        x = self.linears[-1](x)
        return x


# ========== Loss ==========
def diffusion_loss_fn(model, x_0, condition1, condition2, kg, weight, n_steps):
    t = torch.randint(0, n_steps, size=(x_0.shape[0]//2,))
    t = torch.cat([t, n_steps - 1 - t], dim=0).unsqueeze(-1)
    a = alphas_bar_sqrt[t]
    aml = one_minus_alphas_bar_sqrt[t]
    e = torch.randn_like(x_0)
    x = x_0 * a + e * aml
    output = model(x, t.squeeze(-1), condition1, condition2, kg)
    return ((e - output).square().mean(dim=1) * weight).mean()


# ========== Dataloader ==========
batch_size = 256
dataloader = torch.utils.data.DataLoader(
    list(zip(dataset, conditions[:, :4], conditions[:, 4:], kg, weights)),
    batch_size=batch_size, shuffle=True
)

# ========== Train ==========
model = MLPDiffusion(num_steps)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

for epoch in range(8000):
    for batch_x, batch_c1, batch_c2, batch_kg, batch_w in dataloader:
        loss = diffusion_loss_fn(model, batch_x, batch_c1, batch_c2, batch_kg, batch_w, num_steps)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# ========== Save ==========
torch.save({'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()}, '2W_checkpoint.pth')
print('2W_checkpoint.pth模型已保存')
