import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 定义Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = self.fc2(x)
        return z

# 定义向量量化模块 (Vector Quantizer)
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # 码本的初始化
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        # z的形状: (batch_size, latent_dim)
        z_flattened = z.view(-1, self.embedding_dim)

        # 计算距离并找到最近的码本向量
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                     + torch.sum(self.embeddings.weight**2, dim=1)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t()))

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        z_q = self.embeddings(encoding_indices).view(z.shape)

        # 计算损失: 向量量化损失和承诺损失
        vq_loss = F.mse_loss(z_q.detach(), z)
        commitment_loss = self.commitment_cost * F.mse_loss(z_q, z.detach())

        z_q = z + (z_q - z).detach()  # 停止梯度，量化后的向量替换原始z

        return z_q, vq_loss + commitment_loss, encoding_indices

# 定义Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        z = F.relu(self.fc1(z))
        x_recon = torch.sigmoid(self.fc2(z))
        return x_recon

# 定义VQ-VAE模型
class VQVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, latent_dim, commitment_cost)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss, _ = self.vq_layer(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# 训练流程
def train_vqvae(model, data_loader, num_epochs, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        for batch_data in data_loader:
            batch_data = batch_data.view(batch_data.size(0), -1)  # 展开输入数据
            optimizer.zero_grad()

            # 前向传播
            x_recon, vq_loss = model(batch_data)
            recon_loss = F.mse_loss(x_recon, batch_data)

            # 总损失：重构损失 + 向量量化损失
            loss = recon_loss + vq_loss
            loss.backward()

            optimizer.step()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")

# 使用简单的随机数据进行测试
if __name__ == '__main__':
    # 模拟一些随机输入数据
    input_dim = 784  # 假设输入是28x28图像 (例如MNIST)
    hidden_dim = 256
    latent_dim = 64
    num_embeddings = 512
    commitment_cost = 0.25
    batch_size = 32
    num_epochs = 10

    # 构建VQ-VAE模型
    vqvae_model = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings, commitment_cost)

    # 使用随机数据生成DataLoader
    random_data = torch.randn(1000, 1, 28, 28)  # 生成1000张28x28的随机图像
    data_loader = torch.utils.data.DataLoader(random_data, batch_size=batch_size, shuffle=True)

    # 开始训练
    train_vqvae(vqvae_model, data_loader, num_epochs)
