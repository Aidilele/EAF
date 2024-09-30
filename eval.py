import metaworld
import random
from model.condition_model import ConditionModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
import numpy as np

model = ConditionModel(0)
model.load(0)
X = model.trajectory_embedding(model.dataset.data_min)




# 使用TSNE将高维数据降维到二维
tsne = TSNE(n_components=2, random_state=42)
X_embedded = tsne.fit_transform(X)
print('ok')
# 绘制结果
plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], cmap='viridis', s=100)
plt.title('t-SNE Visualization of High Dimensional Data')
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.colorbar(scatter, label='Class Label')
plt.show()
