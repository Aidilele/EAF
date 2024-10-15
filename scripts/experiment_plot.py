import torch
from model.condition_model import ConditionModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import numpy as np
from sklearn.decomposition import PCA
from utils.builder import build_condition_model, build_config


def traj_emb_t_sne(path=1999):
    config = build_config()
    model = build_condition_model(config)
    model.load(path)
    # traj_data
    X = model.dataset.data['obs_data']
    X_mask = model.dataset.data['traj_mask']
    # generate traj representation
    X_patch = []
    batch_size = 32
    for i in range(X.shape[0] // batch_size):
        X_patch.append(
            model.trajectory_embedding(
                X[i * batch_size:(i + 1) * batch_size].to(model.device),
                X_mask[i * batch_size:(i + 1) * batch_size].to(model.device)
            )[0].detach().cpu().numpy())
    X_patch.append(
        model.trajectory_embedding(
            X[(i + 1) * batch_size:].to(model.device),
            X_mask[(i + 1) * batch_size:].to(model.device)
        )[0].detach().cpu().numpy())
    X = np.concatenate(X_patch, axis=0)
    # traj_reawrd
    Y = model.dataset.data['traj_aver'].numpy()

    # generate preference vector
    preference_num = 10
    preference = torch.ones((preference_num, 1), device=model.device)
    for i in range(preference_num):
        preference[i, 0] = (i + 1) / preference_num

    # generate traj representation base on given preference vector
    best_X = model.task_embedding(preference)[0].detach().cpu().numpy()

    # np.random.seed(42)
    X = np.concatenate((X, best_X), 0)
    tsne = TSNE(n_components=2, random_state=42)
    # pca = PCA(n_components=2)
    # X_embedded = pca.fit_transform(X)
    X_embedded = tsne.fit_transform(X)

    plt.figure(figsize=(8, 6))
    draw_start = 0
    draw_length = 1800
    scatter = plt.scatter(X_embedded[draw_start:draw_start + draw_length, 0],
                          X_embedded[draw_start:draw_start + draw_length, 1],
                          c=Y[draw_start:draw_start + draw_length],
                          cmap='viridis',
                          s=2)

    scatter = plt.scatter(X_embedded[-preference_num:, 0], X_embedded[-preference_num:, 1], color='red', s=20)
    plt.colorbar(scatter, label='Class Label')
    plt.title('t-SNE Visualization of High Dimensional Data')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')

    plt.show()


if __name__ == '__main__':
    traj_emb_t_sne(1999)
