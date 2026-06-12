from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np
import torch


def visualize_embeddings(model, dataloader):
    model.eval()
    feats = []

    with torch.no_grad():
        for x1, _ in dataloader:
            x1 = x1.cuda()
            f = model.backbone(x1)
            feats.append(f.cpu().numpy())

    feats = np.concatenate(feats, axis=0)

    emb = TSNE(n_components=2).fit_transform(feats)

    plt.scatter(emb[:,0], emb[:,1], s=2)
    plt.title("DINO SSL Embeddings")
    plt.show()