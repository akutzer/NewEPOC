from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import time


from functools import wraps
from time import time

def timing(f, verbose=False):
    @wraps(f)
    def wrap(*args, **kw):
        ts = time()
        result = f(*args, **kw)
        te = time()
        if verbose:
            args = [arg if isinstance(args, np.ndarray) else f"array({arg.shape}, {arg.dtype})" for arg in args]
            print(f"func:{f.__name__} args:[{args}, {kw}] took: {te-ts:2.4f} sec")
        else:
            print(f"func:{f.__name__} took: {te-ts:2.4f} sec")
        return result
    return wrap


feature_dir = Path("/home/aaron/work/EKFZ/data/NewEPOC/features/features/STAMP_macenko_xiyuewang-ctranspath-7c998680")

feats, clss = [], []
for file in feature_dir.iterdir():
    if file.suffix == ".h5":
        with h5py.File(file, 'r') as f:
            feats.append(np.array(f["feats"]))
            clss.append(np.array(f["classes"]))

feats = np.concatenate(feats)[::10]
clss = np.concatenate(clss)[::10]
print(feats.shape, clss.shape)




id2labels = ['ADI', 'BACK', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']


# PCA
@timing
def fit_PCA(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = PCA(n_components, random_state=125, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_TSNE(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = TSNE(n_components, random_state=125, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_UMAP(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = UMAP(n_components=n_components, random_state=125, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_PCA_UMAP(feats: np.ndarray, n_components: int = 2, latent_components: int = 50, **kwargs) -> np.ndarray:
    pca = PCA(latent_components, random_state=125)
    umap = UMAP(n_components=n_components, random_state=125, **kwargs)
    latent_emb = pca.fit_transform(feats)
    embeddings = umap.fit_transform(latent_emb)
    return embeddings



def plot_UMAP_hparams(feats, clss, id2labels):
    hparams = {
        "latent_components": [16, 32, 64, 128],
        "n_neighbors": [3, 10, 20, 50],
        "min_dist": [0.0, 0.03, 0.1, 0.5],
        "metric": ["manhattan", "euclidean", "cosine", "mahalanobis"],
    }
    default_hparams = {
        "latent_components": 32,
        "n_neighbors": 10,
        "min_dist": 0.1,
        "metric": "euclidean"
    }
    
    fig, axs = plt.subplots(4, 4, figsize=(21, 21))
    for i, (hp_name, hp_vals) in enumerate(hparams.items()):
        for j, hp_val in enumerate(hp_vals):
            hps = default_hparams.copy()
            hps[hp_name] = hp_val
            emb = fit_PCA_UMAP(feats, **hps)
            scatter = axs[i, j].scatter(emb[:, 0], emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
            axs[i, j].set_title(f"{hp_name}={hp_val}")
            
    legend_elements = scatter.legend_elements()[0]
    for el in legend_elements: el.set_alpha(1)
    fig.legend(legend_elements, id2labels, loc='outside lower center', ncol=len(id2labels), labelspacing=0.)
    default_hparams_str = " | ".join(f"{hp_name}={hp_val}" for hp_name, hp_val in default_hparams.items())
    fig.supxlabel("Default: " + default_hparams_str, horizontalalignment="right", x=0.99, y=0.005, size=10)

    plt.tight_layout()
    plt.savefig("UMAP_parameters.png", dpi=300)
    plt.show()


def plot_emb_methods(feats, clss, id2labels):
    pca_emb = fit_PCA(feats)
    tsne_emb = fit_TSNE(feats, learning_rate='auto', init='pca', perplexity=30, n_jobs=-1)
    umap_emb = fit_UMAP(feats, n_neighbors=20, min_dist=0.05, metric='euclidean')
    pca_umap_emb = fit_PCA_UMAP(feats, latent_components=50, n_neighbors=20, min_dist=0.05, metric='euclidean')

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    # Plot for PCA
    scatter_pca = axs[0].scatter(pca_emb[:, 0], pca_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[0].set_title('PCA')

    # Plot for t-SNE
    scatter_tsne = axs[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[1].set_title('t-SNE')

    # Plot for UMAP
    scatter_umap = axs[2].scatter(umap_emb[:, 0], umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[2].set_title('UMAP')

    # Plot for UMAP
    scatter_umap = axs[3].scatter(pca_umap_emb[:, 0], pca_umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[3].set_title('PCA+UMAP')

    legend_elements = scatter_pca.legend_elements()[0]
    for el in legend_elements: el.set_alpha(1)
    fig.legend(legend_elements, id2labels, loc='lower center', ncol=len(id2labels), labelspacing=0.)
    fig.supxlabel(" ", horizontalalignment="right", x=0.99, y=0.005, size=10)

    plt.tight_layout()
    plt.savefig("diff_emb_methods.png", dpi=300)
    plt.show()


plot_emb_methods(feats, clss, id2labels)
# plot_UMAP_hparams(feats, clss, id2labels)

# # pca_emb = fit_PCA(feats)
# # tsne_emb = fit_TSNE(feats, learning_rate='auto', init='pca', perplexity=30, n_jobs=-1)
# # umap_emb = fit_UMAP(feats, n_neighbors=20, min_dist=0.05, metric='euclidean')
# # pca_umap_emb = fit_PCA_UMAP(feats, latent_components=50, n_neighbors=20, min_dist=0.05, metric='euclidean')

# pca_emb = fit_PCA_UMAP(feats, latent_components=16, n_neighbors=20, min_dist=0.05, metric='euclidean')
# tsne_emb = fit_PCA_UMAP(feats, latent_components=32, n_neighbors=20, min_dist=0.05, metric='euclidean')
# umap_emb = fit_PCA_UMAP(feats, latent_components=64, n_neighbors=20, min_dist=0.05, metric='euclidean')
# pca_umap_emb = fit_PCA_UMAP(feats, latent_components=128, n_neighbors=20, min_dist=0.05, metric='euclidean')

# # Plotting
# fig, axs = plt.subplots(4, 4, figsize=(24, 6))

# # Plot for PCA
# scatter_pca = axs[0].scatter(pca_emb[:, 0], pca_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
# legend_elements = scatter_pca.legend_elements()[0]
# for el in legend_elements: el.set_alpha(1)
# legend_pca = axs[0].legend(legend_elements, id2labels, title="Classes")
# axs[0].set_title('PCA')

# # Plot for t-SNE
# scatter_tsne = axs[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
# legend_tsne = axs[1].legend(legend_elements, id2labels, title="Classes")
# axs[1].set_title('t-SNE')

# # Plot for UMAP
# scatter_umap = axs[2].scatter(umap_emb[:, 0], umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
# legend_umap = axs[2].legend(legend_elements, id2labels, title="Classes")
# axs[2].set_title('UMAP')

# # Plot for UMAP
# scatter_umap = axs[3].scatter(pca_umap_emb[:, 0], pca_umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
# legend_umap = axs[3].legend(legend_elements, id2labels, title="Classes")
# axs[3].set_title('PCA+UMAP')

# plt.show()
