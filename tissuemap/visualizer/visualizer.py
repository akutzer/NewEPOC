from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import time
from typing import Union, Iterable
import pandas as pd
from scipy.stats import gaussian_kde

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

@timing
def fit_PCA(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = PCA(n_components, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_TSNE(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = TSNE(n_components, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_UMAP(feats: np.ndarray, n_components: int = 2, **kwargs) -> np.ndarray:
    pca = UMAP(n_components=n_components, **kwargs)
    embeddings = pca.fit_transform(feats)
    return embeddings

@timing
def fit_PCA_UMAP(feats: np.ndarray, n_components: int = 2, latent_components: int = 50, **kwargs) -> np.ndarray:
    pca = PCA(latent_components, random_state=125)
    umap = UMAP(n_components=n_components, **kwargs)
    latent_emb = pca.fit_transform(feats)
    embeddings = umap.fit_transform(latent_emb)
    return embeddings


def plot_UMAP_hparams(feats, clss, id2labels, output_dir: Path):
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
    plt.savefig(output_dir / "UMAP_parameters.png", dpi=300)
    # plt.show()


def plot_emb_methods(feats, clss, id2labels, output_dir: Path):
    pca_emb = fit_PCA(feats)
    tsne_emb = fit_TSNE(feats, learning_rate='auto', init='pca', perplexity=30, n_jobs=-1)
    umap_emb = fit_UMAP(feats, n_neighbors=20, min_dist=0.05, metric='euclidean')
    pca_umap_emb = fit_PCA_UMAP(feats, latent_components=50, n_neighbors=20, min_dist=0.05, metric='euclidean')

    fig, axs = plt.subplots(1, 4, figsize=(24, 6))
    # Plot for PCA
    plot_KDE(pca_emb, axs[0])
    scatter_pca = axs[0].scatter(pca_emb[:, 0], pca_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[0].set_title('PCA')

    # Plot for t-SNE
    axs[1].scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[1].set_title('t-SNE')

    # Plot for UMAP
    plot_KDE(umap_emb, axs[2])
    axs[2].scatter(umap_emb[:, 0], umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[2].set_title('UMAP')

    # Plot for PCA + UMAP
    plot_KDE(pca_umap_emb, axs[3])
    axs[3].scatter(pca_umap_emb[:, 0], pca_umap_emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
    axs[3].set_title('PCA+UMAP')


    legend_elements = scatter_pca.legend_elements()[0]
    for el in legend_elements: el.set_alpha(1)
    fig.legend(legend_elements, id2labels, loc='lower center', ncol=len(id2labels), labelspacing=0.)
    fig.supxlabel(" ", horizontalalignment="right", x=0.99, y=0.005, size=10)

    plt.tight_layout()
    plt.savefig(output_dir / "diff_emb_methods.png", dpi=300)
    # plt.show()


def plot_tSNE_hparams(feats, clss, id2labels, output_dir: Path):
    hparams = {
        "perplexity": [3, 10, 30, 100],
        "early_exaggeration": [1, 4, 12, 24]
    }
    default_hparams = {
        "perplexity": 30,
        "early_exaggeration": 12,
        "learning_rate": 'auto',
        "init": 'pca',
        "n_jobs": -1
    }

    fig, axs = plt.subplots(2, 4, figsize=(24, 12))
    for i, (hp_name, hp_vals) in enumerate(hparams.items()):
        for j, hp_val in enumerate(hp_vals):
            hps = default_hparams.copy()
            hps[hp_name] = hp_val
            emb = fit_TSNE(feats, **hps)
            scatter = axs[i, j].scatter(emb[:, 0], emb[:, 1], c=clss[:, 0]["id"], s=2, alpha=0.1, cmap='Spectral')
            axs[i, j].set_title(f"{hp_name}={hp_val}")


    legend_elements = scatter.legend_elements()[0]
    for el in legend_elements: el.set_alpha(1)
    fig.legend(legend_elements, id2labels, loc='outside lower center', ncol=len(id2labels), labelspacing=0.)
    default_hparams_str = " | ".join(f"{hp_name}={hp_val}" for hp_name, hp_val in default_hparams.items())
    fig.supxlabel("Default: " + default_hparams_str, horizontalalignment="right", x=0.99, y=0.005, size=10)

    plt.tight_layout()
    plt.savefig(output_dir / "tSNE_parameters.png", dpi=300)
    # plt.show()
    

def plot_KDE(embeddings, ax):
    xmin = embeddings[:, 0].min()
    xmax = embeddings[:, 0].max()
    ymin = embeddings[:, 1].min()
    ymax = embeddings[:, 1].max()
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    kernel = gaussian_kde(embeddings.T)
    Z = np.reshape(kernel(positions).T, X.shape)
    # axs[2].imshow(np.rot90(Z), cmap=plt.cm.gist_earth, extent=[xmin, xmax, ymin, ymax], alpha=0.3)
    CS = ax.contour(X, Y, Z, alpha=0.3, colors='k')
    # ax.clabel(CS, fontsize=7, inline=True)


def get_cohort_df(
    clini_table: Union[Path, str], slide_table: Union[Path, str], feature_dir: Union[Path, str]
) -> pd.DataFrame:
    clini_df = pd.read_csv(clini_table, dtype=str) if Path(clini_table).suffix == '.csv' else pd.read_excel(clini_table, dtype=str)
    slide_df = pd.read_csv(slide_table, dtype=str) if Path(slide_table).suffix == '.csv' else pd.read_excel(slide_table, dtype=str)

    if 'PATIENT' not in clini_df.columns:
        raise ValueError("The PATIENT column is missing in the clini_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")
    
    if 'PATIENT' not in slide_df.columns:
        raise ValueError("The PATIENT column is missing in the slide_table.\n\
                         Please ensure the patient identifier column is named PATIENT.")
    
    # Avoid FILENAME_x causing merge conflict
    if 'FILENAME' in clini_df.columns and 'FILENAME' in slide_df.columns:
        clini_df = clini_df.drop(columns=['FILENAME'])
    
    df = clini_df.merge(slide_df, on='PATIENT')

    # filter na and infer categories if not given
    # df = df.dropna(subset=target_label)
    # if categories is None or len(categories) == 0:
    #     categories = df[target_label].unique()
    # categories = np.array(categories)

    # # remove uninteresting
    # df = df[df[target_label].isin(categories)]

    # remove slides we don't have
    h5s = set(feature_dir.glob('*.h5'))
    assert h5s, f'no features found in {feature_dir}!'
    h5_df = pd.DataFrame(h5s, columns=['slide_path'])
    h5_df['FILENAME'] = h5_df.slide_path.map(lambda p: p.stem)
    df = df.merge(h5_df, on='FILENAME')

    # reduce to one row per patient with list of slides in `df['slide_path']`
    patient_df = df.groupby('PATIENT').first().drop(columns='slide_path')
    patient_slides = df.groupby('PATIENT').slide_path.apply(list)
    df = patient_df.merge(patient_slides, left_on='PATIENT', right_index=True).reset_index()

    return df


def reduce_to_binary(labels, id2labels):
    pos_indices = np.where((id2labels == "STR") | (id2labels == "TUM"))[0]
    mask = (labels["id"][..., None] == pos_indices).sum(axis=-1).astype(np.bool_)
    pos_prob = labels["probability"][mask].reshape(-1, pos_indices.shape[0])
    pos_prob = pos_prob.sum(axis=-1)

    id2labels = ["NORM", "TUM"]
    
    labels = np.zeros((labels.shape[0], 2) , labels.dtype)
    labels["id"][:, 1] = 1
    labels["probability"][:, 0] = 1 - pos_prob
    labels["probability"][:, 1] = pos_prob
    labels[..., ::-1].sort(order='probability')

    return labels, id2labels



def visualize(output_dir: Path, feature_dir: Path, clini_table: Path, slide_table: Path, method: str):
    df = get_cohort_df(clini_table, slide_table, feature_dir)
    print(df)

    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_excel(output_dir / "test.xlsx")

    feature_paths = sum(df["slide_path"], [])
    # feature_paths = list(feature_dir.glob(f"**/*.h5"))
    # print(feature_paths)

    feats, clss = [], []
    for file in feature_paths:
        with h5py.File(file, 'r') as f:
            id2labels = np.array(f["id2class"], dtype=str)
            labels = np.array(f["classes"])

            labels, id2labels = reduce_to_binary(labels, id2labels)
            clss.append(labels)
            feats.append(np.array(f["feats"]))
            
            

    feats = np.concatenate(feats)
    clss = np.concatenate(clss)
    print(id2labels)

    plot_emb_methods(feats, clss, id2labels, output_dir)

    # plot_tSNE_hparams(feats, clss, id2labels, output_dir)

    # plot_UMAP_hparams(feats, clss, id2labels, output_dir)