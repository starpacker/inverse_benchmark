import matplotlib

matplotlib.use("Agg")

from sklearn.decomposition import PCA

from sklearn.mixture import GaussianMixture

def cluster_spikes_gmm(snippets, n_clusters, n_pca, random_seed):
    """PCA + GMM clustering fallback."""
    pca = PCA(n_components=min(n_pca, snippets.shape[1]))
    features = pca.fit_transform(snippets)
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="full",
                          n_init=5, random_state=random_seed)
    labels = gmm.fit_predict(features)
    return labels
