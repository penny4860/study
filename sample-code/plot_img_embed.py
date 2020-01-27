# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import umap.umap_ as umap

from matplotlib import offsetbox
from sklearn import datasets


# Scale and visualize the embedding vectors
def plot_embedding(features, labels, original_imgs, title=None):
    """
    # Args
        features : array, shape of (N, 2)
        labels : array, shape of (N,)
        original_imgs: array, shape of (N, H, W, C)
    """
    n_samples = len(features)

    x_min, x_max = np.min(features, 0), np.max(features, 0)
    X = (features - x_min) / (x_max - x_min)

    plt.figure()
    ax = plt.subplot(111)

    # 1. point를 추가
    for i in range(n_samples):
        plt.text(X[i, 0], X[i, 1], str(labels[i]),
                 color=plt.cm.Set1(labels[i] / 10.),
                 fontdict={'weight': 'bold', 'size': 9})

    # 2. image를 추가
    if hasattr(offsetbox, 'AnnotationBbox'):
        # only print thumbnails with matplotlib > 1.0
        shown_images = np.array([[1., 1.]])  # just something big
        for i in range(n_samples):
            dist = np.sum((X[i] - shown_images) ** 2, 1)
            if np.min(dist) < 4e-3:
                # don't show points that are too close
                continue
            shown_images = np.r_[shown_images, [X[i]]]

            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(original_imgs[i], cmap=plt.cm.gray_r), X[i])
            ax.add_artist(imagebox)

    plt.xticks([]), plt.yticks([])
    if title is not None:
        plt.title(title)


if __name__ == '__main__':

    digits = datasets.load_digits(n_class=6)

    images = digits.images      # (N, height, width, ch)
    features = digits.data      # (N, feature_dims)
    labels = digits.target      # (N,)

    reducer = umap.UMAP(random_state=42)
    features_2d = reducer.fit_transform(features)

    plot_embedding(X_tsne, y, images, "embedding of the digits")
    plt.show()
