"""
cluster.py  —  Unsupervised ML (TUTOR tier)
----------------------------------------------
K-Means clustering to group provinces by their STEM performance profile.
Uses StandardScaler + silhouette analysis to choose the optimal k, and
PCA to project clusters into 2D for visualisation.

Contributor: SKYLearn-Innovation Tutor (university undergraduate)
Reviewed by: Dingaan Mahlatse Machethe (SKYLearn-Innovation head)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

RANDOM_STATE = 42


def build_province_profile(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a per-province feature profile for clustering.
    Features: avg STEM pass rate, avg non-STEM pass rate, avg absentee rate,
    avg distinction rate, and pass-rate volatility (std over years).
    """
    stem = (
        df[df["subject_type"] == "STEM"]
        .groupby("province")["pass_rate"].mean()
        .rename("stem_pass_rate")
    )
    nonstem = (
        df[df["subject_type"] == "Non-STEM"]
        .groupby("province")["pass_rate"].mean()
        .rename("nonstem_pass_rate")
    )
    absentee = df.groupby("province")["absentee_rate"].mean().rename("absentee_rate")
    distinction = df.groupby("province")["distinction_rate"].mean().rename("distinction_rate")
    volatility = df.groupby("province")["pass_rate"].std().rename("pass_rate_volatility")

    profile = pd.concat([stem, nonstem, absentee, distinction, volatility], axis=1)
    return profile.reset_index()


def find_optimal_k(profile: pd.DataFrame, k_range: range = range(2, 7)) -> pd.DataFrame:
    """
    Run silhouette analysis across candidate k values.
    Returns a DataFrame with [k, silhouette, inertia].
    """
    feature_cols = [c for c in profile.columns if c != "province"]
    X = StandardScaler().fit_transform(profile[feature_cols])

    rows = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        rows.append({
            "k": k,
            "silhouette": round(silhouette_score(X, labels), 4),
            "inertia": round(km.inertia_, 2),
        })
    return pd.DataFrame(rows)


def cluster_provinces(profile: pd.DataFrame, k: int = 3) -> dict:
    """
    Fit K-Means with the chosen k and project to 2D via PCA.
    Returns the labelled profile, cluster centroids, PCA coords, and the
    silhouette score.
    """
    feature_cols = [c for c in profile.columns if c != "province"]
    scaler = StandardScaler()
    X = scaler.fit_transform(profile[feature_cols])

    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    coords = pca.fit_transform(X)

    labelled = profile.copy()
    labelled["cluster"] = labels
    labelled["pca_x"] = coords[:, 0]
    labelled["pca_y"] = coords[:, 1]

    labelled = _name_clusters(labelled)

    centroids = pd.DataFrame(
        scaler.inverse_transform(km.cluster_centers_), columns=feature_cols
    )
    centroids["cluster"] = range(k)

    return {
        "labelled": labelled,
        "centroids": centroids,
        "silhouette": round(silhouette_score(X, labels), 4),
        "explained_variance": pca.explained_variance_ratio_.tolist(),
        "k": k,
    }


def _name_clusters(labelled: pd.DataFrame) -> pd.DataFrame:
    """
    Assign human-readable tier names to clusters based on mean STEM pass rate.
    Highest = 'High Performing', lowest = 'Needs Support'.
    """
    order = (
        labelled.groupby("cluster")["stem_pass_rate"].mean()
        .sort_values(ascending=False).index.tolist()
    )
    n = len(order)
    if n == 3:
        names = ["High Performing", "Mid Performing", "Needs Support"]
    elif n == 2:
        names = ["Stronger", "Needs Support"]
    else:
        names = [f"Tier {i+1}" for i in range(n)]

    mapping = {cluster: names[i] for i, cluster in enumerate(order)}
    labelled["cluster_name"] = labelled["cluster"].map(mapping)
    return labelled


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from preprocess import load_clean

    df = load_clean()
    profile = build_province_profile(df)
    print("Province profiles:")
    print(profile.round(2).to_string(index=False))

    print("\nSilhouette analysis:")
    print(find_optimal_k(profile).to_string(index=False))

    result = cluster_provinces(profile, k=3)
    print(f"\nClusters (silhouette={result['silhouette']}):")
    print(result["labelled"][["province", "cluster_name", "stem_pass_rate"]].to_string(index=False))
