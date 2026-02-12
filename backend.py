import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers




models = (
    "Hybrid (Recommended)",
    "Course Similarity",
    "KNN Collaborative",
    "Clustering with PCA",
    "Neural Network",
)

MODEL_STATE = {}


def load_ratings():
    return pd.read_csv("ratings.csv")


def load_course_sims():
    return pd.read_csv("sim.csv")


def load_courses():
    df = pd.read_csv("course_processed.csv")
    df["TITLE"] = df["TITLE"].str.title()
    return df


def load_bow():
    return pd.read_csv("courses_bows.csv")


def add_new_ratings(new_courses):
    res_dict = {}
    if len(new_courses) > 0:
        ratings_df = load_ratings()
        new_id = ratings_df["user"].max() + 1
        users = [new_id] * len(new_courses)
        ratings = [3.0] * len(new_courses)
        res_dict["user"] = users
        res_dict["item"] = new_courses
        res_dict["rating"] = ratings
        new_df = pd.DataFrame(res_dict)
        updated_ratings = pd.concat([ratings_df, new_df])
        updated_ratings.to_csv("ratings.csv", index=False)
        return new_id


def get_doc_dicts():
    bow_df = load_bow()
    grouped_df = bow_df.groupby(["doc_index", "doc_id"]).max().reset_index(drop=False)
    idx_id_dict = grouped_df[["doc_id"]].to_dict()["doc_id"]
    id_idx_dict = {v: k for k, v in idx_id_dict.items()}
    del grouped_df
    return idx_id_dict, id_idx_dict


def course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix):
    all_courses = set(idx_id_dict.values())
    unselected_course_ids = all_courses.difference(enrolled_course_ids)
    res = {}
    for enrolled_course in enrolled_course_ids:
        for unselect_course in unselected_course_ids:
            if enrolled_course in id_idx_dict and unselect_course in id_idx_dict:
                idx1 = id_idx_dict[enrolled_course]
                idx2 = id_idx_dict[unselect_course]
                sim = sim_matrix[idx1][idx2]
                if unselect_course not in res:
                    res[unselect_course] = sim
                else:
                    if sim >= res[unselect_course]:
                        res[unselect_course] = sim
    res = {k: v for k, v in sorted(res.items(), key=lambda item: item[1], reverse=True)}
    return res


def _train_hybrid():
    ratings_df = load_ratings()
    pop = ratings_df.groupby("item")["rating"].agg(["mean", "count"]).reset_index()
    pop.columns = ["COURSE_ID", "mean_rating", "rating_count"]
    pop["pop_score"] = (pop["mean_rating"] / 3.0) * np.log1p(pop["rating_count"])
    max_pop = pop["pop_score"].max()
    pop["pop_score"] = pop["pop_score"] / max_pop if max_pop else 0.0
    MODEL_STATE["hybrid_popularity"] = pop


def _recommend_hybrid(user_id, params):
    sim_threshold = params.get("sim_threshold", 45) / 100.0
    top_n = int(params.get("top_courses", 10))
    alpha = float(params.get("alpha", 0.7))

    if "hybrid_popularity" not in MODEL_STATE:
        _train_hybrid()

    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    ratings_df = load_ratings()
    user_ratings = ratings_df[ratings_df["user"] == user_id]
    enrolled = user_ratings["item"].to_list()

    sim_scores = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled, sim_matrix)
    pop = MODEL_STATE["hybrid_popularity"].copy()
    pop_map = dict(zip(pop["COURSE_ID"], pop["pop_score"]))

    rows = []
    for cid, sim in sim_scores.items():
        if sim < sim_threshold:
            continue
        p = pop_map.get(cid, 0.0)
        score = (alpha * float(sim)) + ((1 - alpha) * float(p))
        rows.append((cid, score))

    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]


def _train_knn():
    ratings_df = load_ratings()
    pivot = ratings_df.pivot_table(index="item", columns="user", values="rating", fill_value=0.0)
    item_ids = pivot.index.to_list()
    matrix = pivot.values.astype(float)
    norms = np.linalg.norm(matrix, axis=1)
    norms[norms == 0] = 1.0
    normalized = matrix / norms[:, None]
    MODEL_STATE["knn_item_ids"] = item_ids
    MODEL_STATE["knn_matrix"] = normalized


def _recommend_knn(user_id, params):
    top_n = int(params.get("top_courses", 10))
    sim_threshold = params.get("sim_threshold", 35) / 100.0

    if "knn_matrix" not in MODEL_STATE:
        _train_knn()

    ratings_df = load_ratings()
    user_ratings = ratings_df[ratings_df["user"] == user_id]
    enrolled = set(user_ratings["item"].to_list())
    if not enrolled:
        return []

    item_ids = MODEL_STATE["knn_item_ids"]
    mat = MODEL_STATE["knn_matrix"]
    id_to_idx = {cid: idx for idx, cid in enumerate(item_ids)}
    enrolled_idx = [id_to_idx[cid] for cid in enrolled if cid in id_to_idx]
    if not enrolled_idx:
        return []

    profile = mat[enrolled_idx].mean(axis=0)
    scores = mat @ profile

    rows = []
    for idx, score in enumerate(scores):
        cid = item_ids[idx]
        if cid in enrolled:
            continue
        if score >= sim_threshold:
            rows.append((cid, float(score)))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows[:top_n]

def _train_clustering_pca(params):
    rating_df = load_ratings()
    pivot  = rating_df.pivot_table(index="item",columns="user",values="rating",fill_value=0.1)
    item_ids = pivot.index.to_list()
    matrix = pivot.values.astype(float)
    max_components = min(matrix.shape[0], matrix.shape[1])
    n_components = int(params.get("pca_components", 20))
    n_components = max(1, min(n_components, max_components))
    n_clusters = int(params.get("n_clusters", 10))
    n_clusters = max(2, min(n_clusters, len(item_ids)))

    model = PCA(n_components=n_components, random_state=42)
    reduced = model.fit_transform(matrix)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(reduced)

    MODEL_STATE["cpca_item_ids"] = item_ids
    MODEL_STATE["cpca_matrix"] = reduced
    MODEL_STATE["cpca_labels"] = labels
    MODEL_STATE["cpca_kmeans"] = kmeans


def _recommend_clustering_pca(user_id,params):
    if "cpca_matrix" not in MODEL_STATE:
        _train_clustering_pca(params)
    ratings_df = load_ratings()
    user_ratings = ratings_df[ratings_df["user"] == user_id]
    enrolled = set(user_ratings["item"].to_list())
    if not enrolled:
        return []
    item_ids = MODEL_STATE["cpca_item_ids"]
    X =MODEL_STATE["cpca_matrix"]
    labels = MODEL_STATE["cpca_labels"]
    
    id_to_idx = {cid:i for i,cid in enumerate(item_ids)}
    enrolled_idx = [id_to_idx[cid]for cid in enrolled if cid in id_to_idx]
    if not enrolled_idx:
        return []
    user_vec = X[enrolled_idx].mean(axis=0)

    # Kullanıcının baskın cluster'ı
    enrolled_labels = labels[enrolled_idx]
    target_cluster = pd.Series(enrolled_labels).value_counts().idxmax()

    candidate_idx = [i for i, lb in enumerate(labels) if lb == target_cluster and item_ids[i] not in enrolled]
    if not candidate_idx:
        return []

    # cosine similarity
    user_norm = np.linalg.norm(user_vec)
    if user_norm == 0:
        return []

    rows = []
    for i in candidate_idx:
        vec = X[i]
        denom = (np.linalg.norm(vec) * user_norm)
        score = float(np.dot(vec, user_vec) / denom) if denom > 0 else 0.0
        rows.append((item_ids[i], score))

    rows.sort(key=lambda x: x[1], reverse=True)
    top_n = int(params.get("top_courses", 10))
    return rows[:top_n]


def _train_neural_network(params):
    ratings_df = load_ratings().copy()
    user_ids = sorted(ratings_df["user"].unique())
    item_ids = sorted(ratings_df["item"].unique())
    user_to_idx = {u: i for i, u in enumerate(user_ids)}
    item_to_idx = {it: i for i, it in enumerate(item_ids)}
    u = ratings_df["user"].map(user_to_idx).to_numpy(dtype="int32")
    i = ratings_df["item"].map(item_to_idx).to_numpy(dtype="int32")
    y = ratings_df["rating"].to_numpy(dtype="float32")

    n_users = len(user_ids)
    n_items = len(item_ids)
    emb_dim = int(params.get("nn_emb_dim", 32))

    user_in = keras.Input(shape=(1,), name="user")
    item_in = keras.Input(shape=(1,), name="item")

    user_emb = layers.Embedding(n_users, emb_dim)(user_in)
    item_emb = layers.Embedding(n_items, emb_dim)(item_in)

    x = layers.Concatenate()([layers.Flatten()(user_emb), layers.Flatten()(item_emb)])
    x  =layers.Dense(64,activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32,activation = "relu")(x)
    out = layers.Dense(1,activation="linear")(x)

    model = keras.Model(inputs=[user_in, item_in], outputs=out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")

    epochs = int(params.get("nn_epochs", 8))
    batch_size = int(params.get("nn_batch_size", 256))
    model.fit([u, i], y, epochs=epochs, batch_size=batch_size, verbose=0)

    MODEL_STATE["nn_model"] = model
    MODEL_STATE["nn_user_to_idx"] = user_to_idx
    MODEL_STATE["nn_item_to_idx"] = item_to_idx
    MODEL_STATE["nn_item_ids"] = item_ids


def _recommend_neural_network(user_id, params):
    if "nn_model" not in MODEL_STATE:
        _train_neural_network(params)
    model = MODEL_STATE["nn_model"]
    user_to_idx = MODEL_STATE["nn_user_to_idx"]
    item_to_idx = MODEL_STATE["nn_item_to_idx"]
    item_ids = MODEL_STATE["nn_item_ids"]

    ratings_df = load_ratings()
    enrolled = set(ratings_df[ratings_df["user"] == user_id]["item"].to_list())
    if not enrolled:
        return []
    if user_id in user_to_idx:
        uidx = user_to_idx[user_id]
    else:
        uidx = 0

    cand_ids = [cid for cid in item_ids if cid not in enrolled]

    if not cand_ids:
        return []

    u_arr = np.full((len(cand_ids),), uidx, dtype="int32")
    i_arr = np.array([item_to_idx[cid] for cid in cand_ids], dtype="int32")
    preds = model.predict([u_arr, i_arr], verbose=0).reshape(-1)
    rows = [(cid, float(score)) for cid, score in zip(cand_ids, preds)]
    rows.sort(key=lambda x: x[1], reverse=True)

    top_n = int(params.get("top_courses", 10))
    return rows[:top_n]











def train(model_name, params):
    if model_name == "Hybrid (Recommended)":
        _train_hybrid()
    elif model_name == "KNN Collaborative":
        _train_knn()
    elif model_name == "Clustering with PCA":
        _train_clustering_pca(params)
    elif model_name == "Neural Network":
        _train_neural_network(params)



def predict(model_name, user_ids, params):
    sim_threshold = 0.6
    if "sim_threshold" in params:
        sim_threshold = params["sim_threshold"] / 100.0
    idx_id_dict, id_idx_dict = get_doc_dicts()
    sim_matrix = load_course_sims().to_numpy()
    users = []
    courses = []
    scores = []
    res_dict = {}

    for user_id in user_ids:
        if model_name == "Hybrid (Recommended)":
            rows = _recommend_hybrid(user_id, params)
            for cid, score in rows:
                users.append(user_id)
                courses.append(cid)
                scores.append(score)
        elif model_name == "Course Similarity":
            ratings_df = load_ratings()
            user_ratings = ratings_df[ratings_df["user"] == user_id]
            enrolled_course_ids = user_ratings["item"].to_list()
            res = course_similarity_recommendations(idx_id_dict, id_idx_dict, enrolled_course_ids, sim_matrix)
            top_n = int(params.get("top_courses", 10))
            rank = 0
            for key, score in res.items():
                if score >= sim_threshold:
                    users.append(user_id)
                    courses.append(key)
                    scores.append(score)
                    rank += 1
                    if rank >= top_n:
                        break
        elif model_name == "KNN Collaborative":
            rows = _recommend_knn(user_id, params)
            for cid, score in rows:
                users.append(user_id)
                courses.append(cid)
                scores.append(score)
        elif model_name == "Clustering with PCA":
            rows = _recommend_clustering_pca(user_id, params)
            for cid, score in rows:
                users.append(user_id)
                courses.append(cid)
                scores.append(score)
        elif model_name == "Neural Network":
            rows = _recommend_neural_network(user_id, params)
            for cid, score in rows:
                users.append(user_id); courses.append(cid); scores.append(score)



    res_dict["USER"] = users
    res_dict["COURSE_ID"] = courses
    res_dict["SCORE"] = scores
    res_df = pd.DataFrame(res_dict, columns=["USER", "COURSE_ID", "SCORE"])
    return res_df
