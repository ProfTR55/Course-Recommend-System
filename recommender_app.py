import time
import numpy as np
import pandas as pd
import streamlit as st

import backend as backend
from st_aggrid import AgGrid
from st_aggrid import DataReturnMode
from st_aggrid.grid_options_builder import GridOptionsBuilder


st.set_page_config(
    page_title="Course Recommender Studio",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
        background:
            radial-gradient(circle at 12% 8%, #dbeafe 0%, transparent 34%),
            radial-gradient(circle at 88% 0%, #fde68a 0%, transparent 30%),
            linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
    }
    .hero {
        padding: 1.25rem 1.5rem;
        border-radius: 20px;
        background: linear-gradient(120deg, #0f172a 0%, #1d4ed8 55%, #0ea5e9 100%);
        color: #ffffff;
        box-shadow: 0 16px 42px rgba(15, 23, 42, 0.28);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-weight: 800;
        letter-spacing: -0.02em;
        font-size: 1.8rem;
    }
    .hero p {
        margin: 0.35rem 0 0;
        opacity: 0.95;
        font-size: 0.95rem;
    }
    .stat {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 0.8rem 1rem;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    }
    .stat .k { color: #475569; font-size: 0.78rem; margin-bottom: 0.25rem; }
    .stat .v { color: #0f172a; font-weight: 800; font-size: 1.15rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_ratings():
    return backend.load_ratings()


@st.cache_data
def load_courses():
    return backend.load_courses()


MODEL_DESCRIPTIONS = {
    "Hybrid (Recommended)": "Blends content similarity with popularity. Good default when you want stable and practical recommendations.",
    "Course Similarity": "Pure item-item similarity from course content vectors. Works well for focused, topic-near suggestions.",
    "KNN Collaborative": "Collaborative filtering over user-item patterns. Recommends courses liked by users with similar behavior.",
    "Clustering with PCA": "Reduces item vectors with PCA, clusters items, then ranks candidates by cosine similarity to user profile.",
    "Neural Network": "Embedding-based neural recommender (Keras). Learns user-item interaction patterns from ratings.",
}


def build_eval_params(model_name, top_k):
    params = {"top_courses": int(top_k)}
    if model_name in ("Hybrid (Recommended)", "Course Similarity", "KNN Collaborative"):
        params["sim_threshold"] = 0
    if model_name == "Hybrid (Recommended)":
        params["alpha"] = 0.7
    if model_name == "Clustering with PCA":
        params["pca_components"] = 20
        params["n_clusters"] = 10
    if model_name == "Neural Network":
        params["nn_emb_dim"] = 16
        params["nn_epochs"] = 4
        params["nn_batch_size"] = 256
    return params


def evaluate_models(model_names, top_k=10, max_users=50):
    ratings_df = backend.load_ratings().copy()
    grouped = ratings_df.groupby("user")["item"].apply(list)
    rng = np.random.default_rng(42)

    holdout_item = {}
    valid_users = []
    for user_id, items in grouped.items():
        unique_items = list(dict.fromkeys(items))
        if len(unique_items) < 3:
            continue
        holdout = unique_items[int(rng.integers(0, len(unique_items)))]
        holdout_item[user_id] = holdout
        valid_users.append(user_id)

    if not valid_users:
        return pd.DataFrame()

    if len(valid_users) > max_users:
        valid_users = list(rng.choice(valid_users, size=max_users, replace=False))

    train_df = ratings_df.copy()
    drop_rows = []
    for user_id in valid_users:
        hid = holdout_item[user_id]
        idx = train_df[(train_df["user"] == user_id) & (train_df["item"] == hid)].index
        if len(idx) > 0:
            drop_rows.append(idx[0])
    train_df = train_df.drop(index=drop_rows)

    original_load_ratings = backend.load_ratings
    original_state = dict(backend.MODEL_STATE)
    rows = []

    try:
        backend.load_ratings = lambda: train_df.copy()
        for model_name in model_names:
            backend.MODEL_STATE.clear()
            params = build_eval_params(model_name, top_k)
            try:
                backend.train(model_name, params)
            except Exception:
                pass

            hits = 0
            users_with_recs = 0
            recommended = set()
            for user_id in valid_users:
                try:
                    rec_df = backend.predict(model_name, [user_id], params)
                except Exception:
                    rec_df = pd.DataFrame(columns=["USER", "COURSE_ID", "SCORE"])
                if rec_df.empty:
                    continue
                rec_ids = rec_df["COURSE_ID"].head(top_k).tolist()
                if not rec_ids:
                    continue
                users_with_recs += 1
                recommended.update(rec_ids)
                if holdout_item[user_id] in rec_ids:
                    hits += 1

            total_users = max(len(valid_users), 1)
            coverage_base = max(train_df["item"].nunique(), 1)
            rows.append(
                {
                    "MODEL": model_name,
                    "HIT@K": round(hits / total_users, 4),
                    "COVERAGE": round(len(recommended) / coverage_base, 4),
                    "USERS_WITH_RECS": users_with_recs,
                    "EVAL_USERS": len(valid_users),
                }
            )
    finally:
        backend.load_ratings = original_load_ratings
        backend.MODEL_STATE.clear()
        backend.MODEL_STATE.update(original_state)

    return pd.DataFrame(rows)


def render_header():
    ratings_df = load_ratings()
    courses_df = load_courses()
    total_ratings = len(ratings_df)
    total_users = ratings_df["user"].nunique()
    total_courses = courses_df["COURSE_ID"].nunique()

    st.markdown(
        """
        <div class="hero">
          <h1>Course Recommender Studio</h1>
          <p>Portfolio-ready recommendation app with hybrid ranking and interactive course discovery.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.markdown(
        f'<div class="stat"><div class="k">Course Catalog</div><div class="v">{total_courses}</div></div>',
        unsafe_allow_html=True,
    )
    c2.markdown(
        f'<div class="stat"><div class="k">Users</div><div class="v">{total_users}</div></div>',
        unsafe_allow_html=True,
    )
    c3.markdown(
        f'<div class="stat"><div class="k">Interactions</div><div class="v">{total_ratings}</div></div>',
        unsafe_allow_html=True,
    )


def select_courses():
    with st.spinner("Loading datasets..."):
        course_df = load_courses()

    st.markdown("---")
    st.subheader("Select courses you already completed")

    gb = GridOptionsBuilder.from_dataframe(course_df[["COURSE_ID", "TITLE", "DESCRIPTION"]])
    gb.configure_default_column(
        editable=False,
        filter=True,
        sortable=True,
        resizable=True,
    )
    gb.configure_selection(selection_mode="multiple", use_checkbox=True)
    grid_options = gb.build()

    response = AgGrid(
        course_df[["COURSE_ID", "TITLE", "DESCRIPTION"]],
        gridOptions=grid_options,
        data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
        fit_columns_on_grid_load=False,
        update_on=["selectionChanged", "modelUpdated"],
        height=380,
    )

    selected_rows = response.get("selected_rows", [])
    if selected_rows is None:
        has_selection = False
    elif isinstance(selected_rows, pd.DataFrame):
        has_selection = not selected_rows.empty
    else:
        has_selection = len(selected_rows) > 0

    if has_selection:
        results = pd.DataFrame(selected_rows)
        results = results[["COURSE_ID", "TITLE"]]
    else:
        results = pd.DataFrame(columns=["COURSE_ID", "TITLE"])

    st.caption(f"Selected: {len(results)} course(s)")
    st.dataframe(results, use_container_width=True, hide_index=True)
    return results


def train(model_name, params):
    with st.spinner("Training model..."):
        time.sleep(0.3)
        backend.train(model_name, params)
    st.success("Model trained.")


def predict(model_name, user_ids, params):
    with st.spinner("Generating recommendations..."):
        time.sleep(0.3)
        return backend.predict(model_name, user_ids, params)


render_header()

st.sidebar.title("Recommendation Controls")
st.sidebar.subheader("1) Model")
model_selection = st.sidebar.selectbox("Choose model", backend.models, index=0)
st.sidebar.info(MODEL_DESCRIPTIONS.get(model_selection, "No description available for this model."))

params = {}
st.sidebar.subheader("2) Hyperparameters")

top_courses = st.sidebar.slider("Top recommendations", min_value=1, max_value=30, value=10, step=1)
params["top_courses"] = top_courses

if model_selection in ("Hybrid (Recommended)", "Course Similarity", "KNN Collaborative"):
    sim_threshold = st.sidebar.slider(
        "Similarity threshold (%)",
        min_value=0,
        max_value=100,
        value=45,
        step=5,
    )
    params["sim_threshold"] = sim_threshold

if model_selection == "Hybrid (Recommended)":
    alpha = st.sidebar.slider(
        "Content weight (alpha)",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
    )
    params["alpha"] = alpha

selected_courses_df = select_courses()

st.sidebar.subheader("3) Training")
if st.sidebar.button("Train Selected Model", use_container_width=True):
    train(model_selection, params)

st.sidebar.subheader("4) Recommendation")
if st.sidebar.button("Recommend New Courses", use_container_width=True):
    if selected_courses_df.empty:
        st.warning("Select at least one completed course first.")
    else:
        new_id = backend.add_new_ratings(selected_courses_df["COURSE_ID"].values)
        res_df = predict(model_selection, [new_id], params)
        if res_df.empty:
            st.info("No recommendations for current thresholds. Lower threshold and retry.")
        else:
            merged = pd.merge(
                res_df[["COURSE_ID", "SCORE"]],
                load_courses(),
                on="COURSE_ID",
                how="left",
            )[["TITLE", "DESCRIPTION", "SCORE"]]
            merged["SCORE"] = merged["SCORE"].round(4)
            st.markdown("---")
            st.subheader("Recommended Courses")
            st.dataframe(merged, use_container_width=True, hide_index=True)

st.markdown("---")
st.subheader("Model Evaluation")
st.caption("Compares implemented models with simple offline holdout metrics (Hit@K and Coverage).")

eval_top_k = st.slider("Evaluation K", min_value=3, max_value=20, value=10, step=1)
eval_users = st.slider("Evaluation users", min_value=10, max_value=200, value=50, step=10)
eval_models = [
    m
    for m in (
        "Hybrid (Recommended)",
        "Course Similarity",
        "KNN Collaborative",
        "Clustering with PCA",
        "Neural Network",
    )
    if m in backend.models
]

if st.button("Run Model Comparison", use_container_width=True):
    with st.spinner("Running evaluation..."):
        eval_df = evaluate_models(eval_models, top_k=eval_top_k, max_users=eval_users)
    if eval_df.empty:
        st.warning("Not enough data to evaluate models. Need users with at least 3 interactions.")
    else:
        eval_df = eval_df.sort_values("HIT@K", ascending=False).reset_index(drop=True)
        st.dataframe(eval_df, use_container_width=True, hide_index=True)
        st.bar_chart(eval_df.set_index("MODEL")[["HIT@K", "COVERAGE"]], use_container_width=True)
