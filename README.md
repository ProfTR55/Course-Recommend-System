# Course Recommend System

An interactive Streamlit-based course recommendation app.  
Users can select completed courses, train different recommenders, and compare model behavior inside the UI.

## Features

- Modern Streamlit interface with interactive course selection (`streamlit-aggrid`)
- Multiple recommendation models:
  - `Hybrid (Recommended)`
  - `Course Similarity`
  - `KNN Collaborative`
  - `Clustering with PCA`
  - `Neural Network` (Keras)
- Sidebar controls for model and hyperparameters
- In-app model descriptions
- Built-in offline comparison metrics:
  - `Hit@K`
  - `Coverage`

## Project Structure

```text
Course Recommend System/
├─ recommender_app.py      # Streamlit UI
├─ backend.py              # Model training/prediction logic
├─ ratings.csv             # User-item ratings
├─ sim.csv                 # Course similarity matrix
├─ course_processed.csv    # Processed course metadata
├─ courses_bows.csv        # Bag-of-words/content features
└─ requirements.txt
```

## Installation

### 1) Create a virtual environment

```bash
python -m venv .venv
```

Windows (PowerShell):
```bash
.venv\Scripts\Activate.ps1
```

macOS/Linux:
```bash
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run recommender_app.py
```

## Usage Flow

1. Select a model from the sidebar.
2. Adjust model hyperparameters.
3. Select completed courses in the main table.
4. Click `Train Selected Model`.
5. Click `Recommend New Courses`.
6. Use `Model Evaluation` to compare models with `Hit@K` and `Coverage`.

## Model Notes

- `Hybrid`: combines content similarity and popularity.
- `Course Similarity`: pure item-item similarity approach.
- `KNN Collaborative`: collaborative filtering from user-item interactions.
- `Clustering with PCA`: dimensionality reduction (PCA) + KMeans clustering.
- `Neural Network`: Keras embedding-based rating predictor.

Note: `SCORE` scales are model-specific.  
Compare scores **within the same model**, not across different models.

## Requirements

- Python 3.10+
- TensorFlow (CPU mode works)
- scikit-learn
- Streamlit

## Future Improvements

- Add missing models (`User Profile`, `NMF`, embedding-based regression/classification)
- Add more metrics (`Recall@K`, `MRR@K`, `MAP@K`)
- Add hyperparameter tuning and experiment tracking

## License

If there is no license file in the repository, contact the project owner before reuse.
