import os
import re
import ast
import string
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# Suppress MLflow artifact download warnings
# os.environ["MLFLOW_DISABLE_ARTIFACTS_DOWNLOAD"] = "1"

# Set MLflow Tracking URI & DAGsHub integration
MLFLOW_TRACKING_URI = "https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow"
dagshub.init(repo_owner="Razakhan143", repo_name="PERFUME_HAVEN_MLOPS_PROJECT", mlflow=True)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("sample trial")


# ==========================
# Text Preprocessing Functions
# ==========================
def collapse(L):
    try:
        L1 = []
        for i in ast.literal_eval(L):
            L1.append(i.replace(" ", ""))
        return L1
    except Exception as e:
        print(f"Error in collapse: {e}")
        return []

def normalize_text(perfumes):
    try:
        perfumes = perfumes.copy()
        perfumes['notes'] = perfumes['notes'].apply(collapse)
        perfumes['description'] = perfumes['description'].apply(lambda x: x.split() if isinstance(x, str) else [])
        perfumes['designer'] = perfumes['designer'].apply(lambda x: x.split() if isinstance(x, str) else [])
        perfumes['tags'] = perfumes['notes'] + perfumes['description'] + perfumes['designer']
        perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join([str(i) for i in x if i])).str.lower()
        perfumes['notes'] = perfumes['notes'].apply(
            lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else str(x).capitalize()
        )
        perfumes.reset_index(drop=True, inplace=True)
        return perfumes
    except Exception as e:
        print(f"Error during text normalization: {e}")
        raise

# ==========================
# Load & Prepare Data
# ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path).iloc[:1000]
        print(f"Loaded dataset size: {df.shape}")
        new_perfume = normalize_text(df)
        print(f"Normalized dataset size: {new_perfume.shape}")
        return new_perfume, sparse_cosine_similarity(new_perfume)
    except Exception as e:
        print(f"Error loading data: {e}")
        raise
    
    


# ==========================
# Train & Log Model
# ==========================


def sparse_cosine_similarity(new_perfume):
    try:
        vectorizer = TfidfVectorizer(stop_words = 'english')
        tfidf_matrix = vectorizer.fit_transform(new_perfume['tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        with mlflow.start_run():
            mlflow.log_param("vectorizer", "TfidfVectorizer")
            mlflow.sklearn.log_model(vectorizer, "vectorizer")
            mlflow.log_metric("cosine_similarity_shape", cosine_sim.shape[0])
            print(f"Cosine similarity matrix shape: {cosine_sim.shape}")
        return cosine_sim
    except Exception as e:
        print(f"Error in sparse_cosine_similarity: {e}")
        raise

def recommend_perfumes(cosine_sim, new_perfume, selected_perfumes, n=10):
    selected_indices = new_perfume[new_perfume['title'].isin(selected_perfumes)].index
    sim_scores = cosine_sim[selected_indices].mean(axis=0)
    similar_indices = sim_scores.argsort()[::-1]
    recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]

    return new_perfume.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]


# ==========================
# Main Execution
# ==========================
if __name__ == "__main__":
    df , cos = load_data("notebooks/perfumes_dataset.csv")
    selected_perfumes = df.sample(n = 1, random_state = None)['title']
    print('Selected Perfumes: ' + ', '.join(selected_perfumes) + '\n')
    recomand=recommend_perfumes(cos, df, selected_perfumes, n=10)
    print("Recommended Perfumes:")
    print(recomand[['title', 'designer', 'description', 'notes', 'img_url']].to_string(index=False))
    print("\n\n")
    # Log the recommended perfumes
    with mlflow.start_run():
        mlflow.log_param("selected_perfumes", selected_perfumes.tolist())
        mlflow.log_param("num_recommendations", 10)
        mlflow.log_metric("num_recommended", len(recomand))
        mlflow.log_artifact("notebooks/perfumes_dataset.csv", artifact_path="data")
        mlflow.log_artifact("notebooks/exp3_lor_bow_hp.py", artifact_path="scripts")
        mlflow.log_dict(recomand.to_dict(orient='records'), "recommended_perfumes.json")
    print("MLflow run completed and artifacts logged.")
 
