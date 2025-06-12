import setuptools
import os
import re
import string
import pandas as pd
pd.set_option('future.no_silent_downcasting', True)
import ast
import numpy as np
import mlflow
import mlflow.sklearn
import dagshub
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
import warnings
warnings.simplefilter("ignore", UserWarning)
warnings.filterwarnings("ignore")

# ========================== CONFIGURATION ==========================
CONFIG = {
    "data_path": "notebooks/perfumes_dataset.csv",  # Update to local/cloud path after Git removal
    "data_size": 45000,
    "mlflow_tracking_uri": "https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow",
    "dagshub_repo_owner": "Razakhan143",
    "dagshub_repo_name": "PERFUME_HAVEN_MLOPS_PROJECT",
    "experiment_name": "Bow vs TfIdf"
}

# ========================== SETUP MLflow & DAGSHUB ==========================
mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
mlflow.set_experiment(CONFIG["experiment_name"])

# ========================== TEXT PREPROCESSING ==========================
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

# ========================== LOAD & PREPROCESS DATA ==========================
def load_data(file_path):
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded dataset size: {df.shape}")
        new_perfume = normalize_text(df)
        print(f"Normalized dataset size: {new_perfume.shape}")
        return new_perfume
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

# ========================== FEATURE ENGINEERING ==========================
VECTORIZERS = {
    'BoW': CountVectorizer(stop_words='english', max_features=1000),  # Reduced max_features
    'TF-IDF': TfidfVectorizer(stop_words='english', max_features=1000)
}

# ========================== SPARSE COSINE SIMILARITY ==========================
def sparse_cosine_similarity(matrix):
    try:
        norm_matrix = normalize(matrix, norm='l2', axis=1)
        return norm_matrix.dot(norm_matrix.T)
    except Exception as e:
        print(f"Error in sparse_cosine_similarity: {e}")
        raise

# ========================== TRAIN & EVALUATE ==========================
def train_and_evaluate(new_perfume):
    with mlflow.start_run(run_name="All Experiments") as parent_run:
        for vec_name, vectorizer in VECTORIZERS.items():
            with mlflow.start_run(run_name=f"using vectorizer: {vec_name}", nested=True):
                try:
                    # Feature extraction
                    tfidf_matrix = vectorizer.fit_transform(new_perfume['tags'])
                    print(f"Vectorizer: {vec_name}, Matrix shape: {tfidf_matrix.shape}")
                    
                    # Log preprocessing parameters
                    mlflow.log_params({
                        "vectorizer": vec_name,
                        "data_size": len(new_perfume),
                        "max_features": 1000
                    })

                    # Compute sparse cosine similarity
                    cosine_sim = sparse_cosine_similarity(tfidf_matrix)

                    # Log model parameters
                    mlflow.log_param("cosine_similarity_shape", cosine_sim.shape)
                    print(f"Cosine similarity shape for {vec_name}: {cosine_sim.shape}")
                    return cosine_sim
                except Exception as e:
                    print(f"Error in training {vec_name}: {e}")
                    mlflow.log_param("error", str(e))
                    raise

# ========================== RECOMMEND PERFUMES ==========================
def recommend_perfumes(cosine_sim, new_perfume, selected_perfumes, n=10):
    try:
        selected_indices = new_perfume[new_perfume['title'].isin([selected_perfumes])].index
        if not selected_indices.empty:
            sim_scores = cosine_sim[selected_indices].mean(axis=0)
            similar_indices = sim_scores.argsort()[::-1]
            recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
            return new_perfume.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]
        else:
            print(f"Selected perfume '{selected_perfumes}' not found in dataset")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error in recommend_perfumes: {e}")
        return pd.DataFrame()

# ========================== EXECUTION ==========================
if __name__ == "__main__":
    try:
        df = load_data(CONFIG["data_path"]).sample(2000, random_state=42)
        print(f"Sampled dataset size: {df.shape}")
        cos = train_and_evaluate(df)
        recommendations = recommend_perfumes(cos, df, df, n=10)
        print('Recommendations:\n', recommendations)
    except Exception as e:
        print(f"Main execution error: {e}")





# import setuptools
# import os
# import re
# import string
# import pandas as pd
# pd.set_option('future.no_silent_downcasting', True)
# import ast
# import numpy as np
# import mlflow
# import mlflow.sklearn
# import dagshub
# from sklearn.metrics.pairwise import cosine_similarity
# from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


# import warnings
# warnings.simplefilter("ignore", UserWarning)
# warnings.filterwarnings("ignore")

# # ========================== CONFIGURATION ==========================
# CONFIG = {
#     "data_path": "notebooks/perfumes_dataset.csv",
#     "data_size": 45000,
#     "mlflow_tracking_uri": "https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow",
#     "dagshub_repo_owner": "Razakhan143",
#     "dagshub_repo_name": "PERFUME_HAVEN_MLOPS_PROJECT",
#     "experiment_name": "Bow vs TfIdf"
# }

# # ========================== SETUP MLflow & DAGSHUB ==========================
# mlflow.set_tracking_uri(CONFIG["mlflow_tracking_uri"])
# dagshub.init(repo_owner=CONFIG["dagshub_repo_owner"], repo_name=CONFIG["dagshub_repo_name"], mlflow=True)
# mlflow.set_experiment(CONFIG["experiment_name"])

# # ========================== TEXT PREPROCESSING ==========================

# def collapse(L):
#     L1 = []
#     for i in ast.literal_eval(L):
#         L1.append(i.replace(" ",""))
#     return L1
# def normalize_text(perfumes):
#     try:
#         notes = perfumes['notes'].apply(collapse)
#         description = perfumes['description'].apply(lambda x:x.split())
#         designer = perfumes['designer'].apply(lambda x:x.split())
#         perfumes['tags'] = notes + description + designer
#         perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join(x)).str.lower()
#         perfumes['notes'] = perfumes['notes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x.capitalize())
#         perfumes['notes'] = perfumes['notes'].apply(
#         lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else x
#         )
#         perfumes.reset_index(drop=True, inplace=True)
#         return perfumes
#     except Exception as e:
#         print(f"Error during text normalization: {e}")
#         raise


# # ========================== LOAD & PREPROCESS DATA ==========================
# def load_data(file_path):
#     try:
#         df = pd.read_csv(file_path)
#         new_perfume = normalize_text(df)
#         return new_perfume
#     except Exception as e:
#         print(f"Error loading data: {e}")
#         raise

# # ========================== FEATURE ENGINEERING ==========================
# VECTORIZERS = {
#     'BoW': CountVectorizer(stop_words = 'english',max_features=5000),
#     'TF-IDF': TfidfVectorizer(stop_words = 'english',max_features=5000)
# }



# # ========================== cosin similarity ==========================
# def train_and_evaluate(new_perfume):
#     with mlflow.start_run(run_name="All Experiments") as parent_run:
#         for vec_name, vectorizer in VECTORIZERS.items():
#             with mlflow.start_run(run_name=f"using vectorizer : {vec_name}", nested=True) as child_run:
#                 try:
#                     # Feature extraction
#                     tfidf_matrix = vectorizer.fit_transform(new_perfume['tags'])
                 
#                     # Log preprocessing parameters
#                     mlflow.log_params({
#                         "vectorizer": vec_name,
#                         "data_size": CONFIG["data_size"]
#                     })

#                     # Train model
#                     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

#                     # Log model parameters
#                     log_model_params(cosine_sim, vec_name)
#                     mlflow.log_param("vectorizer", vec_name)
#                     mlflow.log_param("data_size", CONFIG["data_size"])
#                     mlflow.log_param("cosine_similarity_shape", cosine_sim.shape)
#                     print(f"Cosine similarity shape for {vec_name}: {cosine_sim.shape}")
#                     return cosine_sim
#                 except Exception as e:
#                     print(f"Error in training  {vec_name}: {e}")
#                     mlflow.log_param("error", str(e))

# def log_model_params(cosine_sim, vec_name):
#     """Logs hyperparameters of the trained model to MLflow."""
#     params_to_log = {}
#     params_to_log["vectorizer"] = vec_name
#     params_to_log["cosine_similarity_shape"] = cosine_sim.shape
#     mlflow.log_params(params_to_log)

# def recommend_perfumes(cosine_sim, new_perfume, selected_perfumes, n=10):
#     selected_indices = new_perfume[new_perfume['title'].isin(selected_perfumes)].index
#     sim_scores = cosine_sim[selected_indices].mean(axis=0)
#     similar_indices = sim_scores.argsort()[::-1]
#     recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]

#     return new_perfume.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]       
# # ========================== EXECUTION ==========================
# if __name__ == "__main__":
#     df = load_data(CONFIG["data_path"])
#     cos = train_and_evaluate(df)
#     print('data',recommend_perfumes(cos, df, 'Green Maremoto MATCA for women and men', n=10))
