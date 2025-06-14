# import ast
# import uvicorn
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware

# app = FastAPI()

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins for testing; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve static files from /static (for assets like CSS, images)
# app.mount("/static", StaticFiles(directory="perfume_haven/static"), name="static")

# # Serve index.html at root
# @app.get("/")
# async def read_index():
#     return FileResponse("perfume_haven/templates/index.html")

# # Serve search-results.html
# @app.get("/search-results.html")
# async def read_search_results():
#     return FileResponse("perfume_haven/templates/search-results.html")

# # Pydantic model for search request
# class SearchRequest(BaseModel):
#     query: str
# size_dataset = 10000  # Default size of the dataset to sample
# # Perfume data randomly sampled from the dataset
# perfume_data = pd.read_csv("notebooks/perfumes_dataset.csv",)
# perfume_data = perfume_data.sample(size_dataset, random_state=42).reset_index(drop=True)
# # Function to collapse notes into a list of strings
# def collapse(L):
#     L1 = []
#     for i in ast.literal_eval(L):
#         L1.append(i.replace(" ",""))
#     return L1
# # Function to create tags from notes, description, and designer for preprocessing
# def create_tags(perfumes):
#     notes = perfumes['notes'].apply(collapse)
#     description = perfumes['description'].apply(lambda x:x.split())
#     designer = perfumes['designer'].apply(lambda x:x.split())
#     perfumes_ = perfumes['title'].apply(lambda x: x.split())
#     perfumes['tags'] = notes + designer + perfumes_ + description
#     perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join(x)).str.lower()
#     perfumes['notes'] = perfumes['notes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x.capitalize())
#     perfumes['notes'] = perfumes['notes'].apply(
#     lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else x
# )
#     perfumes.reset_index(drop=True, inplace=True)
#     return perfumes

# # Preprocess the perfume data by vectorizing the tags and creating a cosine similarity matrix
# def vectorize_cosine_similarity(perfumes,max_features=1000):
#     from sklearn.feature_extraction.text import TfidfVectorizer
#     from sklearn.metrics.pairwise import cosine_similarity

#     vectorizer = TfidfVectorizer(stop_words = 'english', max_features=max_features)
#     tfidf_matrix = vectorizer.fit_transform(perfumes['tags'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     return cosine_sim

# # Load and preprocess the perfume data and make recommendations from it
# def recommend_perfumes(perfumes, selected_perfumes, cosine_sim, n = 10):
#     selected_indices = perfumes[perfumes['title'].isin(selected_perfumes)].index
#     if len(selected_indices) > 0:
#         sim_scores = cosine_sim[selected_indices].mean(axis=0)
#     else:
#         if len(selected_indices) == 0:
#             print("No selected indices found for similarity calculation.")  
#         sim_scores = np.zeros(cosine_sim.shape[0])  # Or return early / handle gracefully
    
#     # sim_scores = cosine_sim[selected_indices].mean(axis = 0)
#     similar_indices = sim_scores.argsort()[::-1]
#     recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
#     recommended_indices.extend(selected_indices.tolist())
#     recommended_indices = list(set(recommended_indices))
#     recommended_indices.sort()
#     return perfumes.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]

# # Suggestions endpoint
# @app.get("/suggestions")
# async def get_suggestions(query: str):
    
#     if not query:
#         return {"suggestions": []}
#     suggestions = [p["title"] for p in perfume_data.to_dict(orient="records") if query.lower() in p["title"].lower()]
#     return {"suggestions": suggestions[:5]}  # Return top 5 suggestions

# # Search endpoint
# @app.post("/search")
# async def search_perfumes(request: SearchRequest):
#     if not request.query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty")

#     #max_features
#     max_features = 5000
#     print(f"Vectorizing perfume data with max features {max_features}...")
#     # number of recommendations
#     n_recommendations = 10
#     print(f"Number of recommendations to return: {n_recommendations}")

#     perfumes = create_tags(pd.DataFrame(perfume_data))
#     cosine_sim = vectorize_cosine_similarity(perfumes, max_features=max_features)
#     print("Perfume data loaded and preprocessed successfully.")
#     recommended_perfumes = recommend_perfumes(perfumes, [request.query], cosine_sim, n=n_recommendations)
#     print(f"Search query: {request.query}")
   
#     if recommended_perfumes.empty:
#         raise HTTPException(status_code=404, detail="No perfumes found matching the query")
#     print(recommended_perfumes[:5])  # Print first 5 results for debugging
#     return {"results": recommended_perfumes.to_dict(orient="records")}


# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)





# import ast
# import uvicorn
# import pandas as pd
# import numpy as np
# from pydantic import BaseModel
# from fastapi import FastAPI, HTTPException
# from fastapi.responses import FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import mlflow
# import dagshub
# import os
# from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
# import time
# from pathlib import Path
# app = FastAPI()

# # MLflow and DagsHub setup
# # -------------------------------------------------------------------------------------
# # Set up DagsHub credentials for MLflow tracking
# dagshub_token = os.getenv("CAPSTONE_TEST")
# if not dagshub_token:
#     raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

# os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
# os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

# dagshub_url = "https://dagshub.com"
# repo_owner = "Razakhan143"
# repo_name = "PERFUME_HAVEN_MLOPS_PROJECT"
# # Set up MLflow tracking URI
# mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
# # -------------------------------------------------------------------------------------
# # Below code block is for local use
# # -------------------------------------------------------------------------------------
# # MLFLOW_TRACKING_URI = "https://dagshub.com/Razakhan143/PERFUME_HAVEN_MLOPS_PROJECT.mlflow"
# # dagshub.init(repo_owner="Razakhan143", repo_name="PERFUME_HAVEN_MLOPS_PROJECT", mlflow=True)
# # mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# # mlflow.set_experiment("capstone-trial")
# # -------------------------------------------------------------------------------------
# # Prometheus metrics setup
# registry = CollectorRegistry()
# REQUEST_COUNT = Counter(
#     "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
# )
# REQUEST_LATENCY = Histogram(
#     "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
# )
# PREDICTION_COUNT = Counter(
#     "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
# )

# # CORS configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Allow all origins for testing; restrict in production
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Serve static files from /static (for assets like CSS, images)
# static_dir = Path("perfume_haven/static")

# # Verify path exists (for debugging)
# print(f"Static files directory exists: {static_dir.exists()}")
# print(f"Contents: {list(static_dir.glob('*'))}")
# app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# # Serve index.html at root
# @app.get("/")
# async def read_index():
#     print("Serving index.html")
#     REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
#     start_time = time.time()
#     response = FileResponse("perfume_haven/templates/index.html")
#     REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
#     return response

# # Serve search-results.html
# @app.get("/search-results.html")
# async def read_search_results():
#     REQUEST_COUNT.labels(method="GET", endpoint="/search-results.html").inc()
#     start_time = time.time()
#     response = FileResponse("perfume_haven/templates/search-results.html")
#     REQUEST_LATENCY.labels(endpoint="/search-results.html").observe(time.time() - start_time)
#     return response

# # Pydantic model for search request
# class SearchRequest(BaseModel):
#     query: str

# size_dataset = 500
# perfume_data = pd.read_csv(
#     "notebooks/perfumes_dataset.csv",
#     usecols=['title', 'designer', 'description', 'notes', 'img_url'],
#     nrows=500  # Only reads first 500 rows
# ).sample(size_dataset, random_state=42).reset_index(drop=True)
# print(f"Loaded {len(perfume_data)} perfumes from dataset.")

# # Function to collapse notes into a list of strings
# def collapse(L):
#     L1 = []
#     for i in ast.literal_eval(L):
#         L1.append(i.replace(" ",""))
#     return L1

# # Function to create tags from notes, description, and designer for preprocessing
# def create_tags(perfumes):
#     notes = perfumes['notes'].apply(collapse)
#     description = perfumes['description'].apply(lambda x:x.split())
#     designer = perfumes['designer'].apply(lambda x:x.split())
#     perfumes_ = perfumes['title'].apply(lambda x: x.split())
#     perfumes['tags'] = notes + designer + perfumes_ + description
#     perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join(x)).str.lower()
#     perfumes['notes'] = perfumes['notes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x.capitalize())
#     perfumes['notes'] = perfumes['notes'].apply(
#         lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else x
#     )
#     perfumes.reset_index(drop=True, inplace=True)
#     print("Tags created successfully.")
#     return perfumes

# # Preprocess the perfume data by vectorizing the tags and creating a cosine similarity matrix
# def vectorize_cosine_similarity(perfumes,max_features=1000):
#     vectorizer = TfidfVectorizer(stop_words = 'english', max_features=max_features)
#     tfidf_matrix = vectorizer.fit_transform(perfumes['tags'])
#     cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#     print(f"Cosine similarity matrix created with shape {cosine_sim.shape}.")
#     return cosine_sim

# # Load and preprocess the perfume data and make recommendations from it
# def recommend_perfumes(perfumes, selected_perfumes, cosine_sim, n = 10):
#     selected_indices = perfumes[perfumes['title'].isin(selected_perfumes)].index
#     if len(selected_indices) > 0:
#         sim_scores = cosine_sim[selected_indices].mean(axis=0)
#     else:
#         if len(selected_indices) == 0:
#             print("No selected indices found for similarity calculation.")  
#         sim_scores = np.zeros(cosine_sim.shape[0])  # Or return early / handle gracefully
    
#     similar_indices = sim_scores.argsort()[::-1]
#     recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
#     recommended_indices.extend(selected_indices.tolist())
#     recommended_indices = list(set(recommended_indices))
#     recommended_indices.sort()
#     print(f"Recommended {len(recommended_indices)} perfumes based on the query.")
#     return perfumes.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]

# # Suggestions endpoint
# @app.get("/suggestions")
# async def get_suggestions(query: str):
#     REQUEST_COUNT.labels(method="GET", endpoint="/suggestions").inc()
#     start_time = time.time()
#     print(f"Received suggestions request: {query}")
#     if not query:
#         response = {"suggestions": []}
#     else:
#         suggestions = [p["title"] for p in perfume_data.to_dict(orient="records") if query.lower() in p["title"].lower()]
#         response = {"suggestions": suggestions[:5]}  # Return top 5 suggestions
    
#     REQUEST_LATENCY.labels(endpoint="/suggestions").observe(time.time() - start_time)
#     print(f"Suggestions response: {response}")
#     print(f"Number of suggestions: {len(response['suggestions'])}")
#     return response

# # Search endpoint
# @app.post("/search")
# async def search_perfumes(request: SearchRequest):
#     print(f"Received search request: {request.query}")
#     """Search for perfumes based on the query."""
#     REQUEST_COUNT.labels(method="POST", endpoint="/search").inc()
#     start_time = time.time()

#     if not request.query:
#         raise HTTPException(status_code=400, detail="Query cannot be empty")

#     #max_features
#     max_features = 5000
#     print(f"Vectorizing perfume data with max features {max_features}...")
#     # number of recommendations
#     n_recommendations = 10
#     print(f"Number of recommendations to return: {n_recommendations}")

#     perfumes = create_tags(pd.DataFrame(perfume_data))
#     cosine_sim = vectorize_cosine_similarity(perfumes, max_features=max_features)
#     print("Perfume data loaded and preprocessed successfully.")
#     recommended_perfumes = recommend_perfumes(perfumes, [request.query], cosine_sim, n=n_recommendations)
#     print(f"Search query: {request.query}")
#     print(f"Number of recommended perfumes: {len(recommended_perfumes)}")
#     if recommended_perfumes.empty:
#         raise HTTPException(status_code=404, detail="No perfumes found matching the query")
    
#     print(recommended_perfumes[:5])  # Print first 5 results for debugging
#     response = {"results": recommended_perfumes.to_dict(orient="records")}
    
#     REQUEST_LATENCY.labels(endpoint="/search").observe(time.time() - start_time)
#     print(f"Search response: {response}")
#     return response

# # Metrics endpoint for Prometheus
# @app.get("/metrics")
# async def metrics():
#     """Expose only custom Prometheus metrics."""
#     return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}

# if __name__ == "__main__":
#     uvicorn.run(
#         "app:app",  # Replace with actual filename (without `.py`)
#         host="0.0.0.0",
#         port=int(os.getenv("PORT", 5000)),
#         reload=True,               # Auto-reloads on code changes
#         log_level="debug",         # Ensures detailed logs
#     )







import ast
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import mlflow
import dagshub
import os
from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, CONTENT_TYPE_LATEST
import time
from pathlib import Path
import logging
from typing import Dict, Any
import traceback

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

app = FastAPI()

def log_error(error: Exception, context: Dict[str, Any] = None):
    """Helper function to log errors with context."""
    error_info = {
        "error": str(error),
        "type": type(error).__name__,
        "traceback": traceback.format_exc(),
        "context": context or {}
    }
    logger.error(f"Error occurred: {error_info}")

# MLflow and DagsHub setup
try:
    dagshub_token = os.getenv("CAPSTONE_TEST")
    if not dagshub_token:
        raise EnvironmentError("CAPSTONE_TEST environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    dagshub_url = "https://dagshub.com"
    repo_owner = "Razakhan143"
    repo_name = "PERFUME_HAVEN_MLOPS_PROJECT"
    mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')
    logger.info("MLflow tracking configured successfully")
except Exception as e:
    log_error(e, {"stage": "MLflow initialization"})
    raise

# Prometheus metrics setup
try:
    registry = CollectorRegistry()
    REQUEST_COUNT = Counter(
        "app_request_count", "Total number of requests to the app", ["method", "endpoint"], registry=registry
    )
    REQUEST_LATENCY = Histogram(
        "app_request_latency_seconds", "Latency of requests in seconds", ["endpoint"], registry=registry
    )
    PREDICTION_COUNT = Counter(
        "model_prediction_count", "Count of predictions for each class", ["prediction"], registry=registry
    )
    logger.info("Prometheus metrics configured")
except Exception as e:
    log_error(e, {"stage": "Prometheus setup"})
    raise

# CORS configuration
try:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    logger.info("CORS middleware configured")
except Exception as e:
    log_error(e, {"stage": "CORS setup"})
    raise
@app.get("/healthz")
async def health_check():
    return {"status": "healthy"}
# Static files setup
try:
    static_dir = Path("perfume_haven/static")
    if not static_dir.exists():
        logger.warning(f"Static directory not found at {static_dir}")
        static_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created static directory at {static_dir}")

    logger.info(f"Static files directory exists: {static_dir.exists()}")
    logger.info(f"Static directory contents: {list(static_dir.glob('*'))}")
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
except Exception as e:
    log_error(e, {"stage": "Static files setup"})
    raise
# # Pydantic model for search request
class SearchRequest(BaseModel):
    query: str
# Data loading
try:
    size_dataset = 5000
    perfume_data = pd.read_csv(
        "notebooks/perfumes_dataset.csv",
        usecols=['title', 'designer', 'description', 'notes', 'img_url'],
        nrows=5000
    ).sample(size_dataset, random_state=42).reset_index(drop=True)
    logger.info(f"Successfully loaded {len(perfume_data)} perfume records")
except Exception as e:
    log_error(e, {
        "stage": "Data loading",
        "size_dataset": size_dataset,
        "columns": ['title', 'designer', 'description', 'notes', 'img_url']
    })
    raise

# Helper functions with error handling
def collapse(L):
    try:
        L1 = []
        for i in ast.literal_eval(L):
            L1.append(i.replace(" ",""))
        return L1
    except Exception as e:
        log_error(e, {"function": "collapse", "input": L})
        raise

def create_tags(perfumes):
    try:
        notes = perfumes['notes'].apply(collapse)
        description = perfumes['description'].apply(lambda x:x.split())
        designer = perfumes['designer'].apply(lambda x:x.split())
        perfumes_ = perfumes['title'].apply(lambda x: x.split())
        perfumes['tags'] = notes + designer + perfumes_ + description
        perfumes['tags'] = perfumes['tags'].apply(lambda x: " ".join(x)).str.lower()
        perfumes['notes'] = perfumes['notes'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x.capitalize())
        perfumes['notes'] = perfumes['notes'].apply(
            lambda x: ', '.join(word.title() for word in x) if isinstance(x, list) else x
        )
        perfumes.reset_index(drop=True, inplace=True)
        logger.info("Successfully created tags")
        return perfumes
    except Exception as e:
        log_error(e, {"function": "create_tags", "input_shape": perfumes.shape})
        raise

def vectorize_cosine_similarity(perfumes, max_features=1000):
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=max_features)
        tfidf_matrix = vectorizer.fit_transform(perfumes['tags'])
        cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
        logger.info(f"Created cosine similarity matrix with shape {cosine_sim.shape}")
        return cosine_sim
    except Exception as e:
        log_error(e, {
            "function": "vectorize_cosine_similarity",
            "max_features": max_features,
            "tags_sample": perfumes['tags'].head(3).tolist()
        })
        raise

def recommend_perfumes(perfumes, selected_perfumes, cosine_sim, n=10):
    try:
        selected_indices = perfumes[perfumes['title'].isin(selected_perfumes)].index
        if len(selected_indices) > 0:
            sim_scores = cosine_sim[selected_indices].mean(axis=0)
        else:
            logger.warning("No selected indices found for similarity calculation")
            sim_scores = np.zeros(cosine_sim.shape[0])
        
        similar_indices = sim_scores.argsort()[::-1]
        recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
        recommended_indices.extend(selected_indices.tolist())
        recommended_indices = list(set(recommended_indices))
        recommended_indices.sort()
        logger.info(f"Generated {len(recommended_indices)} recommendations")
        return perfumes.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]
    except Exception as e:
        log_error(e, {
            "function": "recommend_perfumes",
            "selected_perfumes": selected_perfumes,
            "cosine_sim_shape": cosine_sim.shape
        })
        raise

# API endpoints with error handling
@app.get("/")
async def read_index():
    REQUEST_COUNT.labels(method="GET", endpoint="/").inc()
    start_time = time.time()
    try:
        logger.info("Serving index.html")
        response = FileResponse("perfume_haven/templates/index.html")
        REQUEST_LATENCY.labels(endpoint="/").observe(time.time() - start_time)
        return response
    except Exception as e:
        log_error(e, {"endpoint": "/", "method": "GET"})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/search-results.html")
async def read_search_results():
    REQUEST_COUNT.labels(method="GET", endpoint="/search-results.html").inc()
    start_time = time.time()
    try:
        logger.info("Serving search-results.html")
        response = FileResponse("perfume_haven/templates/search-results.html")
        REQUEST_LATENCY.labels(endpoint="/search-results.html").observe(time.time() - start_time)
        return response
    except Exception as e:
        log_error(e, {"endpoint": "/search-results.html", "method": "GET"})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.get("/suggestions")
async def get_suggestions(query: str):
    REQUEST_COUNT.labels(method="GET", endpoint="/suggestions").inc()
    start_time = time.time()
    try:
        logger.info(f"Suggestions request received for query: '{query}'")
        if not query:
            response = {"suggestions": []}
        else:
            suggestions = [p["title"] for p in perfume_data.to_dict(orient="records") if query.lower() in p["title"].lower()]
            response = {"suggestions": suggestions[:5]}
        
        logger.info(f"Returning {len(response['suggestions'])} suggestions")
        REQUEST_LATENCY.labels(endpoint="/suggestions").observe(time.time() - start_time)
        return response
    except Exception as e:
        log_error(e, {"endpoint": "/suggestions", "query": query})
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/search")
async def search_perfumes(request: SearchRequest):
    REQUEST_COUNT.labels(method="POST", endpoint="/search").inc()
    start_time = time.time()
    try:
        logger.info(f"Search request received for query: '{request.query}'")
        
        if not request.query:
            logger.warning("Empty query received")
            raise HTTPException(status_code=400, detail="Query cannot be empty")

        max_features = 5000
        n_recommendations = 10
        
        logger.info(f"Processing search with max_features={max_features}, n_recommendations={n_recommendations}")
        
        perfumes = create_tags(pd.DataFrame(perfume_data))
        cosine_sim = vectorize_cosine_similarity(perfumes, max_features=max_features)
        recommended_perfumes = recommend_perfumes(perfumes, [request.query], cosine_sim, n=n_recommendations)
        
        if recommended_perfumes.empty:
            logger.warning(f"No results found for query: '{request.query}'")
            raise HTTPException(status_code=404, detail="No perfumes found matching the query")
        
        response = {"results": recommended_perfumes.to_dict(orient="records")}
        logger.info(f"Returning {len(response['results'])} recommendations")
        
        REQUEST_LATENCY.labels(endpoint="/search").observe(time.time() - start_time)
        return response
    except HTTPException:
        raise  # Re-raise HTTP exceptions as they are
    except Exception as e:
        log_error(e, {
            "endpoint": "/search",
            "query": request.query if hasattr(request, 'query') else None,
            "max_features": max_features,
            "n_recommendations": n_recommendations
        })

@app.get("/metrics")
async def metrics():
    try:
        logger.info("Serving metrics endpoint")
        return generate_latest(registry), 200, {"Content-Type": CONTENT_TYPE_LATEST}
    except Exception as e:
        log_error(e, {"endpoint": "/metrics"})
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    try:
        logger.info("Starting Uvicorn server")
        uvicorn.run(
            "app:app",
            host="0.0.0.0",
            port=int(os.getenv("PORT", 5000)),
            reload=True,
            log_level="debug",
            log_config={
                "version": 1,
                "disable_existing_loggers": False,
                "formatters": {
                    "default": {
                        "()": "uvicorn.logging.DefaultFormatter",
                        "fmt": "%(levelprefix)s %(message)s",
                        "use_colors": None,
                    },
                    "access": {
                        "()": "uvicorn.logging.AccessFormatter",
                        "fmt": '%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                    },
                },
                "handlers": {
                    "default": {
                        "formatter": "default",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stderr",
                    },
                    "access": {
                        "formatter": "access",
                        "class": "logging.StreamHandler",
                        "stream": "ext://sys.stdout",
                    },
                },
                "loggers": {
                    "uvicorn": {"handlers": ["default"], "level": "INFO"},
                    "uvicorn.error": {"level": "INFO"},
                    "uvicorn.access": {"handlers": ["access"], "level": "INFO", "propagate": False},
                },
            }
        )
    except Exception as e:
        log_error(e, {"stage": "Server startup"})
        raise