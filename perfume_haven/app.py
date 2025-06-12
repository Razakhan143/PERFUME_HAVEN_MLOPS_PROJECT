

import ast
import uvicorn
import pandas as pd
import numpy as np
from pydantic import BaseModel
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing; restrict in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static files from /static (for assets like CSS, images)
app.mount("/static", StaticFiles(directory="perfume_haven/static"), name="static")

# Serve index.html at root
@app.get("/")
async def read_index():
    return FileResponse("perfume_haven/templates/index.html")

# Serve search-results.html
@app.get("/search-results.html")
async def read_search_results():
    return FileResponse("perfume_haven/templates/search-results.html")

# Pydantic model for search request
class SearchRequest(BaseModel):
    query: str
size_dataset = 10000  # Default size of the dataset to sample
# Perfume data randomly sampled from the dataset
perfume_data = pd.read_csv("notebooks/perfumes_dataset.csv",)
perfume_data = perfume_data.sample(size_dataset, random_state=42).reset_index(drop=True)
# Function to collapse notes into a list of strings
def collapse(L):
    L1 = []
    for i in ast.literal_eval(L):
        L1.append(i.replace(" ",""))
    return L1
# Function to create tags from notes, description, and designer for preprocessing
def create_tags(perfumes):
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
    return perfumes

# Preprocess the perfume data by vectorizing the tags and creating a cosine similarity matrix
def vectorize_cosine_similarity(perfumes,max_features=1000):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    vectorizer = TfidfVectorizer(stop_words = 'english', max_features=max_features)
    tfidf_matrix = vectorizer.fit_transform(perfumes['tags'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cosine_sim

# Load and preprocess the perfume data and make recommendations from it
def recommend_perfumes(perfumes, selected_perfumes, cosine_sim, n = 10):
    selected_indices = perfumes[perfumes['title'].isin(selected_perfumes)].index
    if len(selected_indices) > 0:
        sim_scores = cosine_sim[selected_indices].mean(axis=0)
    else:
        if len(selected_indices) == 0:
            print("No selected indices found for similarity calculation.")  
        sim_scores = np.zeros(cosine_sim.shape[0])  # Or return early / handle gracefully
    
    # sim_scores = cosine_sim[selected_indices].mean(axis = 0)
    similar_indices = sim_scores.argsort()[::-1]
    recommended_indices = [i for i in similar_indices if i not in selected_indices][:n]
    recommended_indices.extend(selected_indices.tolist())
    recommended_indices = list(set(recommended_indices))
    recommended_indices.sort()
    return perfumes.iloc[recommended_indices][['title', 'designer', 'description', 'notes', 'img_url']]

# Suggestions endpoint
@app.get("/suggestions")
async def get_suggestions(query: str):
    
    if not query:
        return {"suggestions": []}
    suggestions = [p["title"] for p in perfume_data.to_dict(orient="records") if query.lower() in p["title"].lower()]
    return {"suggestions": suggestions[:5]}  # Return top 5 suggestions

# Search endpoint
@app.post("/search")
async def search_perfumes(request: SearchRequest):
    if not request.query:
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    #max_features
    max_features = 5000
    print(f"Vectorizing perfume data with max features {max_features}...")
    # number of recommendations
    n_recommendations = 10
    print(f"Number of recommendations to return: {n_recommendations}")

    perfumes = create_tags(pd.DataFrame(perfume_data))
    cosine_sim = vectorize_cosine_similarity(perfumes, max_features=max_features)
    print("Perfume data loaded and preprocessed successfully.")
    recommended_perfumes = recommend_perfumes(perfumes, [request.query], cosine_sim, n=n_recommendations)
    print(f"Search query: {request.query}")
   
    if recommended_perfumes.empty:
        raise HTTPException(status_code=404, detail="No perfumes found matching the query")
    print(recommended_perfumes[:5])  # Print first 5 results for debugging
    return {"results": recommended_perfumes.to_dict(orient="records")}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)