import os
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from fastapi.responses import FileResponse

app = FastAPI(title="Movie Recommender API")

# Configure CORS for Vercel deployment and local development
import os

# Determine allowed origins based on environment
allowed_origins = [
    "http://localhost:3000",  # Common React dev port
    "http://localhost:8000",  # Our default port
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8000",
]

# In Vercel deployments, we can allow the Vercel URL
vercel_url = os.environ.get("VERCEL_URL")
if vercel_url:
    allowed_origins.extend([
        f"https://{vercel_url}",
        f"http://{vercel_url}",
    ])

# For production, we could add specific domains here
# For now, we'll keep it flexible but not overly permissive
# The same-origin policy will handle frontend-backend communication when served together

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load your exported pkl artifacts safely
BASE_DIR = Path(__file__).resolve().parent
with open(BASE_DIR / "movie_model.pkl", "rb") as f:
    artifacts = pickle.load(f)

MOVIE_FACTORS = artifacts["movie_factors"]
MOVIE_BIAS = artifacts["movie_bias"]
MOVIE_TITLES = artifacts["movie_titles"]
Y_RANGE = artifacts["y_range"]

class RatingRequest(BaseModel):
    ratings: dict  # Example: {"Star Wars (1977)": 5, "Titanic (1997)": 2}

def fit_new_user_embedding(rated_indices, ratings_given, y_range=(0, 5.5), n_epochs=150, lr=0.05, wd=0.01):
    n_factors = MOVIE_FACTORS.shape[1]
    movie_factors_t = torch.tensor(MOVIE_FACTORS, dtype=torch.float32)
    movie_bias_t = torch.tensor(MOVIE_BIAS, dtype=torch.float32)

    idx = torch.tensor(rated_indices, dtype=torch.long)
    y = torch.tensor(ratings_given, dtype=torch.float32)

    user_vec = torch.zeros(n_factors, requires_grad=True)
    user_bias = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([user_vec, user_bias], lr=lr, weight_decay=wd)

    lo, hi = y_range
    for _ in range(n_epochs):
        opt.zero_grad()
        dot = (movie_factors_t[idx] * user_vec).sum(dim=1)
        raw_pred = dot + movie_bias_t[idx] + user_bias
        pred = torch.sigmoid(raw_pred) * (hi - lo) + lo
        loss = F.mse_loss(pred, y)
        loss.backward()
        opt.step()

    return user_vec.detach().numpy(), user_bias.detach().item()

@app.post("/recommend")
def recommend(payload: RatingRequest):
    rated_titles_and_scores = payload.ratings
    title_to_idx = {t: i for i, t in enumerate(MOVIE_TITLES)}
    
    rated_indices = [title_to_idx[t] for t in rated_titles_and_scores if t in title_to_idx]
    ratings_given = [float(rated_titles_and_scores[MOVIE_TITLES[i]]) for i in rated_indices]

    if not rated_indices:
        return {"recommendations": []}

    user_vec, user_bias = fit_new_user_embedding(rated_indices, ratings_given, Y_RANGE)

    movie_factors_t = torch.tensor(MOVIE_FACTORS, dtype=torch.float32)
    movie_bias_t = torch.tensor(MOVIE_BIAS, dtype=torch.float32)
    user_vec_t = torch.tensor(user_vec, dtype=torch.float32)

    raw_pred = (movie_factors_t * user_vec_t).sum(dim=1) + movie_bias_t + user_bias
    lo, hi = Y_RANGE
    preds = torch.sigmoid(raw_pred) * (hi - lo) + lo

    preds_np = preds.detach().numpy()
    preds_np[rated_indices] = -np.inf  # Don't recommend already rated movies

    top_idx = np.argsort(-preds_np)[:10]
    recs = [(MOVIE_TITLES[i], float(preds_np[i])) for i in top_idx]

    return {"recommendations": recs}

@app.get("/")
def serve_frontend():
    # This tells the server to load your HTML file when someone visits the site
    return FileResponse(BASE_DIR / "index.html")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)