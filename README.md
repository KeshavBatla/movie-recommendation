#Website link: https://movie-recommendation-nine-ashen.vercel.app/
# 🎬 Movie Recommendation System using Collaborative Filtering

A movie recommendation system built using FastAI's Collaborative Filtering module on the MovieLens 100K dataset.

The project learns user and movie embeddings to predict movie ratings and recommend movies based on user preferences.

---

## Features

- Collaborative Filtering Recommendation
- Matrix Factorization Model
- Neural Collaborative Filtering
- MovieLens 100K Dataset
- FastAI Implementation

---

## Dataset

This project uses the **MovieLens 100K** dataset.

The dataset is automatically downloaded using FastAI:

```python
path = untar_data(URLs.ML_100k)
```

Dataset contains:

- 943 Users
- 1,682 Movies
- 100,000 Ratings

---

## Models

### Matrix Factorization

- Embedding Size: 50
- Weight Decay: 0.1

```python
collab_learner(
    dls,
    n_factors=50,
    y_range=(0,5.5)
)
```

---

### Neural Collaborative Filtering

```python
collab_learner(
    dls,
    use_nn=True,
    layers=[100,50],
    y_range=(0,5.5)
)
```

---

## Technologies Used

- Python
- FastAI
- PyTorch
- Pandas

---

## Installation

Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/movie-recommendation-system.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run

Run the notebook

```bash
jupyter notebook
```

or execute the Python script.

---

## Project Workflow

```
MovieLens Dataset
        │
        ▼
Load Ratings
        │
        ▼
Merge Movie Titles
        │
        ▼
Create DataLoaders
        │
        ▼
Train Collaborative Filtering Model
        │
        ▼
Generate Movie Recommendations
```

---

## Example

After training, the model can identify movies with the highest learned bias scores, indicating generally well-rated or popular movies.

---

## Future Improvements

- Content-based recommendation
- Hybrid recommendation system
- Streamlit web interface
- Deploy recommendation API using FastAPI

---

## Author

Keshav Batla
