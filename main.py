from fastapi import FastAPI
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from datasets import load_dataset
from Feature_selection import feature_selection
from prediction import recommend_movies

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

train_df = None

@app.on_event("startup")
def load_data():
    global train_df
    ds = load_dataset("jquigl/imdb-genres")
    train_df = feature_selection(ds["train"].to_pandas())
    print("✅ Dataset loaded")

class Request(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    genre: Optional[str] = None
    top_n: int = 5

@app.post("/api/recommend")
def recommend(req: Request):
    if train_df is None:
        return JSONResponse({"error": "Dataset not loaded"}, status_code=500)

    if not req.title and not req.description:
        return JSONResponse({"error": "Enter title or description"}, status_code=400)

    result = recommend_movies(
        title=req.title,
        description=req.description,
        genre=req.genre,
        top_n=req.top_n,
        df=train_df
    )

    if isinstance(result, str):
        return JSONResponse({"error": result}, status_code=404)

    return {
        "recommendations": result.to_dict("records")
    }


@app.get("/api/genres")
def genres():
    if train_df is None:
        return {"genres": []}

    import ast
    all_genres = set()

    for g in train_df["genres"].dropna():
        try:
            all_genres.update(ast.literal_eval(g))
        except:
            pass

    return {"genres": list(all_genres)}

if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def home():
    return FileResponse("static/index.html")

@app.get("/{path:path}")
def catch_all(path: str):
    if path.startswith("api/"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)