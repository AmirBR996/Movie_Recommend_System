# 🎬 Movie Recommendation System

AI-powered movie recommendations using TF-IDF + cosine similarity

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)

## 📋 Features

- 🔍 Search by movie title or description
- 🏷️ Filter by genre
- 📊 162,362 movies from IMDB
- ⚡ Fast recommendations (0.5-1 second per query)
- 🎨 Beautiful web interface

## 🚀 Quick Start

```bash
# 1. Clone & enter directory
git clone https://github.com/AmirBR996/Movie_Recommend_System.git
cd Movie_Recommend_System

# 2. Setup environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run server
python3 main.py
```

Visit **http://localhost:8000** in your browser

## 📱 Usage

**Search by title:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "top_n": 5}'
  ![Movie Title Recommendation](/recommendation_using_title.png)

```

**Search by description:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"description": "A thrilling heist movie", "top_n": 5}'
  ![Movie Title Recommendation](/recommendation_using_description.png)

```

**Filter by genre:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "The Godfather", "genre": "Drama", "top_n": 5}'
```

## 📁 Project Structure

```
Movie_Recommend_System/
├── main.py                 # FastAPI server
├── prediction.py           # TF-IDF + cosine similarity algorithm
├── Feature_selection.py    # Data preprocessing
├── clean_text.py           # Text cleaning utilities
├── static/
│   └── index.html          # Web UI
├── requirements.txt        # Dependencies
└── README.md              # This file
```

## ⚙️ How It Works

1. **Load**: Preprocess 162,362 IMDB movies
2. **Vectorize**: Convert movie descriptions to TF-IDF vectors
3. **Search**: User enters title or description
4. **Calculate**: Compute cosine similarity against all movies
5. **Return**: Top N most similar movies ranked by score

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| Port 8000 in use | `python3 -m uvicorn main:app --port 8001` |
| Stopwords missing | `python3 -c "import nltk; nltk.download('stopwords')"` |
| Slow first run | Dataset download takes 5-10 min (cached after) |

## 📚 More Info

- See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for deep technical analysis
- API responses include similarity scores (0-1 range)
- Frontend has 3 search modes: Title, Description, Advanced

---

**Happy Recommending! 🎬**
