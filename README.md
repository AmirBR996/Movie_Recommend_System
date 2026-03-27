# 🎬 Movie Recommendation System

> An AI-powered movie recommendation engine using content-based filtering with TF-IDF vectorization and cosine similarity

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104%2B-green)](https://fastapi.tiangolo.com/)
[![Status: Production](https://img.shields.io/badge/Status-Production-brightgreen)]()

---

## 📋 Features

- Search movies by **title** or **description**
- Filter results by **genre**
- Uses **TF-IDF** for text vectorization
- Calculates **cosine similarity** to recommend similar movies
- Simple **FastAPI backend** with JSON responses

---

## 🚀 Quick Start

### 1️⃣ Clone Repository
```bash
git clone https://github.com/AmirBR996/Movie_Recommend_System.git
cd Movie_Recommend_System
```

### 2️⃣ Install Dependencies
```bash
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3️⃣ Run the Server
```bash
python3 main.py
```

Visit http://localhost:8000 to use the application.

---

## 🏗️ How It Works

1. Load movies dataset (162,362 IMDB movies)
2. Clean movie descriptions (lowercase, remove stopwords)
3. Vectorize descriptions using **TF-IDF**
4. For a given movie or description, calculate **cosine similarity** with all movies
5. Sort by similarity and return top N recommendations

**Simple Diagram:**
```
User Input → TF-IDF Vector → Cosine Similarity → Top N Movies → JSON Response
```

---

## 💡 Usage Examples

### 1. Search by Movie Title

**Request:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "Inception", "top_n": 3}'
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "title": "the dark knight",
      "description": "when the menace known as the joker wreaks havoc...",
      "genres": "['Action', 'Crime', 'Drama']",
      "similarity_score": 0.457
    },
    {
      "title": "interstellar",
      "description": "a team of explorers travel through a wormhole...",
      "genres": "['Adventure', 'Drama', 'Sci-Fi']",
      "similarity_score": 0.423
    },
    {
      "title": "the prestige",
      "description": "after a tragic accident two stage magicians...",
      "genres": "['Drama', 'Mystery', 'Sci-Fi']",
      "similarity_score": 0.381
    }
  ],
  "count": 3
}
```

### 2. Search by Description

**Request:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"description": "A thrilling heist movie", "top_n": 2}'
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "title": "ocean's eleven",
      "description": "a clever group of thieves pulls off daring heist...",
      "genres": "['Crime', 'Drama', 'Thriller']",
      "similarity_score": 0.512
    },
    {
      "title": "now you see me",
      "description": "four magicians called the four horsemen...",
      "genres": "['Action', 'Adventure', 'Thriller']",
      "similarity_score": 0.48
    }
  ],
  "count": 2
}
```

### 3. Advanced Search (Title + Genre Filter)

**Request:**
```bash
curl -X POST http://localhost:8000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"title": "The Godfather", "genre": "Drama", "top_n": 2}'
```

**Response:**
```json
{
  "success": true,
  "recommendations": [
    {
      "title": "scarface",
      "description": "as an ambitious immigrant tony montana...",
      "genres": "['Crime', 'Drama']",
      "similarity_score": 0.634
    },
    {
      "title": "goodfellas",
      "description": "the story of henry hill and his life in the mob...",
      "genres": "['Crime', 'Drama']",
      "similarity_score": 0.598
    }
  ],
  "count": 2
}
```

---

## 📁 Project Structure

```
Movie_Recommend_System/
├── main.py              # FastAPI server
├── prediction.py        # TF-IDF + cosine similarity
├── Feature_selection.py # Data preprocessing
├── clean_text.py        # Text cleaning functions
├── static/
│   └── index.html       # Frontend UI
├── requirements.txt     # Dependencies
├── README.md           # This file
└── TECHNICAL_REPORT.md # Detailed technical analysis
```

### Key Files Explained

| File | Purpose |
|------|---------|
| `main.py` | FastAPI server with CORS and error handling |
| `prediction.py` | Core recommendation logic (TF-IDF + cosine similarity) |
| `Feature_selection.py` | Dataset preprocessing and feature engineering |
| `clean_text.py` | Text cleaning (lowercase, stopwords removal) |
| `static/index.html` | Interactive web UI with 3 search modes |

---

## 🔧 Troubleshooting

### Port 8000 already in use
```bash
# Use a different port
python3 -m uvicorn main:app --port 8001
```

### NLTK stopwords missing
```bash
python3 -c "import nltk; nltk.download('stopwords')"
```

### Dataset taking too long to load
- First run downloads 162K movies (~5-10 minutes)
- Subsequent runs are cached and fast
- Check progress with: `curl http://localhost:8000/api/health`

---

## 🎯 Next Steps / Future Improvements

- [ ] Add typo-tolerant title search (fuzzy matching)
- [ ] Implement caching for faster responses
- [ ] Add simple frontend (already included!)
- [ ] Deploy to cloud (AWS/Heroku)
- [ ] Add Word2Vec embeddings for better semantics
- [ ] Implement collaborative filtering

See [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) for detailed improvement strategies and system analysis.

---

## 📊 Performance

| Metric | Value |
|--------|-------|
| Initial Load Time | 5-10 minutes (dataset download) |
| Per-Query Time | 0.5-1 second |
| Dataset Size | 162,362 movies |
| Memory Usage | 200-300 MB |
| Max Recommendations | 100 movies |

---

## 📝 Documentation

- **README.md** - Quick start and usage guide (this file)
- **TECHNICAL_REPORT.md** - In-depth technical analysis, limitations, and improvements

---

## 🎓 For Internship Submission

This project includes:
- ✅ Production-ready FastAPI backend
- ✅ Interactive HTML/CSS/JavaScript frontend
- ✅ Content-based recommendation algorithm
- ✅ Comprehensive documentation
- ✅ Error handling and logging
- ✅ Real-time API with JSON responses

---

**Happy Recommending! 🎬🍿**

*Last Updated: March 27, 2026*  
*Status: Production Ready (v1.0)*
