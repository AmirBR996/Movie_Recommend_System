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

**System Architecture:**
```
┌─────────────────────────────────────────────────────────────┐
│                    🌐 Frontend (Browser)                     │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  🔍 Search Bar (Title / Description / Advanced)      │   │
│  │  Beautiful UI with 3 tabs & animations              │   │
│  └───────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ HTTP POST/GET (JSON)
                         ↓
┌─────────────────────────────────────────────────────────────┐
│            📡 FastAPI Backend (Port 8000)                    │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  POST /api/recommend → Get recommendations           │   │
│  │  GET /api/genres → List available genres            │   │
│  │  GET /api/health → Check server status              │   │
│  └───────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ↓
┌─────────────────────────────────────────────────────────────┐
│     🧠 Recommendation Engine (TF-IDF + Cosine Similarity)   │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  1️⃣  Load 162,362 movies from IMDB dataset           │   │
│  │  2️⃣  Preprocess descriptions (clean, tokenize)       │   │
│  │  3️⃣  Create TF-IDF matrix (162K × 5,000)             │   │
│  │  4️⃣  Find target movie or vectorize query            │   │
│  │  5️⃣  Compute cosine similarity scores                │   │
│  │  6️⃣  Sort by score & apply genre filter              │   │
│  │  7️⃣  Return top N recommendations                    │   │
│  └───────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │ JSON Response
                         ↓
┌─────────────────────────────────────────────────────────────┐
│           📊 Frontend Displays Results                       │
│  ┌───────────────────────────────────────────────────────┐   │
│  │  🎬 Movie Cards with:                               │   │
│  │     • Title   • Description   • Genres               │   │
│  │     • Similarity Score (0-1)                         │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

**Algorithm Flow:**
```
User Input: "Find movies like Inception"
         ↓
   🔎 Parse Request & Validate
         ↓
   📦 Load Preprocessed Data (162,362 movies)
         ↓
   🎯 Find "Inception" in dataset OR vectorize description
         ↓
   📐 Calculate Cosine Similarity with ALL movies
         ↓
   ⬆️  Sort by highest similarity score
         ↓
   🏷️  Apply genre filter if specified
         ↓
   ✅ Return top 5 recommendations as JSON
         ↓
   🎨 Display beautiful movie cards
```

---

## 💡 Usage Examples

### 🎯 Frontend Screenshot (ASCII UI)

```
╔════════════════════════════════════════════════════════════════╗
║                 🎬 Movie Recommendation System                  ║
║              Find your next favorite movie                      ║
╠════════════════════════════════════════════════════════════════╣
║                                                                ║
║  [By Title]  [By Description]  [Advanced Search]              ║
║                                                                ║
║  ┌──────────────────────────────────────────────────────┐    ║
║  │ Movie Title                                          │    ║
║  │ ┌────────────────────────────────────────────────┐   │    ║
║  │ │ Inception                                      │   │    ║
║  │ └────────────────────────────────────────────────┘   │    ║
║  │                                                      │    ║
║  │ Number of Recommendations                           │    ║
║  │ ┌────────────────────────────────────────────────┐   │    ║
║  │ │ 5                                              │   │    ║
║  │ └────────────────────────────────────────────────┘   │    ║
║  │                                                      │    ║
║  │ [ Find Recommendations ]                            │    ║
║  └──────────────────────────────────────────────────────┘    ║
║                                                                ║
║  ✓ Found 3 recommendations!                                   ║
║                                                                ║
║  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────┐ ║
║  │ The Dark Knight  │  │ Interstellar     │  │ The Prestige │ ║
║  │ Action,Crime     │  │ Adventure,Drama  │  │ Drama,Sci-Fi │ ║
║  │ Score: 0.457     │  │ Score: 0.423     │  │ Score: 0.381 │ ║
║  └──────────────────┘  └──────────────────┘  └──────────────┘ ║
║                                                                ║
╚════════════════════════════════════════════════════════════════╝
```

### 1️⃣ Search by Movie Title

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

### 2️⃣ Search by Description

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

### 3️⃣ Advanced Search (Title + Genre Filter)

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
│
├── 📄 main.py                    # 🚀 FastAPI server (core)
│
├── 🧠 prediction.py              # Core TF-IDF + cosine similarity algorithm
│
├── 📊 Feature_selection.py        # Data preprocessing & feature extraction
│
├── 🔤 clean_text.py              # Text cleaning utilities
│
├── 📁 static/
│   └── 🎨 index.html             # Beautiful interactive frontend UI
│
├── 📋 requirements.txt            # Python dependencies
│
├── 📖 README.md                  # Quick start guide (you are here)
│
├── 📚 TECHNICAL_REPORT.md        # In-depth technical analysis
│
├── 🔧 run.sh                     # Linux/Mac startup script
│
└── 🔧 run.bat                    # Windows startup script
```

### Key Files Explained

| File | Purpose | What It Does |
|------|---------|-------------|
| `main.py` | 🚀 FastAPI Server | Handles HTTP requests, manages endpoints |
| `prediction.py` | 🧠 Algorithm | TF-IDF vectorization + cosine similarity |
| `Feature_selection.py` | 📊 Preprocessing | Clean dataset, select features |
| `clean_text.py` | 🔤 Text Tools | Lowercase, remove stopwords, tokenize |
| `static/index.html` | 🎨 Frontend | 3 search modes, responsive design |

### Data Flow Diagram

```
📁 Raw Dataset (IMDB)
  │
  ├─→ 🔤 clean_text.py
  │   (Lowercase, remove special chars, remove stopwords)
  │
  ├─→ 📊 Feature_selection.py
  │   (Remove duplicates, format data)
  │
  ├─→ 🧠 prediction.py
  │   (Convert to TF-IDF vectors)
  │
  └─→ 💾 Memory (cached vectors)
      │
      ├─→ 🎬 User searches "Inception"
      │   │
      │   ├─→ 📐 Cosine Similarity Calculation
      │   │   (Compare with 162K movies)
      │   │
      │   ├─→ ⬆️ Sort by highest score
      │   │
      │   └─→ 🏷️ Apply genre filter
      │
      └─→ 📡 FastAPI returns JSON
          │
          └─→ 🎨 Frontend displays results
```

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
