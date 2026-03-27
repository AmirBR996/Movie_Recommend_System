from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from clean_text import clean_text
import pandas as pd
def recommend_movies(title=None, description=None, genre=None, top_n=5 , df = 'train_df'):
    tfidf = TfidfVectorizer(max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['description'])
    indices = pd.Series(df.index, index=df['title'])
    from sklearn.metrics.pairwise import cosine_similarity
    results = None
    
    if title:
        title_clean = title.lower().strip()
        matches = df[df['title'].str.contains(title_clean, case=False, na=False)]
        if matches.empty:
            return "Movie title not found!"
        idx = matches.index[0]
        similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix)[0]
        similarity_scores = [s for s in enumerate(similarity_scores) if s[0] != idx]
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in similarity_scores]
        results = df.iloc[movie_indices]

    elif description:
        clean_description = clean_text(description)
        desc_vec = tfidf.transform([clean_description])
        similarity_scores = cosine_similarity(desc_vec, tfidf_matrix)[0]
        similarity_scores = list(enumerate(similarity_scores))
        similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
        movie_indices = [i[0] for i in similarity_scores]
        results = df.iloc[movie_indices]

    else:
        return "Please provide a movie title or description!"
    
    if genre:
        results = results[results['genres'].str.contains(genre.lower(), case=False, na=False)]

    return results[['title', 'description', 'genres']].head(top_n)