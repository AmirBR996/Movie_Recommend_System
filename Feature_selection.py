from clean_text import clean_text
def feature_selection(df):
    df[['title', 'year']] = df['movie title - year'].str.rsplit(' - ', n=1, expand=True)
    df = df.drop(['movie title - year' , 'genre' , 'year' , 'rating'] , axis= 1)
    df = df.rename(columns={'expanded-genres': 'genres'})
    df['description'] = df['description'].apply(clean_text)
    df['genres'] = df['genres'].str.lower()
    df['title'] = df['title'].str.lower()
    df = df.drop_duplicates(subset=['title', 'description', 'genres'])
    return df