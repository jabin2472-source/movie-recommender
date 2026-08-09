import ast
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movies_file, credits_file):
        print("Loading TMDB datasets...")

        movies = pd.read_csv(movies_file)
        credits = pd.read_csv(credits_file)

        # The two TMDB files share the title column.
        self.movies = movies.merge(credits, on="title")

        # Keep useful columns and remove duplicate/empty rows.
        self.movies = self.movies[
            ["title", "genres", "keywords", "overview", "cast", "crew",
             "vote_average", "release_date", "popularity"]
        ].copy()

        self.movies = self.movies.drop_duplicates(subset="title")
        self.movies["title"] = self.movies["title"].fillna("")
        self.movies["overview"] = self.movies["overview"].fillna("")

        # Convert TMDB JSON-like columns into searchable text.
        self.movies["genres_text"] = self.movies["genres"].apply(self.get_names)
        self.movies["keywords_text"] = self.movies["keywords"].apply(self.get_names)
        self.movies["cast_text"] = self.movies["cast"].apply(self.get_cast)
        self.movies["crew_text"] = self.movies["crew"].apply(self.get_director)

        self.movies["tags"] = (
            self.movies["genres_text"] + " " +
            self.movies["keywords_text"] + " " +
            self.movies["cast_text"] + " " +
            self.movies["crew_text"] + " " +
            self.movies["overview"]
        )

        self.movies["tags"] = self.movies["tags"].str.lower()

        # Convert movie text into vectors and calculate similarity.
        self.vectorizer = CountVectorizer(
            max_features=5000,
            stop_words="english"
        )
        vectors = self.vectorizer.fit_transform(self.movies["tags"])
        self.similarity = cosine_similarity(vectors)

        self.title_to_index = {
            title.lower(): index
            for index, title in self.movies["title"].items()
        }

        print(f"Loaded {len(self.movies)} movies.")
        print("Recommendation model is ready.")

    @staticmethod
    def get_names(value):
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
            return " ".join(
                item.get("name", "").replace(" ", "")
                for item in items
                if isinstance(item, dict)
            )
        except (ValueError, SyntaxError):
            return ""

    @staticmethod
    def get_cast(value):
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
            return " ".join(
                item.get("name", "").replace(" ", "")
                for item in items[:5]
                if isinstance(item, dict)
            )
        except (ValueError, SyntaxError):
            return ""

    @staticmethod
    def get_director(value):
        try:
            items = ast.literal_eval(value) if isinstance(value, str) else []
            for item in items:
                if isinstance(item, dict) and item.get("job") == "Director":
                    return item.get("name", "").replace(" ", "")
            return ""
        except (ValueError, SyntaxError):
            return ""

    def format_movie(self, row):
        release = row.get("release_date", "")
        year = ""
        if isinstance(release, str) and len(release) >= 4:
            year = release[:4]

        return {
            "title": row["title"],
            "year": year,
            "rating": round(float(row["vote_average"]), 1),
            "overview": row["overview"],
            "genres": row["genres_text"].replace(" ", ", "),
            "popularity": float(row["popularity"])
        }

    def recommend(self, movie_name, limit=10):
        query = movie_name.lower()

        # Exact title first.
        index = self.title_to_index.get(query)

        # If exact title isn't found, use the closest title containing the query.
        if index is None:
            matches = [
                (title, i)
                for title, i in self.title_to_index.items()
                if query in title
            ]
            if not matches:
                return []
            index = matches[0][1]

        scores = list(enumerate(self.similarity[index]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)

        results = []
        for movie_index, score in scores:
            if movie_index == index:
                continue

            row = self.movies.iloc[movie_index]
            movie = self.format_movie(row)
            movie["similarity"] = round(float(score), 3)
            results.append(movie)

            if len(results) >= limit:
                break

        return results

    def search(self, query, limit=20):
        query = query.lower()

        matches = self.movies[
            self.movies["title"].str.lower().str.contains(query, na=False)
        ].head(limit)

        return [self.format_movie(row) for _, row in matches.iterrows()]

    def get_by_genre(self, genre_name, limit=20):
        """Return top movies (by popularity) whose genres match genre_name.

        TMDB genre names are stored with spaces stripped (e.g. "Science Fiction"
        becomes "ScienceFiction"), so we strip spaces from the query too before
        matching, and search case-insensitively.
        """
        query = genre_name.replace(" ", "").lower()

        matches = self.movies[
            self.movies["genres_text"].str.lower().str.contains(query, na=False)
        ].sort_values("popularity", ascending=False).head(limit)

        return [self.format_movie(row) for _, row in matches.iterrows()]

    def list_genres(self, top_n=None):
        """Return the distinct genre names present in the dataset, most common first."""
        from collections import Counter

        counts = Counter()
        for text in self.movies["genres_text"]:
            # genres_text is space-joined, each token is one genre name (spaces stripped)
            for token in text.split():
                counts[token] += 1

        ordered = [name for name, _ in counts.most_common(top_n)]
        return ordered
