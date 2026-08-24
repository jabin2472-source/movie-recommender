import ast
import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class MovieRecommender:
    def __init__(self, movies_file, credits_file, tamil_movies_file=None,
                 imdb_movies_file=None, extra_movies_file=None):
        print("Loading TMDB dataset...")

        movies = pd.read_csv(movies_file)
        credits = pd.read_csv(credits_file)

        # The two TMDB files share the title column.
        tmdb = movies.merge(credits, on="title")

        # Keep useful columns.
        tmdb = tmdb[
            ["title", "genres", "keywords", "overview", "cast", "crew",
             "vote_average", "release_date", "popularity"]
        ].copy()

        tmdb["title"] = tmdb["title"].fillna("")
        tmdb["overview"] = tmdb["overview"].fillna("")

        # TMDB's own dataset stores the film's original language as a short
        # code (e.g. "en", "hi", "ta") in the "original_language" column.
        # We only care about two buckets here: Tamil, and everything else
        # bucketed as English (matches the Tamil dataset we merge in below).
        if "original_language" in movies.columns:
            code_map = movies.drop_duplicates(subset="title").set_index("title")["original_language"]
            codes = tmdb["title"].map(code_map).fillna("en").astype(str).str.lower()
        else:
            codes = pd.Series(["en"] * len(tmdb), index=tmdb.index)

        tmdb["language"] = codes.apply(lambda c: "Tamil" if c == "ta" else "English")

        # Convert TMDB JSON-like columns into searchable text.
        tmdb["genres_text"] = tmdb["genres"].apply(self.get_names)
        tmdb["keywords_text"] = tmdb["keywords"].apply(self.get_names)
        tmdb["cast_text"] = tmdb["cast"].apply(self.get_cast)
        tmdb["crew_text"] = tmdb["crew"].apply(self.get_director)

        tmdb["poster"] = ""
        tmdb["duration"] = ""

        tmdb = tmdb[
            ["title", "genres_text", "keywords_text", "cast_text", "crew_text",
             "overview", "vote_average", "release_date", "popularity", "language",
             "poster", "duration"]
        ]

        frames = [tmdb]

        if tamil_movies_file:
            print("Loading Tamil movies dataset...")
            tamil = pd.read_csv(tamil_movies_file)

            years = pd.to_numeric(tamil["Year"], errors="coerce").fillna(0).astype(int)

            tamil_df = pd.DataFrame({
                "title": tamil["MovieName"].fillna(""),
                "genres_text": tamil["Genre"].fillna("").apply(
                    lambda g: str(g).replace(" ", "")
                ),
                "keywords_text": "",
                "cast_text": tamil["Actor"].fillna("").apply(
                    lambda a: str(a).replace(" ", "")
                ),
                "crew_text": tamil["Director"].fillna("").apply(
                    lambda d: str(d).replace(" ", "")
                ),
                "overview": "",
                "vote_average": pd.to_numeric(tamil["Rating"], errors="coerce").fillna(0),
                "release_date": years.astype(str) + "-01-01",
                "popularity": pd.to_numeric(tamil["PeopleVote"], errors="coerce").fillna(0),
                "language": "Tamil",
                "poster": "",
                "duration": "",
            })

            frames.append(tamil_df)
            print(f"Loaded {len(tamil_df)} Tamil movies.")

        if imdb_movies_file:
            print("Loading Tamil (IMDb) movies dataset...")
            imdb = pd.read_csv(imdb_movies_file)

            # Votes look like "28,737" or "595" (and some are blank) — strip commas.
            votes = imdb["Votes"].astype(str).str.replace(",", "", regex=False)
            votes = pd.to_numeric(votes, errors="coerce").fillna(0)

            # Titles are numbered ("1. Thunivu") — strip the leading "N. ".
            clean_titles = imdb["Movie Name"].fillna("").astype(str).str.replace(
                r"^\d+\.\s*", "", regex=True
            )

            imdb_df = pd.DataFrame({
                "title": clean_titles,
                "genres_text": "",
                "keywords_text": "",
                "cast_text": "",
                "crew_text": "",
                "overview": imdb["Description"].fillna(""),
                "vote_average": pd.to_numeric(imdb["IMdb rating"], errors="coerce").fillna(0),
                "release_date": "",
                "popularity": votes,
                "language": "Tamil",
                "poster": "",
                "duration": imdb["Duration"].fillna("") if "Duration" in imdb.columns else "",
            })

            frames.append(imdb_df)
            print(f"Loaded {len(imdb_df)} IMDb Tamil movies.")

        if extra_movies_file:
            print("Loading extra movies dataset...")
            extra = pd.read_csv(extra_movies_file)

            title_col = "title_x" if "title_x" in extra.columns else "title"
            years = pd.to_numeric(
                extra["year_of_release"], errors="coerce"
            ).fillna(0).astype(int) if "year_of_release" in extra.columns else 0

            genres_text = extra["genres"].fillna("").apply(
                lambda g: " ".join(part.strip().replace(" ", "") for part in str(g).split("|") if part.strip())
            ) if "genres" in extra.columns else ""

            cast_text = extra["actors"].fillna("").apply(
                lambda a: " ".join(part.strip().replace(" ", "") for part in str(a).split("|")[:5] if part.strip())
            ) if "actors" in extra.columns else ""

            overview = extra["story"].fillna(extra.get("summary", "")).fillna("") \
                if "story" in extra.columns else extra.get("summary", "").fillna("")

            duration = extra["runtime"].fillna("").astype(str).apply(
                lambda r: f"{r} min" if r.strip().isdigit() else ""
            ) if "runtime" in extra.columns else ""

            # This dataset has no language column, so we detect it from the
            # cast: it's built from Wikipedia's Hindi-cinema pages, so we
            # default to "Hindi" and only override to "Tamil" when a known
            # Tamil star appears in the cast. This is a heuristic, not
            # perfect — but there's no language field to go on otherwise.
            TAMIL_ACTORS = [
                "Rajinikanth", "Kamal Haasan", "Vijay", "Ajith Kumar", "Suriya",
                "Dhanush", "Vikram", "Sivakarthikeyan", "Simbu", "Silambarasan",
                "Karthi", "Jayam Ravi", "Arya", "Vishal", "Nayanthara", "Trisha",
                "Samantha", "Vijay Sethupathi", "Yogi Babu", "Prabhu Deva",
                "Sarath Kumar", "Jyothika", "Sneha", "Vadivelu", "Santhanam",
            ]
            actors_raw = extra["actors"].fillna("") if "actors" in extra.columns else pd.Series([""] * len(extra))
            language = actors_raw.apply(
                lambda a: "Tamil" if any(t in str(a) for t in TAMIL_ACTORS) else "Hindi"
            )

            extra_df = pd.DataFrame({
                "title": extra[title_col].fillna(""),
                "genres_text": genres_text,
                "keywords_text": "",
                "cast_text": cast_text,
                "crew_text": "",
                "overview": overview,
                "vote_average": pd.to_numeric(extra.get("imdb_rating", 0), errors="coerce").fillna(0),
                "release_date": years.astype(str) + "-01-01" if "year_of_release" in extra.columns else "",
                "popularity": pd.to_numeric(extra.get("imdb_votes", 0), errors="coerce").fillna(0),
                "language": language,
                "poster": extra["poster_path"].fillna("") if "poster_path" in extra.columns else "",
                "duration": duration,
            })

            # Put this frame right after the TMDB frame (before the older
            # Tamil sources, which don't carry posters) — when a title
            # collides across sources, drop_duplicates keeps whichever
            # row appears FIRST, so this order lets the poster-bearing
            # row win instead of silently losing its poster.
            frames.insert(1, extra_df)
            print(f"Loaded {len(extra_df)} extra movies "
                  f"({(language == 'Tamil').sum()} detected Tamil, "
                  f"{(language == 'Hindi').sum()} detected Hindi).")

        # Combine every dataset into one searchable/recommendable pool.
        self.movies = pd.concat(frames, ignore_index=True)
        self.movies = self.movies.drop_duplicates(subset="title")
        # Reset the index so row labels line up 1:1 with the similarity
        # matrix's row positions (dropping duplicates leaves gaps otherwise).
        self.movies = self.movies.reset_index(drop=True)

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

        print(f"Loaded {len(self.movies)} movies total.")
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

        poster = row.get("poster", "")
        if pd.isna(poster):
            poster = ""

        duration = row.get("duration", "")
        if pd.isna(duration):
            duration = ""

        return {
            "title": row["title"],
            "year": year,
            "rating": round(float(row["vote_average"]), 1),
            "overview": row["overview"],
            "genres": row["genres_text"].replace(" ", ", "),
            "popularity": float(row["popularity"]),
            "language": row.get("language", "English"),
            "poster": poster,
            "duration": duration
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

    def get_by_language(self, language_name, limit=20):
        """Return top movies (by popularity) matching a given language, e.g. 'Tamil'."""
        query = language_name.strip().lower()

        matches = self.movies[
            self.movies["language"].str.lower() == query
        ].sort_values("popularity", ascending=False).head(limit)

        return [self.format_movie(row) for _, row in matches.iterrows()]

    def list_languages(self):
        """Return the distinct languages present in the dataset, most common first."""
        counts = self.movies["language"].value_counts()
        return list(counts.index)
