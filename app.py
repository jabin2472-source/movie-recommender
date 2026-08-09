from flask import Flask, render_template, request, jsonify
from recommendation import MovieRecommender

app = Flask(__name__)

# Load the TMDB 5000 dataset and build the recommendation model
recommender = MovieRecommender(
    "dataset/tmdb_5000_movies.csv",
    "dataset/tmdb_5000_credits.csv"
)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/recommend", methods=["GET"])
def recommend():
    movie = request.args.get("movie", "").strip()
    limit = request.args.get("limit", 30, type=int)

    if not movie:
        return jsonify({"error": "Please enter a movie name."}), 400

    results = recommender.recommend(movie, limit)
    return jsonify(results)

@app.route("/search", methods=["GET"])
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    return jsonify(recommender.search(query, 20))

@app.route("/genre", methods=["GET"])
def genre():
    name = request.args.get("name", "").strip()
    limit = request.args.get("limit", 30, type=int)

    if not name:
        return jsonify({"error": "Please provide a genre."}), 400

    results = recommender.get_by_genre(name, limit)
    return jsonify(results)

if __name__ == "__main__":
    app.run(debug=True)
