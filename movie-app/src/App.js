import React, { useState } from "react";
import "./App.css";

const OMDB_API_KEY = process.env.REACT_APP_OMDB_API_KEY;
const API_URL =
  process.env.REACT_APP_RECOMMEND_API_URL ||
  "http://127.0.0.1:8000/recommend";

function App() {
  const [title, setTitle] = useState("");
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [baseTitle, setBaseTitle] = useState(null);
  const [movies, setMovies] = useState([]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    console.log("[App] handleSubmit", {title, topK, API_URL});
    setError(null);

    if (!title.trim()) {
      setError("Please enter a movie title.");
      return;
    }

    const safeTopK = Number(topK) || 1;

    setLoading(true);
    setMovies([]);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ title: title.trim(), top_k: safeTopK }),
      });

      if (!response.ok) {
        throw new Error(
          `Backend returned status ${response.status}. Check if the API is running.`
        );
      }

      const data = await response.json();

      if (!data || !Array.isArray(data.recommendations)) {
        throw new Error("Unexpected response format from recommendation API.");
      }

      setBaseTitle(data.title || title.trim());

      const movieInfos = await Promise.all(
        data.recommendations.map((recTitle) => fetchMovieInfo(recTitle))
      );

      setMovies(movieInfos);
    } catch (err) {
      console.error(err);
      setError(err.message || "Something went wrong.");
    } finally {
      setLoading(false);
    }
  };


  const fetchMovieInfo = async (rawTitle) => {
      if (!OMDB_API_KEY) {
        return {
          id: rawTitle,
          title: rawTitle,
          year: "",
          poster: "",
          rating: "",
          plot: "",
        };
      }

      const { queryTitle, yearParam } = parseTitle(rawTitle);

      const url = new URL("https://www.omdbapi.com/");
      url.searchParams.set("t", queryTitle);
      if (yearParam) {
        url.searchParams.set("y", yearParam);
      }
      url.searchParams.set("apikey", OMDB_API_KEY);

      try {
        const res = await fetch(url.toString());
        const json = await res.json();

        if (json.Response === "False") {
          return {
            id: rawTitle,
            title: rawTitle,
            year: "",
            poster: "",
            rating: "",
            plot: "",
          };
        }

        return {
          id: rawTitle,
          title: json.Title || rawTitle,
          year: json.Year || "",
          poster: json.Poster && json.Poster !== "N/A" ? json.Poster : "",
          rating: json.imdbRating && json.imdbRating !== "N/A" ? json.imdbRating : "",
          plot: json.Plot && json.Plot !== "N/A" ? json.Plot : "",
        };
      } catch (err) {
        console.error("OMDb fetch error:", err);
        return {
          id: rawTitle,
          title: rawTitle,
          year: "",
          poster: "",
          rating: "",
          plot: "",
        };
      }
    };

    const parseTitle = (raw) => {
      const match = raw.match(/(.+)\s\((\d{4})\)$/);
      if (match) {
        const namePart = match[1].trim();
        const year = match[2];

        return {
          queryTitle: namePart,
          yearParam: year,
        };
      }
      return {
        queryTitle: raw.trim(),
        yearParam: "",
      };
    };
    return (
      <div className="app-root">
      <header className="app-header">
        <div className="logo">CINEMATCH</div>
        <div className="header-tagline">
          Movie recommendations powered by AI
        </div>
      </header>

      <main className="app-main">
        <section className="hero">
          <div className="hero-content">
            <h1 className="hero-title">Find your next favorite movie</h1>
            <p className="hero-subtitle">
              Type a movie title, choose how many recommendations you want and let
              the AI suggest similar films.
            </p>

            <form className="search-form" onSubmit={handleSubmit}>
              <div className="form-row">
                <label className="field">
                  <span className="field-label">Movie title</span>
                  <input
                    type="text"
                    className="field-input"
                    placeholder="Toy Story"
                    value={title}
                    onChange={(e) => setTitle(e.target.value)}
                  />
                </label>

                <label className="field field-small">
                  <span className="field-label">Top Movies</span>
                  <input
                    type="number"
                    min="1"
                    max="20"
                    className="field-input"
                    value={topK}
                    onChange={(e) => setTopK(e.target.value)}
                  />
                </label>
              </div>

              <button className="search-button" type="submit" disabled={loading}>
                {loading ? "Finding movies..." : "Get recommendations"}
              </button>

              {error && <div className="error-banner">{error}</div>}
              {!OMDB_API_KEY && (
                <div className="warning-banner">
                  OMDb API key is not set. Posters and extra info will not be shown.
                </div>
              )}
            </form>
          </div>
        </section>

        <section className="results-section">
          {loading && (
            <div className="loading-row">
              <div className="spinner" />
              <span>Loading recommendations...</span>
            </div>
          )}

          {!loading && baseTitle && (
            <>
              <h2 className="results-title">
                Recommendations for <span className="highlight">{baseTitle}</span>
              </h2>

              {movies.length === 0 ? (
                <p className="no-results">No recommendations found.</p>
              ) : (
                <div className="movies-grid">
                  {movies.map((movie) => (
                    <MovieCard key={movie.id} movie={movie} />
                  ))}
                </div>
              )}
            </>
          )}
        </section>
      </main>
    </div>
  );
}

function MovieCard({ movie }) {
  const { title, year, poster, rating, plot } = movie;

  return (
    <article className="movie-card">
      <div className="movie-poster-wrapper">
        {poster ? (
          <img src={poster} alt={title} className="movie-poster" loading="lazy" />
        ) : (
          <div className="movie-poster placeholder">
            <span className="placeholder-text">No poster</span>
          </div>
        )}

        <div className="movie-overlay">
          <h3 className="movie-title">{title}</h3>
          <div className="movie-meta">
            {year && <span className="chip">{year}</span>}
            {rating && <span className="chip rating">IMDb {rating}</span>}
          </div>
          {plot && <p className="movie-plot">{plot}</p>}
        </div>
      </div>
    </article>
  );
}

export default App;


   