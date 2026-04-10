import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FlixSense — Film Recommendation Engine",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;700;900&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --bg: #0a0a0f;
    --surface: #13131a;
    --surface2: #1c1c28;
    --accent: #e8b84b;
    --accent2: #c0392b;
    --text: #f0ece4;
    --muted: #7a7a8c;
    --border: #2a2a3a;
}

html, body, [data-testid="stAppViewContainer"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text) !important;
}

h1, h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--text) !important;
}

.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3.5rem;
    font-weight: 900;
    letter-spacing: -1px;
    line-height: 1;
    background: linear-gradient(135deg, #f0ece4 0%, #e8b84b 60%, #c0392b 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.hero-sub {
    color: var(--muted);
    font-size: 1rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    font-weight: 300;
}

.movie-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
    transition: transform 0.35s ease, box-shadow 0.35s ease, border-color 0.35s ease;
    height: 100%;
    position: relative;
}


.movie-card:hover {
    transform: scale(1.12) translateY(-10px);
    border-color: var(--accent);
    z-index: 20;
    box-shadow: 0 25px 50px rgba(0, 0, 0, 0.6);
}

.movie-poster {
    width: 100%;
    aspect-ratio: 2/3;
    object-fit: cover;
    background: var(--surface2);
}

.poster-placeholder {
    width: 100%;
    aspect-ratio: 2/3;
    background: linear-gradient(135deg, var(--surface2), var(--border));
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 3rem;
}

.movie-info {
    padding: 12px;
}

.movie-title-card {
    font-family: 'Playfair Display', serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text);
    margin-bottom: 4px;
    line-height: 1.3;
}

.movie-meta {
    font-size: 0.75rem;
    color: var(--muted);
    display: flex;
    gap: 8px;
    align-items: center;
    flex-wrap: wrap;
}

.badge {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 0.7rem;
    color: var(--muted);
    letter-spacing: 0.05em;
}

.badge-accent {
    background: rgba(232,184,75,0.15);
    border-color: rgba(232,184,75,0.4);
    color: var(--accent);
}

.score-bar-wrap {
    margin-top: 8px;
}

.score-label {
    font-size: 0.68rem;
    color: var(--muted);
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 3px;
}

.score-bar-bg {
    height: 3px;
    background: var(--border);
    border-radius: 2px;
    overflow: hidden;
}

.score-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, var(--accent2), var(--accent));
    border-radius: 2px;
}

.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text);
    border-left: 3px solid var(--accent);
    padding-left: 12px;
    margin-bottom: 1rem;
}

.stat-box {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}

.stat-number {
    font-family: 'Playfair Display', serif;
    font-size: 2rem;
    font-weight: 900;
    color: var(--accent);
}

.stat-label {
    font-size: 0.75rem;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

.stSelectbox > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stTextInput > div > div > input {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}

.stSlider > div > div > div > div {
    background: var(--accent) !important;
}

.stButton > button {
    background: linear-gradient(135deg, var(--accent2), var(--accent)) !important;
    color: #0a0a0f !important;
    border: none !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    border-radius: 8px !important;
    padding: 0.5rem 1.5rem !important;
}

.stButton > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px);
}

.stMultiSelect > div > div {
    background: var(--surface2) !important;
    border-color: var(--border) !important;
}

div[data-testid="stMetricValue"] {
    color: var(--accent) !important;
    font-family: 'Playfair Display', serif !important;
}

.genre-pill {
    display: inline-block;
    background: rgba(232,184,75,0.1);
    border: 1px solid rgba(232,184,75,0.3);
    color: var(--accent);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.7rem;
    margin: 2px;
    letter-spacing: 0.05em;
}

hr { border-color: var(--border) !important; }

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Dataset ───────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    movies = [
        {"id": "tt0111161", "title": "The Shawshank Redemption", "year": 1994, "genres": "Drama", "rating": 9.3, "votes": 2800000, "director": "Frank Darabont", "overview": "Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency. A story of hope, friendship, and resilience inside the brutal world of prison."},
        {"id": "tt0068646", "title": "The Godfather", "year": 1972, "genres": "Crime,Drama", "rating": 9.2, "votes": 1950000, "director": "Francis Ford Coppola", "overview": "The aging patriarch of an organized crime dynasty transfers control of his clandestine empire to his reluctant son. A sweeping tale of family, power, and betrayal."},
        {"id": "tt0071562", "title": "The Godfather Part II", "year": 1974, "genres": "Crime,Drama", "rating": 9.0, "votes": 1320000, "director": "Francis Ford Coppola", "overview": "The early life and career of Vito Corleone in 1920s New York City is portrayed while his son, Michael, expands and tightens his grip on the family crime syndicate."},
        {"id": "tt0468569", "title": "The Dark Knight", "year": 2008, "genres": "Action,Crime,Drama", "rating": 9.0, "votes": 2750000, "director": "Christopher Nolan", "overview": "When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice."},
        {"id": "tt0050083", "title": "12 Angry Men", "year": 1957, "genres": "Crime,Drama", "rating": 9.0, "votes": 800000, "director": "Sidney Lumet", "overview": "A jury holdout attempts to prevent a miscarriage of justice by forcing his colleagues to reconsider the evidence in a murder trial."},
        {"id": "tt0108052", "title": "Schindler's List", "year": 1993, "genres": "Biography,Drama,History", "rating": 8.9, "votes": 1380000, "director": "Steven Spielberg", "overview": "In German-occupied Poland during World War II, industrialist Oskar Schindler gradually becomes concerned for his Jewish workforce after witnessing their persecution by the Nazis."},
        {"id": "tt0167260", "title": "The Lord of the Rings: The Return of the King", "year": 2003, "genres": "Action,Adventure,Drama", "rating": 8.9, "votes": 1820000, "director": "Peter Jackson", "overview": "Gandalf and Aragorn lead the World of Men against Sauron's army to draw his gaze from Frodo and Sam as they approach Mount Doom with the One Ring."},
        {"id": "tt0110912", "title": "Pulp Fiction", "year": 1994, "genres": "Crime,Drama", "rating": 8.9, "votes": 2100000, "director": "Quentin Tarantino", "overview": "The lives of two mob hitmen, a boxer, a gangster and his wife, and a pair of diner bandits intertwine in four tales of violence and redemption."},
        {"id": "tt0060196", "title": "The Good, the Bad and the Ugly", "year": 1966, "genres": "Adventure,Western", "rating": 8.8, "votes": 760000, "director": "Sergio Leone", "overview": "A bounty hunting scam joins two men in an uneasy alliance against a third in a race to find a fortune in gold buried in a remote cemetery."},
        {"id": "tt0137523", "title": "Fight Club", "year": 1999, "genres": "Drama", "rating": 8.8, "votes": 2100000, "director": "David Fincher", "overview": "An insomniac office worker and a devil-may-care soap maker form an underground fight club that evolves into something much, much more dangerous."},
        {"id": "tt0120737", "title": "The Lord of the Rings: The Fellowship of the Ring", "year": 2001, "genres": "Action,Adventure,Drama", "rating": 8.8, "votes": 1870000, "director": "Peter Jackson", "overview": "A meek Hobbit from the Shire and eight companions set out on a journey to destroy the powerful One Ring and save Middle-earth from the Dark Lord Sauron."},
        {"id": "tt1375666", "title": "Inception", "year": 2010, "genres": "Action,Adventure,Sci-Fi", "rating": 8.8, "votes": 2350000, "director": "Christopher Nolan", "overview": "A thief who steals corporate secrets through the use of dream-sharing technology is given the inverse task of planting an idea into the mind of a C.E.O."},
        {"id": "tt0167261", "title": "The Lord of the Rings: The Two Towers", "year": 2002, "genres": "Action,Adventure,Drama", "rating": 8.7, "votes": 1620000, "director": "Peter Jackson", "overview": "While Frodo and Sam edge closer to Mordor with the help of the shifty Gollum, the divided fellowship makes a stand against Sauron's new ally, Saruman, and his hordes of Isengard."},
        {"id": "tt0080684", "title": "Star Wars: The Empire Strikes Back", "year": 1980, "genres": "Action,Adventure,Fantasy", "rating": 8.7, "votes": 1310000, "director": "Irvin Kershner", "overview": "After the Rebels are brutally overpowered by the Empire on the ice planet Hoth, Luke Skywalker begins Jedi training with Yoda. His friends are pursued by Darth Vader and bounty hunter Boba Fett."},
        {"id": "tt0133093", "title": "The Matrix", "year": 1999, "genres": "Action,Sci-Fi", "rating": 8.7, "votes": 1940000, "director": "The Wachowskis", "overview": "A computer hacker learns from mysterious rebels about the true nature of his reality and his role in the war against its controllers."},
        {"id": "tt0073486", "title": "One Flew Over the Cuckoo's Nest", "year": 1975, "genres": "Drama", "rating": 8.7, "votes": 970000, "director": "Milos Forman", "overview": "A criminal pleads insanity and is admitted to a mental institution, where he rebels against the oppressive nurse and rallies up the other patients."},
        {"id": "tt0099685", "title": "Goodfellas", "year": 1990, "genres": "Biography,Crime,Drama", "rating": 8.7, "votes": 1130000, "director": "Martin Scorsese", "overview": "The story of Henry Hill and his life in the mob, covering his relationship with his wife Karen Hill and his mob partners Jimmy Conway and Tommy DeVito."},
        {"id": "tt0047478", "title": "Seven Samurai", "year": 1954, "genres": "Action,Adventure,Drama", "rating": 8.6, "votes": 530000, "director": "Akira Kurosawa", "overview": "A poor village under attack by bandits recruits seven samurai to help them defend themselves."},
        {"id": "tt0317248", "title": "City of God", "year": 2002, "genres": "Crime,Drama", "rating": 8.6, "votes": 780000, "director": "Fernando Meirelles", "overview": "In the slums of Rio, two kids' paths diverge as one struggles to become a photographer and the other a kingpin."},
        {"id": "tt0038650", "title": "It's a Wonderful Life", "year": 1946, "genres": "Drama,Family,Fantasy", "rating": 8.6, "votes": 440000, "director": "Frank Capra", "overview": "An angel is sent from Heaven to help a desperately frustrated businessman by showing him what life would have been like if he had never existed."},
        {"id": "tt0245429", "title": "Spirited Away", "year": 2001, "genres": "Animation,Adventure,Family", "rating": 8.6, "votes": 750000, "director": "Hayao Miyazaki", "overview": "During her family's move to the suburbs, a sullen 10-year-old girl wanders into a world ruled by gods, witches, and spirits."},
        {"id": "tt0114369", "title": "Se7en", "year": 1995, "genres": "Crime,Drama,Mystery", "rating": 8.6, "votes": 1540000, "director": "David Fincher", "overview": "Two detectives, a rookie and a veteran, hunt a serial killer who uses the seven deadly sins as his motives in a gloomy, crime-ridden city."},
        {"id": "tt0102926", "title": "The Silence of the Lambs", "year": 1991, "genres": "Crime,Drama,Thriller", "rating": 8.6, "votes": 1460000, "director": "Jonathan Demme", "overview": "A young F.B.I. cadet must receive the help of an incarcerated and manipulative cannibal killer to help catch another serial killer."},
        {"id": "tt0172495", "title": "Gladiator", "year": 2000, "genres": "Action,Adventure,Drama", "rating": 8.5, "votes": 1460000, "director": "Ridley Scott", "overview": "A former Roman General sets out to exact vengeance against the corrupt emperor who murdered his family and sent him into slavery."},
        {"id": "tt0110413", "title": "Léon: The Professional", "year": 1994, "genres": "Action,Crime,Drama", "rating": 8.5, "votes": 1130000, "director": "Luc Besson", "overview": "12-year-old Mathilda is reluctantly taken in by Léon, a professional assassin, after her family is murdered. An unusual relationship forms between them."},
        {"id": "tt0482571", "title": "The Prestige", "year": 2006, "genres": "Drama,Mystery,Sci-Fi", "rating": 8.5, "votes": 1320000, "director": "Christopher Nolan", "overview": "After a tragic accident, two stage magicians engage in a battle to create the ultimate illusion while sacrificing everything they have to outwit each other."},
        {"id": "tt0816692", "title": "Interstellar", "year": 2014, "genres": "Adventure,Drama,Sci-Fi", "rating": 8.6, "votes": 1840000, "director": "Christopher Nolan", "overview": "A team of explorers travel through a wormhole in space in an attempt to ensure humanity's survival as Earth becomes uninhabitable."},
        {"id": "tt0078748", "title": "Alien", "year": 1979, "genres": "Horror,Sci-Fi", "rating": 8.4, "votes": 890000, "director": "Ridley Scott", "overview": "After a space merchant vessel receives an unknown transmission as a distress call, one of the crew is attacked by a mysterious lifeform and they soon realize that its life cycle has merely begun."},
        {"id": "tt0364569", "title": "Oldboy", "year": 2003, "genres": "Action,Drama,Mystery", "rating": 8.4, "votes": 580000, "director": "Chan-wook Park", "overview": "After being kidnapped and imprisoned for fifteen years, Oh Dae-Su is released, only to find that he must find his captor in five days."},
        {"id": "tt0407887", "title": "The Departed", "year": 2006, "genres": "Crime,Drama,Thriller", "rating": 8.5, "votes": 1340000, "director": "Martin Scorsese", "overview": "An undercover cop and a mole in the police attempt to identify each other while simultaneously infiltrating an Irish gang in South Boston."},
        {"id": "tt0253474", "title": "The Pianist", "year": 2002, "genres": "Biography,Drama,Music", "rating": 8.5, "votes": 780000, "director": "Roman Polanski", "overview": "A Polish Jewish musician struggles to survive the destruction of the Warsaw ghetto of World War II."},
        {"id": "tt0211915", "title": "Amélie", "year": 2001, "genres": "Comedy,Romance", "rating": 8.3, "votes": 740000, "director": "Jean-Pierre Jeunet", "overview": "Amélie is an innocent and naive girl in Paris with her own sense of justice, decides to help those around her and along the way, discovers love."},
        {"id": "tt0120689", "title": "The Green Mile", "year": 1999, "genres": "Crime,Drama,Fantasy", "rating": 8.6, "votes": 1310000, "director": "Frank Darabont", "overview": "The lives of guards on Death Row are affected by one of their charges: a black man accused of child murder and rape, yet who has a mysterious gift."},
        {"id": "tt1345836", "title": "The Dark Knight Rises", "year": 2012, "genres": "Action,Adventure,Thriller", "rating": 8.4, "votes": 1620000, "director": "Christopher Nolan", "overview": "Eight years after the Joker's reign of anarchy, Batman is forced from his exile to save Gotham City from the brutal guerrilla terrorist Bane."},
        {"id": "tt0027977", "title": "Modern Times", "year": 1936, "genres": "Comedy,Drama,Family", "rating": 8.5, "votes": 245000, "director": "Charles Chaplin", "overview": "The Tramp struggles to live in modern industrial society with the help of a young homeless woman."},
        {"id": "tt0056058", "title": "Harakiri", "year": 1962, "genres": "Action,Drama", "rating": 8.7, "votes": 63000, "director": "Masaki Kobayashi", "overview": "An elder samurai inquires to commit ritual suicide at a feudal lord's palace, claiming poverty. But the lord's senior counsellor has a hidden motive for his acceptance."},
        {"id": "tt1130884", "title": "Shutter Island", "year": 2010, "genres": "Mystery,Thriller", "rating": 8.2, "votes": 1210000, "director": "Martin Scorsese", "overview": "In 1954, a U.S. Marshal investigates the disappearance of a murderer who escaped from a hospital for the criminally insane."},
        {"id": "tt0054215", "title": "Psycho", "year": 1960, "genres": "Horror,Mystery,Thriller", "rating": 8.5, "votes": 660000, "director": "Alfred Hitchcock", "overview": "A secretary on the run from her boss ends up at the secluded Bates Motel, managed by a young man under his mother's domineering influence."},
        {"id": "tt0435761", "title": "Toy Story 3", "year": 2010, "genres": "Animation,Adventure,Comedy", "rating": 8.3, "votes": 860000, "director": "Lee Unkrich", "overview": "The toys are mistakenly delivered to a day-care center instead of the attic right before Andy leaves for college, and it's up to Woody to convince the other toys that they weren't abandoned."},
        {"id": "tt0209144", "title": "Memento", "year": 2000, "genres": "Mystery,Thriller", "rating": 8.4, "votes": 1170000, "director": "Christopher Nolan", "overview": "A man with short-term memory loss attempts to track down his wife's murderer. He uses notes and tattoos to help him remember."},
        {"id": "tt0338013", "title": "Eternal Sunshine of the Spotless Mind", "year": 2004, "genres": "Drama,Romance,Sci-Fi", "rating": 8.3, "votes": 1050000, "director": "Michel Gondry", "overview": "When their relationship turns sour, a couple undergoes a medical procedure to have each other erased from their memories."},
        {"id": "tt0266543", "title": "Finding Nemo", "year": 2003, "genres": "Animation,Adventure,Comedy", "rating": 8.1, "votes": 1040000, "director": "Andrew Stanton", "overview": "After his son is taken by a scuba diver, an overprotective clownfish embarks on a journey across the ocean with a regal blue tang who has short-term memory loss."},
        {"id": "tt0118799", "title": "Life is Beautiful", "year": 1997, "genres": "Comedy,Drama,Romance", "rating": 8.6, "votes": 700000, "director": "Roberto Benigni", "overview": "When an open-spirited Jewish waiter and his son become victims of the Holocaust, he uses a perfect mixture of will, humor, and imagination to protect his son from the dangerous realities around them."},
        {"id": "tt0361748", "title": "Inglourious Basterds", "year": 2009, "genres": "Adventure,Drama,War", "rating": 8.3, "votes": 1380000, "director": "Quentin Tarantino", "overview": "In Nazi-occupied France during World War II, a plan to assassinate Nazi leaders by a group of Jewish U.S. soldiers coincides with a theatre owner's revenge plot."},
        {"id": "tt0405094", "title": "The Lives of Others", "year": 2006, "genres": "Drama,Mystery,Thriller", "rating": 8.4, "votes": 270000, "director": "Florian Henckel von Donnersmarck", "overview": "In 1984 East Berlin, an agent of the secret police conducts surveillance on a writer and his lover, but his work slowly becomes entangled with theirs."},
        {"id": "tt1187043", "title": "3 Idiots", "year": 2009, "genres": "Comedy,Drama", "rating": 8.4, "votes": 430000, "director": "Rajkumar Hirani", "overview": "Two friends are searching for their long lost companion. They revisit their college days and recall the memory of their friend who inspired them to think differently."},
        {"id": "tt2582802", "title": "Whiplash", "year": 2014, "genres": "Drama,Music", "rating": 8.5, "votes": 870000, "director": "Damien Chazelle", "overview": "A promising young drummer enrolls at a cut-throat music conservatory where his dreams of greatness are mentored by an instructor who will stop at nothing to realize a student's potential."},
        {"id": "tt0816711", "title": "Parasite", "year": 2019, "genres": "Comedy,Drama,Thriller", "rating": 8.5, "votes": 770000, "director": "Bong Joon-ho", "overview": "Greed and class discrimination threaten the newly formed symbiotic relationship between the wealthy Park family and the destitute Kim clan."},
        {"id": "tt6751668", "title": "Joker", "year": 2019, "genres": "Crime,Drama,Thriller", "rating": 8.4, "votes": 1110000, "director": "Todd Phillips", "overview": "In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime."},
        {"id": "tt4154796", "title": "Avengers: Endgame", "year": 2019, "genres": "Action,Adventure,Drama", "rating": 8.4, "votes": 1120000, "director": "Anthony and Joe Russo", "overview": "After the devastating events of Infinity War, the universe is in ruins. With the help of remaining allies, the Avengers assemble once more in order to reverse Thanos' actions and restore balance."},
        {"id": "tt1853728", "title": "Django Unchained", "year": 2012, "genres": "Drama,Western", "rating": 8.4, "votes": 1400000, "director": "Quentin Tarantino", "overview": "With the help of a German bounty hunter, a freed slave sets out to rescue his wife from a brutal Mississippi plantation owner."},
        {"id": "tt0120586", "title": "American History X", "year": 1998, "genres": "Crime,Drama", "rating": 8.5, "votes": 1080000, "director": "Tony Kaye", "overview": "A former neo-nazi skinhead tries to prevent his younger brother from going down the same wrong path that he did."},
    ]
    df = pd.DataFrame(movies)
    df['genres_list'] = df['genres'].apply(lambda x: x.split(','))
    return df

@st.cache_resource
def build_model(df):
    df = df.copy()
    # Build feature text
    df['features'] = (
        df['genres'].str.replace(',', ' ') + ' ' +
        df['director'].fillna('') + ' ' +
        df['overview'].fillna('')
    )
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
    tfidf_matrix = tfidf.fit_transform(df['features'])
    cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    return cos_sim

TMDB_KEY = "8265bd1679663a7ea12ac168da84d2e8"  # public demo key

@st.cache_data(ttl=3600)
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/find/{movie_id}?api_key={TMDB_KEY}&external_source=imdb_id"
        r = requests.get(url, timeout=4)
        data = r.json()
        results = data.get('movie_results', [])
        if results and results[0].get('poster_path'):
            return f"https://image.tmdb.org/t/p/w500{results[0]['poster_path']}"
    except:
        pass
    return None

def get_all_genres(df):
    all_genres = set()
    for gl in df['genres_list']:
        all_genres.update(gl)
    return sorted(all_genres)

def recommend(movie_title, df, cos_sim, n=8, genre_filter=None, year_range=None, min_rating=0):
    idx_list = df.index[df['title'] == movie_title].tolist()
    if not idx_list:
        return pd.DataFrame()
    idx = idx_list[0]
    sim_scores = list(enumerate(cos_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = [(i, s) for i, s in sim_scores if i != idx]

    result_rows = []
    for i, score in sim_scores:
        row = df.iloc[i].copy()
        row['similarity'] = round(score * 100, 1)
        result_rows.append(row)

    rec_df = pd.DataFrame(result_rows)
    if rec_df.empty:
        return rec_df

    if genre_filter:
        rec_df = rec_df[rec_df['genres_list'].apply(lambda gl: any(g in gl for g in genre_filter))]
    if year_range:
        rec_df = rec_df[(rec_df['year'] >= year_range[0]) & (rec_df['year'] <= year_range[1])]
    rec_df = rec_df[rec_df['rating'] >= min_rating]
    return rec_df.head(n).reset_index(drop=True)

def movie_card_html(title, year, genres, rating, votes, similarity, poster_url):
    poster_html = (
        f'<img class="movie-poster" src="{poster_url}" alt="{title}">'
        if poster_url else
        f'<div class="poster-placeholder">🎬</div>'
    )
    genres_pills = ''.join([f'<span class="genre-pill">{g.strip()}</span>' for g in genres.split(',')[:2]])
    score_pct = min(similarity, 100)
    stars = '★' * int(round(rating / 2))
    votes_k = f"{votes//1000}K" if votes >= 1000 else str(votes)
    return f"""
    <div class="movie-card">
        {poster_html}
        <div class="movie-info">
            <div class="movie-title-card">{title}</div>
            <div class="movie-meta">
                <span>{year}</span>
                <span style="color:#e8b84b">{stars}</span>
                <span>{rating}/10</span>
                <span class="badge">{votes_k} votes</span>
            </div>
            <div style="margin-top:6px">{genres_pills}</div>
            <div class="score-bar-wrap">
                <div class="score-label">Match Score — {similarity:.0f}%</div>
                <div class="score-bar-bg"><div class="score-bar-fill" style="width:{score_pct}%"></div></div>
            </div>
        </div>
    </div>
    """

# ── Load Data ─────────────────────────────────────────────────────────────────
df = load_data()
cos_sim = build_model(df)
all_genres = get_all_genres(df)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title" style="font-size:2rem">🎬<br>Flix Verse</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub" style="margin-bottom:1.5rem">Powered by TF-IDF · Cosine Similarity</div>', unsafe_allow_html=True)
    st.divider()

    st.markdown("### 🔍 Search Movie")
    search_query = st.text_input("", placeholder="Search titles…", label_visibility="collapsed")

    if search_query:
        matches = df[df['title'].str.contains(search_query, case=False, na=False)]['title'].tolist()
        movie_options = matches if matches else df['title'].tolist()
    else:
        movie_options = df['title'].tolist()

    selected_movie = st.selectbox("Choose a movie", movie_options, label_visibility="collapsed")

    st.divider()
    st.markdown("### 🎛 Filters")
    genre_filter = st.multiselect("Genres", all_genres)
    year_range = st.slider("Year Range", 1940, 2024, (1960, 2024))
    min_rating = st.slider("Min IMDb Rating", 0.0, 10.0, 7.5, 0.1)
    n_recs = st.slider("Recommendations", 4, 12, 8)

    st.divider()
    find_btn = st.button("✨ Find Recommendations", use_container_width=True)

    st.divider()
    st.markdown(f"""
    <div style="font-size:0.72rem;color:var(--muted);line-height:1.6">
    <b style="color:var(--accent)">How it works</b><br>
    TF-IDF vectorizes genres, directors & plot overviews. 
    Cosine similarity ranks the closest neighbours. 
    Filters refine by genre, decade, and rating.
    </div>
    """, unsafe_allow_html=True)

# ── Main Content ──────────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">Flix Verse</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-sub" style="margin-bottom:2rem">Your Intelligent Movie Recommendation Engine</div>', unsafe_allow_html=True)

# Stats row
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><div class="stat-number">50+</div><div class="stat-label">Movies</div></div>', unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><div class="stat-number">15+</div><div class="stat-label">Genres</div></div>', unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><div class="stat-number">ML</div><div class="stat-label">Powered</div></div>', unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><div class="stat-number">Live</div><div class="stat-label">Posters</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Show selected movie ───────────────────────────────────────────────────────
if selected_movie:
    movie_row = df[df['title'] == selected_movie].iloc[0]
    poster = fetch_poster(movie_row['id'])

    st.markdown('<div class="section-header">Selected Movie</div>', unsafe_allow_html=True)
    col_p, col_i = st.columns([1, 3])
    with col_p:
        if poster:
            st.image(poster, use_container_width=True)
        else:
            st.markdown('<div class="poster-placeholder" style="font-size:5rem;padding:2rem;background:#13131a;border-radius:8px;text-align:center">🎬</div>', unsafe_allow_html=True)
    with col_i:
        st.markdown(f'<h2 style="font-family:Playfair Display,serif;font-size:2rem;margin-bottom:4px">{movie_row["title"]}</h2>', unsafe_allow_html=True)
        st.markdown(f'<div style="color:#7a7a8c;font-size:0.9rem;margin-bottom:12px">{movie_row["year"]} · {movie_row["director"]}</div>', unsafe_allow_html=True)
        genres_pills = ''.join([f'<span class="genre-pill">{g.strip()}</span>' for g in movie_row['genres'].split(',')])
        st.markdown(f'<div style="margin-bottom:12px">{genres_pills}</div>', unsafe_allow_html=True)
        st.markdown(f'<div style="font-size:0.9rem;color:#b0aaa0;line-height:1.6;margin-bottom:12px">{movie_row["overview"]}</div>', unsafe_allow_html=True)
        rating_col, votes_col = st.columns(2)
        with rating_col:
            st.metric("IMDb Rating", f"⭐ {movie_row['rating']}/10")
        with votes_col:
            st.metric("Votes", f"{movie_row['votes']//1000}K+")

st.markdown("<br>", unsafe_allow_html=True)

# ── Recommendations ───────────────────────────────────────────────────────────
if find_btn or selected_movie:
    recs = recommend(
        selected_movie, df, cos_sim,
        n=n_recs,
        genre_filter=genre_filter if genre_filter else None,
        year_range=year_range,
        min_rating=min_rating
    )

    if recs.empty:
        st.warning("No recommendations match your current filters. Try loosening them!")
    else:
        st.markdown(f'<div class="section-header">Recommended for You · {len(recs)} Matches</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        for i, (_, row) in enumerate(recs.iterrows()):
            poster_url = fetch_poster(row['id'])
            with cols[i % 4]:
                st.markdown(
                    movie_card_html(
                        row['title'], row['year'], row['genres'],
                        row['rating'], row['votes'], row['similarity'], poster_url
                    ),
                    unsafe_allow_html=True
                )
                st.markdown("<br>", unsafe_allow_html=True)

# ── Genre Explorer ────────────────────────────────────────────────────────────
st.divider()
st.markdown('<div class="section-header">🎭 Genre Explorer</div>', unsafe_allow_html=True)
genre_pick = st.selectbox("Browse by genre", ["All"] + all_genres, label_visibility="visible")

filtered = df if genre_pick == "All" else df[df['genres_list'].apply(lambda gl: genre_pick in gl)]
filtered = filtered.sort_values('rating', ascending=False).head(8)

gcols = st.columns(4)
for i, (_, row) in enumerate(filtered.iterrows()):
    poster_url = fetch_poster(row['id'])
    with gcols[i % 4]:
        st.markdown(
            movie_card_html(row['title'], row['year'], row['genres'],
                            row['rating'], row['votes'], row['rating'] * 10, poster_url),
            unsafe_allow_html=True
        )
        st.markdown("<br>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown("""
<div style="text-align:center;color:#3a3a4a;font-size:0.8rem;padding:1rem">
    Built with Streamlit · scikit-learn · TMDb API &nbsp;|&nbsp; TF-IDF + Cosine Similarity Engine
</div>
""", unsafe_allow_html=True)
