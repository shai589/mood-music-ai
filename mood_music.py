import streamlit as st
from sentence_transformers import SentenceTransformer, util
import numpy as np
import random

# ------------------------------------------------------------
# PAGE SETTINGS & STYLES
# ------------------------------------------------------------

st.set_page_config(
    page_title="Mood Music: AI Vibes Engine",
    page_icon="🎵",
    layout="centered",
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Quicksand&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"]  {
        font-family: 'Quicksand', sans-serif;
        background-color: #121212;
        color: #f0f0f0;
    }
    .title {
        text-align: center;
        font-size: 48px;
        color: #1DB954;
        margin-top: 30px;
        margin-bottom: 5px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #b3b3b3;
        margin-bottom: 30px;
    }
    .card {
        background-color: #1e1e1e;
        padding: 30px;
        border-radius: 12px;
        box-shadow: 0 0 15px rgba(0, 0, 0, 0.4);
        margin-top: 20px;
    }
    a {
        color: #1DB954;
        text-decoration: none;
        font-weight: bold;
    }
    </style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# HEADER
# ------------------------------------------------------------

st.markdown("""
    <div class='title'>Mood Music: AI Vibes Engine 🎧</div>
    <div class='subtitle'>Let your feelings drive your soundtrack</div>
""", unsafe_allow_html=True)

# ------------------------------------------------------------
# LOAD EMBEDDING MODEL
# ------------------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

model = load_model()

# ------------------------------------------------------------
# PLAYLIST DATABASE
# ------------------------------------------------------------

playlists = [
    {
        "name": "Chill Lo-Fi Beats",
        "description": "Calm, focused, relaxing, mellow beats for studying.",
        "link": "https://open.spotify.com/playlist/37i9dQZF1DX4WYpdgoIcn6"
    },
    {
        "name": "Party Hits",
        "description": "Energetic, happy, upbeat pop songs for dancing and parties.",
        "link": "https://open.spotify.com/playlist/37i9dQZF1DXaXB8fQg7xif"
    },
    {
        "name": "Deep Focus",
        "description": "Concentration, minimal, ambient music to stay productive.",
        "link": "https://open.spotify.com/playlist/37i9dQZF1DWZeKCadgRdKQ"
    },
    {
        "name": "Acoustic Evening",
        "description": "Warm, mellow, soothing acoustic guitar and soft vocals.",
        "link": "https://open.spotify.com/playlist/37i9dQZF1DX7K31D69s4M1"
    },
    {
        "name": "Motivation Boost",
        "description": "High energy, powerful songs for working out or staying motivated.",
        "link": "https://open.spotify.com/playlist/37i9dQZF1DX76Wlfdnj7AP"
    }
]

# ------------------------------------------------------------
# PRE-ENCODE PLAYLIST DESCRIPTIONS
# ------------------------------------------------------------

playlist_descriptions = [p["description"] for p in playlists]
playlist_embeddings = model.encode(playlist_descriptions)

# ------------------------------------------------------------
# MODERN ARTIST QUOTES
# ------------------------------------------------------------

quotes_by_modern_artists = [
    "“I want to make what I want to make, and make it sound how I want it to sound.” – Billie Eilish",
    "“People haven’t always been there for me but music always has.” – Taylor Swift",
    "“When all is said and done, more is always said than done.” – Drake",
    "“I always thought it was me against the world. Then I realized it’s just me against me.” – Kendrick Lamar",
    "“The most alluring thing a woman can have is confidence.” – Beyoncé",
    "“Everything will be okay in the end. If it’s not okay, it’s not the end.” – Ed Sheeran",
    "“I believe happiness is the best success.” – Ariana Grande",
    "“A dream is only a dream until you decide to make it real.” – Harry Styles",
]

# ------------------------------------------------------------
# INPUT UI
# ------------------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    user_mood = st.text_input(
        "",
        placeholder="Describe your mood… e.g. calm and dreamy",
        help="Type how you feel. Let AI be your DJ!"
    )

# ------------------------------------------------------------
# MAIN LOGIC
# ------------------------------------------------------------

if user_mood:
    with st.spinner("🔎 Analyzing your vibe..."):
        user_embedding = model.encode(user_mood)
        similarities = util.cos_sim(user_embedding, playlist_embeddings).flatten()
        best_idx = np.argmax(similarities)
        best_match = playlists[best_idx]

    st.markdown(f"""
        <div class='card'>
            <h2 style='color:#1DB954;'>🎧 Recommended Playlist: {best_match['name']}</h2>
            <p style='color:#BBBBBB; font-size:18px;'>
                <strong>Description:</strong> {best_match['description']}
            </p>
            <p style='font-size:18px;'>
                <a href='{best_match['link']}' target='_blank'>Listen on Spotify</a>
            </p>
            <p style='color:#666666; font-size:14px;'>
                Similarity Score: {similarities[best_idx]:.2f}
            </p>
        </div>
    """, unsafe_allow_html=True)

    # Show a random quote
    quote = random.choice(quotes_by_modern_artists)

    st.markdown(f"""
        <div style='background-color:#1e1e1e; padding:15px; border-radius:10px; margin-top:20px;'>
            <p style='font-size:18px; color:#1DB954; text-align:center;'>{quote}</p>
        </div>
    """, unsafe_allow_html=True)

else:
    st.info("Start by describing how you feel above to discover your AI-curated music vibe!")
