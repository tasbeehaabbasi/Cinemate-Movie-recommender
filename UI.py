import os
import pickle
import requests
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw, ImageFilter

# ------------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------------
APP_NAME = "Cinemate"  # cine + mate — swap this if you want a different name

# API key is read from Streamlit secrets (.streamlit/secrets.toml) first,
# falling back to an environment variable — never hardcoded here, so it's
# safe to commit this file. Get a FREE key at https://www.themoviedb.org/settings/api
TMDB_API_KEY = st.secrets.get("TMDB_API_KEY", os.environ.get("TMDB_API_KEY", ""))

if not TMDB_API_KEY:
    st.error(
        "No TMDB API key found. Add TMDB_API_KEY to `.streamlit/secrets.toml` "
        "or set it as an environment variable, then restart the app."
    )
    st.stop()


# ------------------------------------------------------------------
# CUSTOM ICON — a cinema "ticket stub" (rounded tag shape, perforation
# notches, dashed tear-line, beveled play glyph) generated in-code, so
# it's never a flat stock emoji and never depends on a licensed
# icon-pack URL. Used as the browser tab favicon.
# ------------------------------------------------------------------
@st.cache_resource
def generate_favicon(size=256):
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))

    yy, xx = np.mgrid[0:size, 0:size]
    t = np.clip((xx / size) * 0.6 + (yy / size) * 0.4, 0, 1)
    c1 = np.array([26, 20, 12])     # deep ink-brown
    c2 = np.array([212, 162, 76])   # brass
    grad = (c1[None, None, :] * (1 - t[..., None]) + c2[None, None, :] * t[..., None]).astype(np.uint8)
    base = Image.fromarray(grad, "RGB").convert("RGBA")

    pad = int(size * 0.10)
    radius = int(size * 0.22)
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle([pad, pad, size - pad, size - pad], radius=radius, fill=255)
    base.putalpha(mask)

    shadow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).rounded_rectangle(
        [pad + 8, pad + 12, size - pad + 8, size - pad + 12], radius=radius, fill=(0, 0, 0, 130)
    )
    canvas = Image.alpha_composite(canvas, shadow.filter(ImageFilter.GaussianBlur(9)))
    canvas = Image.alpha_composite(canvas, base)

    draw = ImageDraw.Draw(canvas)
    draw.rounded_rectangle([pad, pad, size - pad, size - pad], radius=radius,
                            outline=(255, 224, 170, 180), width=max(2, size // 90))

    # perforation notches — reads as a ticket, not a generic badge
    notch_r = size * 0.045
    for ny in (pad, size - pad):
        draw.ellipse([size / 2 - notch_r, ny - notch_r, size / 2 + notch_r, ny + notch_r], fill=(13, 16, 21, 255))

    # dashed tear-line
    dash_y = size * 0.5
    x = pad + size * 0.14
    while x < size - pad - size * 0.14:
        draw.line([(x, dash_y), (x + size * 0.045, dash_y)], fill=(20, 15, 8, 160), width=max(2, size // 110))
        x += size * 0.085

    gloss = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    ImageDraw.Draw(gloss).polygon(
        [(pad, pad), (size * 0.42, pad), (size * 0.20, size - pad), (pad, size - pad)],
        fill=(255, 255, 255, 35),
    )
    canvas = Image.alpha_composite(canvas, gloss.filter(ImageFilter.GaussianBlur(4)))

    draw = ImageDraw.Draw(canvas)
    cx, cy, s = size / 2, size * 0.735, size * 0.115
    top = [(cx - s * 0.55, cy - s), (cx - s * 0.55, cy), (cx + s * 0.95, cy - s * 0.05)]
    bot = [(cx - s * 0.55, cy), (cx - s * 0.55, cy + s), (cx + s * 0.95, cy - s * 0.05)]
    draw.polygon(top, fill=(255, 236, 196, 255))
    draw.polygon(bot, fill=(150, 104, 42, 255))

    return canvas


st.set_page_config(page_title=APP_NAME, page_icon=generate_favicon(), layout="wide")


# ------------------------------------------------------------------
# DATA
# ------------------------------------------------------------------
@st.cache_data
def load_data():
    movies = pickle.load(open("movie_list.pkl", "rb"))
    similarity = pickle.load(open("similarity.pkl", "rb"))
    return movies, similarity


movies, similarity = load_data()


@st.cache_data
def fetch_details(movie_id):
    try:
        url = (
            f"https://api.themoviedb.org/3/movie/{movie_id}"
            f"?api_key={TMDB_API_KEY}&language=en-US&append_to_response=credits"
        )
        data = requests.get(url, timeout=6).json()
        poster_path = data.get("poster_path")
        poster = (
            f"https://image.tmdb.org/t/p/w500{poster_path}"
            if poster_path
            else "https://via.placeholder.com/500x750/141822/6b7280?text=No+Poster"
        )
        year = (data.get("release_date") or "")[:4] or "—"
        genres = [g["name"] for g in data.get("genres", [])][:3]
        rating = data.get("vote_average") or 0
        vote_count = data.get("vote_count") or 0
        overview = data.get("overview") or ""
        overview_short = (overview[:130] + "…") if len(overview) > 130 else overview

        runtime = data.get("runtime") or 0
        runtime_str = f"{runtime // 60}h {runtime % 60:02d}m" if runtime else "—"

        director = "—"
        for member in (data.get("credits", {}) or {}).get("crew", []):
            if member.get("job") == "Director":
                director = member.get("name", "—")
                break

        quality_tag = "4K UHD" if rating >= 7.5 else ("HD" if rating >= 5 else "SD")

        return {
            "poster": poster, "year": year, "genres": genres,
            "rating": rating, "vote_count": vote_count, "overview": overview_short,
            "runtime": runtime_str, "director": director, "quality_tag": quality_tag,
        }
    except Exception:
        return {
            "poster": "https://via.placeholder.com/500x750/141822/6b7280?text=No+Poster",
            "year": "—", "genres": [], "rating": 0, "vote_count": 0, "overview": "",
            "runtime": "—", "director": "—", "quality_tag": "HD",
        }


@st.cache_data
def fetch_trending_posters(n=10):
    """Decorative posters for the hero panel — scroll continuously so the
    panel never looks static. Fetches both day+week trending for variety."""
    posters = []
    try:
        for window in ("day", "week"):
            url = f"https://api.themoviedb.org/3/trending/movie/{window}?api_key={TMDB_API_KEY}"
            data = requests.get(url, timeout=6).json()
            for item in data.get("results", []):
                p = item.get("poster_path")
                if p:
                    posters.append(f"https://image.tmdb.org/t/p/w200{p}")
            if len(posters) >= n:
                break
        # de-dupe, keep order
        seen, unique = set(), []
        for p in posters:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        return unique[:n]
    except Exception:
        return []


def recommend(movie, n=6):
    idx = movies[movies["title"] == movie].index[0]
    distances = similarity[idx]
    ranked = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1 : n + 1]
    out = []
    for i, score in ranked:
        row = movies.iloc[i]
        out.append({"title": row.title, "match": round(score * 100), **fetch_details(row.id)})
    return out


def stars(rating_10):
    n = round(rating_10 / 2)
    return "●" * n + "○" * (5 - n)


# ------------------------------------------------------------------
# STYLE — ink navy, single brass accent, serif headlines
# ------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,wght@0,300;0,500;1,400&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #0D1015;
        --panel: #161A22;
        --line: #262b36;
        --brass: #D4A24C;
        --text: #EDEEF2;
        --muted: #8A8FA0;
    }
    .stApp { background: var(--ink); font-family: 'Inter', sans-serif; }
    h1,h2,h3,p,span,label,.stMarkdown { color: var(--text) !important; }
    #MainMenu, footer { visibility: hidden; }

    /* ---------- TOP ACCENT ---------- */
    .reel-topbar { height: 3px; background: linear-gradient(90deg, transparent, #D4A24C, transparent); margin-bottom: 0; }

    /* ---------- HEADER — animated 3D-style mark ---------- */
    .reel-header { display: flex; align-items: center; gap: 14px; padding: 18px 2px 26px 2px; border-bottom: 1px solid var(--line); margin-bottom: 36px; }
    .brand-icon { width: 34px; height: 34px; overflow: visible; }
    .brand-core { transform-origin: 50% 50%; animation: brand-tilt 3.4s ease-in-out infinite; }
    @keyframes brand-tilt {
        0%, 100% { transform: rotate(-6deg) translateY(0); }
        50% { transform: rotate(6deg) translateY(-2px); }
    }
    .reel-word { font-family: 'Fraunces', serif; font-weight: 500; font-size: 1.3rem; letter-spacing: 0.01em; }
    .reel-tagline { color: var(--muted) !important; font-size: 0.72rem; letter-spacing: 0.03em; margin-top: -2px; }

    /* ---------- SEARCH SECTION ---------- */
    .reel-eyebrow { color: var(--brass) !important; font-size: 0.72rem; letter-spacing: 0.14em; text-transform: uppercase; margin-bottom: 10px; }
    .reel-heading { font-family: 'Fraunces', serif; font-style: italic; font-weight: 400; font-size: 2.1rem; margin: 0 0 8px 0; line-height: 1.25; }
    .reel-sub { color: var(--muted) !important; font-size: 0.92rem; max-width: 480px; margin-bottom: 26px; line-height: 1.6; }

    div[data-testid="stSelectbox"] > div { background: var(--panel) !important; border: 1px solid var(--line) !important; border-radius: 8px !important; }
    div.stButton > button {
        background: var(--brass); color: #14110A !important; border: none; border-radius: 8px;
        padding: 0.6rem 1.3rem; font-weight: 600; font-size: 0.88rem; width: 100%;
        transition: opacity 0.2s ease;
    }
    div.stButton > button:hover { opacity: 0.85; }

    /* ---------- HERO RIGHT PANEL: metric card + poster reveal ---------- */
    .metric-card {
        background: linear-gradient(135deg, var(--panel), #1c2130);
        border: 1px solid var(--line); border-left: 3px solid var(--brass);
        border-radius: 8px; padding: 18px 20px; margin-bottom: 18px;
    }
    .metric-value { font-family: 'Fraunces', serif; font-size: 2rem; color: var(--brass) !important; line-height: 1; }
    .metric-label { color: var(--muted) !important; font-size: 0.76rem; letter-spacing: 0.04em; margin-top: 6px; }

    .poster-marquee { overflow: hidden; -webkit-mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent);
        mask-image: linear-gradient(90deg, transparent, #000 8%, #000 92%, transparent); }
    .poster-track { display: flex; gap: 10px; width: max-content; animation: poster-scroll 22s linear infinite; }
    .poster-track img { width: 90px; height: 130px; object-fit: cover; border-radius: 5px; flex-shrink: 0; box-shadow: 0 6px 16px rgba(0,0,0,0.4); }
    @keyframes poster-scroll { from { transform: translateX(0); } to { transform: translateX(-50%); } }

    /* ---------- PICK PANEL ---------- */
    .reel-pick { display: flex; gap: 22px; background: var(--panel); border: 1px solid var(--line); border-radius: 4px; padding: 22px; margin: 34px 0 42px 0; }
    .reel-pick img { width: 110px; border-radius: 3px; flex-shrink: 0; }
    .reel-pick-label { color: var(--brass) !important; font-size: 0.66rem; letter-spacing: 0.12em; text-transform: uppercase; margin-bottom: 6px; }
    .reel-pick-title { font-family: 'Fraunces', serif; font-size: 1.35rem; margin-bottom: 4px; }
    .reel-pick-meta { color: var(--muted) !important; font-size: 0.78rem; margin-bottom: 10px; }
    .reel-pick-overview { color: var(--muted) !important; font-size: 0.84rem; line-height: 1.6; }

    /* ---------- RESULTS ---------- */
    .reel-section-title { font-family: 'Fraunces', serif; font-style: italic; font-size: 1.25rem; margin: 0 0 20px 0; }

    .reel-card {
        background: var(--panel); border: 1px solid var(--line); border-radius: 10px;
        padding: 10px 10px 14px 10px; position: relative; opacity: 0;
        animation: reel-fade-up 0.5s ease forwards; transition: border-color 0.25s ease, transform 0.25s ease;
    }
    .reel-card:hover { border-color: var(--brass); transform: translateY(-2px); }
    @keyframes reel-fade-up { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

    .poster-wrap { position: relative; border-radius: 6px; overflow: hidden; }
    .reel-card img { width: 100%; border-radius: 6px; display: block; transition: transform 0.35s ease, opacity 0.25s ease; }
    .reel-card:hover img { opacity: 0.92; transform: scale(1.025); }

    .badge-quality {
        position: absolute; top: 8px; left: 8px; background: rgba(13,16,21,0.78);
        border: 1px solid rgba(212,162,76,0.5); color: var(--brass) !important;
        font-size: 0.62rem; letter-spacing: 0.04em; padding: 3px 7px; border-radius: 20px;
    }
    .badge-match {
        position: absolute; top: 8px; right: 8px; background: var(--brass); color: #14110A !important;
        font-size: 0.66rem; font-weight: 700; padding: 3px 8px; border-radius: 20px;
    }

    .reel-card-title { font-weight: 600; font-size: 0.95rem; margin-top: 12px; line-height: 1.3; }
    .reel-card-meta { color: var(--muted) !important; font-size: 0.72rem; margin-top: 3px; }

    .genre-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
    .genre-pill {
        border: 1px solid var(--line); color: var(--muted) !important; font-size: 0.66rem;
        padding: 2px 8px; border-radius: 20px; letter-spacing: 0.02em;
    }

    .reel-card-stars { color: var(--brass) !important; font-size: 0.8rem; margin-top: 10px; letter-spacing: 2px; }
    .reel-card-stars .rating-count { color: var(--muted) !important; font-size: 0.7rem; letter-spacing: 0; margin-left: 4px; }
    .reel-card-overview { color: var(--muted) !important; font-size: 0.78rem; line-height: 1.55; margin-top: 8px; }

    /* "load more" skeleton-style card */
    .skeleton-card {
        background: var(--panel); border: 1px dashed var(--line); border-radius: 10px;
        padding: 16px; display: flex; flex-direction: column; align-items: center; justify-content: center;
        text-align: center; min-height: 260px; gap: 10px;
    }
    .skeleton-spinner {
        width: 26px; height: 26px; border-radius: 50%; border: 2px solid var(--line);
        border-top-color: var(--brass); animation: brand-spin-load 0.9s linear infinite;
    }
    @keyframes brand-spin-load { to { transform: rotate(360deg); } }
    .skeleton-card-title { font-size: 0.82rem; color: var(--text) !important; }
    .skeleton-card-sub { font-size: 0.68rem; color: var(--muted) !important; }
    .skeleton-bar { height: 8px; width: 80%; border-radius: 4px; background: var(--line);
        animation: skeleton-pulse 1.4s ease-in-out infinite; }
    @keyframes skeleton-pulse { 0%, 100% { opacity: 0.5; } 50% { opacity: 1; } }

    .reel-footer { border-top: 1px solid var(--line); margin-top: 60px; padding: 24px 2px; color: var(--muted) !important; font-size: 0.72rem; text-align: center; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HEADER — same ticket-stub mark as the favicon (perforation notches,
# tear-line, play glyph), gently tilting like a ticket being handed
# over. No emoji, no external image request.
# ------------------------------------------------------------------
brand_svg = """
<svg class="brand-icon" viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <linearGradient id="brandTicket" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F0D9B0"/>
      <stop offset="55%" stop-color="#D4A24C"/>
      <stop offset="100%" stop-color="#7A5A28"/>
    </linearGradient>
  </defs>
  <g class="brand-core">
    <rect x="6" y="4" width="36" height="40" rx="9" fill="url(#brandTicket)"
          stroke="#F0D9B0" stroke-width="1" opacity="0.96"/>
    <circle cx="24" cy="4" r="3.2" fill="#161A22"/>
    <circle cx="24" cy="44" r="3.2" fill="#161A22"/>
    <line x1="12" y1="24" x2="15" y2="24" stroke="#3a2c14" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="19" y1="24" x2="22" y2="24" stroke="#3a2c14" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="26" y1="24" x2="29" y2="24" stroke="#3a2c14" stroke-width="1.6" stroke-linecap="round"/>
    <line x1="33" y1="24" x2="36" y2="24" stroke="#3a2c14" stroke-width="1.6" stroke-linecap="round"/>
    <path d="M19 30.5 L29 35 L19 39.5 Z" fill="#161A22" opacity="0.9"/>
  </g>
</svg>
"""

st.markdown('<div class="reel-topbar"></div>', unsafe_allow_html=True)
st.markdown(
    f'<div class="reel-header">{brand_svg}<div><div class="reel-word">{APP_NAME}</div>'
    f'<div class="reel-tagline">A quiet corner for finding what to watch next</div>'
    f'</div></div>',
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# HERO — search on the left, metric card + staggered poster reveal
# on the right (previously empty space)
# ------------------------------------------------------------------
hero_left, hero_right = st.columns([3, 2])

with hero_left:
    st.markdown('<div class="reel-eyebrow">Content-based discovery</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="reel-heading">Tell me a film you loved,<br>I\'ll find its relatives.</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="reel-sub">Matches are drawn from plot, cast, crew, and genre — '
        'ranked by cosine similarity, not popularity.</div>',
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns([4, 1])
    with col1:
        selected_movie = st.selectbox(
            "Search", movies["title"].values, index=None,
            placeholder="Start typing a title…", label_visibility="collapsed",
        )
    with col2:
        go = st.button("Find similar")

    st.markdown(
        f'<div style="color:#8A8FA0; font-size:0.7rem; letter-spacing:0.04em; margin-top:10px;">'
        f'Searching across {len(movies):,} indexed titles</div>',
        unsafe_allow_html=True,
    )

with hero_right:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-value">{len(movies):,}</div>
            <div class="metric-label">FILMS INDEXED · LIVE TMDB DATA</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    trending_posters = fetch_trending_posters(10)
    if trending_posters:
        # duplicate the list so the scroll loop is seamless (translateX -50%)
        loop_posters = trending_posters + trending_posters
        posters_html = "".join(f'<img src="{src}"/>' for src in loop_posters)
        st.markdown(
            f'<div class="poster-marquee"><div class="poster-track">{posters_html}</div></div>',
            unsafe_allow_html=True,
        )

if "reel_movie" not in st.session_state:
    st.session_state.reel_movie = None
if "rec_count" not in st.session_state:
    st.session_state.rec_count = 6

if go and selected_movie:
    st.session_state.reel_movie = selected_movie
    st.session_state.rec_count = 6
elif go and not selected_movie:
    st.warning("Pick a title first.")

# ------------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------------
if st.session_state.reel_movie:
    movie = st.session_state.reel_movie
    pick = fetch_details(movies[movies["title"] == movie].iloc[0].id)

    st.markdown(
        f"""
        <div class="reel-pick">
            <img src="{pick['poster']}" />
            <div>
                <div class="reel-pick-label">Your pick</div>
                <div class="reel-pick-title">{movie}</div>
                <div class="reel-pick-meta">{pick['year']} · {' / '.join(pick['genres'])} · {pick['rating']:.1f} rating</div>
                <div class="reel-pick-overview">{pick['overview']}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    max_n = min(len(movies) - 1, 24)
    n = min(st.session_state.rec_count, max_n)
    with st.spinner("Ranking similar films…"):
        results = recommend(movie, n=n)

    st.markdown(f'<div class="reel-section-title">Top {len(results)} recommendations for "{movie}"</div>', unsafe_allow_html=True)

    cols = st.columns(3)
    for idx, r in enumerate(results):
        with cols[idx % 3]:
            delay = (idx % 6) * 0.08
            pill_html = "".join(f'<span class="genre-pill">{g}</span>' for g in r["genres"]) or '<span class="genre-pill">—</span>'
            rating_count = f"{r.get('vote_count', 0):,}"

            st.markdown(
                f"""
                <div class="reel-card" style="animation-delay:{delay}s;">
                    <div class="poster-wrap">
                        <span class="badge-quality">{r['quality_tag']}</span>
                        <span class="badge-match">{r['match']}% Match</span>
                        <img src="{r['poster']}" />
                    </div>
                    <div class="reel-card-title">{r['title']}</div>
                    <div class="reel-card-meta">Dir. {r['director']} · {r['year']} · {r['runtime']}</div>
                    <div class="genre-pills">{pill_html}</div>
                    <div class="reel-card-stars">{stars(r['rating'])} <span class="rating-count">{r['rating']:.1f} ({rating_count})</span></div>
                    <div class="reel-card-overview">{r['overview']}</div>
                </div>
                <div style="height:14px"></div>
                """,
                unsafe_allow_html=True,
            )

    # trailing "load more" card, styled like a computing/skeleton slot
    remaining = max_n - n
    with cols[len(results) % 3]:
        if remaining > 0:
            st.markdown(
                """
                <div class="skeleton-card">
                    <div class="skeleton-spinner"></div>
                    <div class="skeleton-card-title">Load more matches</div>
                    <div class="skeleton-card-sub">Rank further titles by similarity</div>
                    <div class="skeleton-bar"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if st.button("Load 6 more", key="load_more"):
                st.session_state.rec_count = min(st.session_state.rec_count + 6, max_n)
                st.rerun()

st.markdown(
    f'<div class="reel-footer">{APP_NAME} · Recommendations ranked by content similarity, not popularity.</div>',
    unsafe_allow_html=True,
)
