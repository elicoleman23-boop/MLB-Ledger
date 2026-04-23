"""
Custom CSS for The Hit Ledger.

Aesthetic direction: editorial / newspaper-ledger feel.
Off-white serif headings, deep charcoal background, amber/green accents
for hit probability (think: old-school stadium scoreboard).
"""

CSS = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,800&family=JetBrains+Mono:wght@400;600&display=swap');

    :root {
        --bg: #0f0f0e;
        --bg-elev: #1a1a19;
        --bg-elev-2: #242423;
        --text: #e8e4d8;
        --text-muted: #8a8679;
        --accent: #d4a24c;        /* amber — hot */
        --accent-cool: #5a8a6a;   /* green — cool */
        --accent-warn: #b85c4a;   /* rust — fade */
        --border: #2d2d2b;
    }

    .stApp {
        background: var(--bg);
        color: var(--text);
    }

    h1, h2, h3, h4, .ledger-title {
        font-family: 'Fraunces', Georgia, serif !important;
        font-weight: 800 !important;
        letter-spacing: -0.02em;
        color: var(--text) !important;
    }

    .ledger-title {
        font-size: 3.5rem;
        line-height: 1;
        margin: 0 0 0.25rem 0;
    }

    .ledger-subtitle {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.2em;
        color: var(--text-muted);
        margin-bottom: 2rem;
        border-bottom: 1px solid var(--border);
        padding-bottom: 1rem;
    }

    .hot-card {
        background: var(--bg-elev);
        border-left: 3px solid var(--accent);
        padding: 1rem 1.25rem;
        margin-bottom: 0.5rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
    }

    .hot-card .name {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 1.1rem;
    }

    .hot-card .meta {
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.75rem;
        color: var(--text-muted);
    }

    .hot-card .prob {
        font-family: 'JetBrains Mono', monospace;
        font-size: 1.5rem;
        font-weight: 600;
        color: var(--accent);
    }

    .matchup-edge-pos {
        color: var(--accent);
        font-weight: 600;
    }

    .matchup-edge-neg {
        color: var(--accent-warn);
        font-weight: 600;
    }

    /* Streamlit widget tweaks */
    .stDataFrame {
        font-family: 'JetBrains Mono', monospace;
    }

    .stButton > button {
        background: var(--accent);
        color: var(--bg);
        border: none;
        font-family: 'JetBrains Mono', monospace;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 0;
    }

    .stButton > button:hover {
        background: var(--text);
        color: var(--bg);
    }

    [data-testid="stSidebar"] {
        background: var(--bg-elev);
        border-right: 1px solid var(--border);
    }

    /* Progress bar in sidebar */
    .stProgress > div > div > div > div {
        background: var(--accent);
    }

    /* Expander for matchup details */
    .streamlit-expanderHeader {
        font-family: 'Fraunces', serif;
        font-weight: 600;
    }

    /* Data table cells */
    [data-testid="stDataFrame"] td {
        font-family: 'JetBrains Mono', monospace !important;
    }
</style>
"""
