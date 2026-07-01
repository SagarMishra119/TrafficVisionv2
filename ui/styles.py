from ui.theme import ACCENT, BG, BORDER, FONT, SURFACE, SURFACE_ALT, TEXT, TEXT_MUTED

def inject_css(st) -> None:
    st.markdown(f"""
    <style>
    .stApp {{ background:{BG}; font-family:{FONT}; }}
    #MainMenu, footer, header {{ visibility:hidden; }}
    [data-testid="stSidebar"] {{ background:{SURFACE}; border-right:1px solid {BORDER}; }}
    [data-testid="stMetric"] {{
        background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:12px;
        padding:14px 16px;
    }}
    [data-testid="stMetricLabel"] {{ color:{TEXT_MUTED} !important; font-size:12px !important; }}
    [data-testid="stMetricValue"] {{ color:{ACCENT} !important; font-size:22px !important; }}
    h1,h2,h3,h4,p,span,div {{ color:{TEXT}; }}
    .tv-card {{
        background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:14px;
        padding:18px 20px; margin-bottom:14px;
    }}
    .tv-card h4 {{ margin:0 0 10px 0; font-size:13px; color:{TEXT_MUTED}; text-transform:uppercase; letter-spacing:.06em; }}
    .tv-chip {{
        display:inline-flex; align-items:center; gap:6px; padding:4px 12px;
        border-radius:999px; font-size:12px; font-weight:600;
    }}
    .tv-skeleton {{
        background:linear-gradient(90deg,{SURFACE_ALT} 25%,{BORDER} 50%,{SURFACE_ALT} 75%);
        background-size:200% 100%; animation:tv-shimmer 1.4s infinite; border-radius:10px; height:80px;
    }}
    @keyframes tv-shimmer {{ 0% {{background-position:200% 0;}} 100% {{background-position:-200% 0;}} }}
    div[data-testid="stExpander"] {{ background:{SURFACE_ALT}; border:1px solid {BORDER}; border-radius:12px; }}
    .stButton>button {{
        border-radius:10px; border:1px solid {BORDER}; background:{SURFACE_ALT}; color:{TEXT};
    }}
    .stButton>button:hover {{ border-color:{ACCENT}; color:{ACCENT}; }}
    </style>
    """, unsafe_allow_html=True)
