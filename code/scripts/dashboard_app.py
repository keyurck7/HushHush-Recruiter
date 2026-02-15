# dashboard_app.py
from __future__ import annotations

from pathlib import Path
import pandas as pd
import streamlit as st

# =========================
# Paths
# =========================
RANKED_PATH = Path("data/processed/final_ranked_candidates.csv")
SHORTLIST_PATH = Path("data/processed/shortlist.csv")

# =========================
# Page Config
# =========================
st.set_page_config(
    page_title="HushHush Recruiter Dashboard",
    layout="wide"
)

st.title("🧑‍💼 Project Manager Dashboard")
st.caption("Candidates ranked by model confidence (strong_probability)")

# =========================
# Load Data
# =========================
if not RANKED_PATH.exists():
    st.error("Ranked candidates file not found. Please run ranking script first.")
    st.stop()

df = pd.read_csv(RANKED_PATH)

required_cols = {"username", "strong_probability"}
missing = required_cols - set(df.columns)
if missing:
    st.error(f"Missing required columns in ranked CSV: {missing}")
    st.stop()

# Ensure correct ordering
df = df.sort_values("strong_probability", ascending=False).reset_index(drop=True)
df["rank"] = df.index + 1

# =========================
# Sidebar Controls
# =========================
st.sidebar.header("Controls")

top_n = st.sidebar.slider(
    "Show Top N Candidates",
    min_value=5,
    max_value=50,
    value=10,
    step=1
)

search = st.sidebar.text_input("Search username")

# =========================
# Filter View
# =========================
view = df.head(top_n).copy()

if search.strip():
    view = view[view["username"].str.contains(search, case=False, na=False)]

# =========================
# Shortlist State
# =========================
if "shortlisted" not in st.session_state:
    st.session_state["shortlisted"] = set()

view["shortlist"] = view["username"].isin(st.session_state["shortlisted"])

# =========================
# Display Table
# =========================
st.subheader(f"🏆 Top {len(view)} Ranked Candidates")

edited = st.data_editor(
    view[["shortlist", "rank", "username", "strong_probability"]],
    use_container_width=True,
    hide_index=True
)

# =========================
# Save Shortlist
# =========================
if st.button("💾 Save Shortlist"):
    selected = edited.loc[edited["shortlist"] == True, "username"].tolist()
    st.session_state["shortlisted"] = set(selected)

    SHORTLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"username": selected}).to_csv(SHORTLIST_PATH, index=False)

    st.success(f"Saved {len(selected)} candidates to shortlist.")

# =========================
# Show Shortlist
# =========================
st.divider()
st.subheader("📌 Shortlisted Candidates")

if SHORTLIST_PATH.exists():
    sl = pd.read_csv(SHORTLIST_PATH)
    st.dataframe(sl, use_container_width=True, hide_index=True)

    st.download_button(
        "⬇️ Download Shortlist CSV",
        data=sl.to_csv(index=False).encode("utf-8"),
        file_name="shortlist.csv",
        mime="text/csv"
    )
else:
    st.info("No candidates shortlisted yet.")

# =========================
# Footer Note
# =========================
st.divider()
st.caption(
    "Note: Candidates are ranked purely by model confidence score. "
    "No binary filtering is applied."
)
