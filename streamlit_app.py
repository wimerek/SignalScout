import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from io import StringIO

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalScout",
    page_icon="🏈",
    layout="wide"
)

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"
PLAYERS_FILE = os.path.join(DATA_DIR, "players.csv")
ARCHETYPES_FILE = os.path.join(DATA_DIR, "archetypes.csv")

METRICS = ["height", "weight", "arm_length", "forty", "vertical", "broad_jump", "three_cone", "shuttle"]
METRIC_LABELS = {
    "height":      "Height (e.g. 6042 — ft/in/eighths)",
    "weight":      "Weight (lbs)",
    "arm_length":  "Arm Length (inches)",
    "forty":       "40-Yard Dash (seconds)",
    "vertical":    "Vertical Jump (inches)",
    "broad_jump":  "Broad Jump (inches)",
    "three_cone":  "3-Cone Drill (seconds)",
    "shuttle":     "Short Shuttle (seconds)",
}
# Metrics where LOWER is better
LOWER_IS_BETTER = {"forty", "three_cone", "shuttle"}

POSITIONS = ["RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "S", "QB"]

WEIGHT_MAP = {"rank1": 2.0, "rank2": 1.75, "rank3": 1.5, "rank4": 1.25}
HW_WEIGHT = 1.1       # height & weight universal secondary importance
DEFAULT_WEIGHT = 1.0  # all other unranked metrics
MISSING_PENALTY = 0.15  # added to distance per missing metric

# ── Height encoding helpers ────────────────────────────────────────────────────
def parse_height(h):
    """Convert NFL combine height encoding (e.g. 6042) to decimal inches."""
    try:
        h = str(int(h)).zfill(4)
        feet = int(h[0])
        inches = int(h[1:3])
        eighths = int(h[3])
        return feet * 12 + inches + eighths / 8
    except:
        return np.nan

def format_height(h):
    """Convert decimal inches back to display string like 6'4 2/8\"."""
    try:
        h = float(h)
        feet = int(h // 12)
        remaining = h - feet * 12
        inches = int(remaining)
        eighths = round((remaining - inches) * 8)
        return f"{feet}'{inches} {eighths}/8\""
    except:
        return str(h)

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_players():
    try:
        df = pd.read_csv(PLAYERS_FILE)
        df["height_display"] = df["height"].apply(lambda x: format_height(parse_height(x)))
        df["height_in"] = df["height"].apply(parse_height)
        return df
    except Exception as e:
        st.error(f"Could not load players data: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=60)
def load_archetypes():
    try:
        return pd.read_csv(ARCHETYPES_FILE)
    except Exception as e:
        st.error(f"Could not load archetypes data: {e}")
        return pd.DataFrame()

def save_players(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df_save = df.drop(columns=["height_display", "height_in"], errors="ignore")
    df_save.to_csv(PLAYERS_FILE, index=False)
    st.cache_data.clear()

def save_archetypes(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(ARCHETYPES_FILE, index=False)
    st.cache_data.clear()

# ── Weighting logic ────────────────────────────────────────────────────────────
def get_weights(position, archetypes_df):
    """Return a dict of metric -> weight for the given position."""
    weights = {m: DEFAULT_WEIGHT for m in METRICS}
    weights["height"] = HW_WEIGHT
    weights["weight"] = HW_WEIGHT

    row = archetypes_df[archetypes_df["position"] == position]
    if row.empty:
        return weights

    row = row.iloc[0]
    for rank_key, rank_weight in WEIGHT_MAP.items():
        metric = row.get(rank_key, None)
        if pd.notna(metric) and str(metric).strip() in METRICS:
            weights[str(metric).strip()] = rank_weight

    return weights

# ── Core comparison engine ─────────────────────────────────────────────────────
def compute_comps(prospect: dict, position: str, players_df: pd.DataFrame, archetypes_df: pd.DataFrame, top_n=10):
    """
    Returns a DataFrame of top_n comps sorted by similarity score descending.
    prospect: dict of metric -> raw value (height still in encode form)
    """
    # Convert prospect height to inches
    prospect_in = prospect.copy()
    if pd.notna(prospect.get("height")):
        prospect_in["height"] = parse_height(prospect["height"])
    else:
        prospect_in["height"] = np.nan

    # Filter to same position
    pos_df = players_df[players_df["position"] == position].copy()
    if pos_df.empty:
        return pd.DataFrame()

    # Use height_in column for calculations
    pos_df = pos_df.copy()

    weights = get_weights(position, archetypes_df)

    results = []
    for _, row in pos_df.iterrows():
        dist_sq = 0.0
        metrics_used = 0
        missing_count = 0

        for metric in METRICS:
            col = "height_in" if metric == "height" else metric
            p_val = prospect_in.get(metric, np.nan)
            r_val = row.get(col, np.nan)

            if pd.isna(p_val) or pd.isna(r_val):
                missing_count += 1
                continue

            # Compute position-level mean and std for normalization
            metric_col = "height_in" if metric == "height" else metric
            pos_vals = pos_df[metric_col].dropna()
            if len(pos_vals) < 2:
                continue
            mean = pos_vals.mean()
            std = pos_vals.std()
            if std == 0:
                continue

            z_prospect = (float(p_val) - mean) / std
            z_row = (float(r_val) - mean) / std

            # Flip sign for lower-is-better metrics
            if metric in LOWER_IS_BETTER:
                z_prospect = -z_prospect
                z_row = -z_row

            w = weights.get(metric, DEFAULT_WEIGHT)
            dist_sq += w * (z_prospect - z_row) ** 2
            metrics_used += 1

        if metrics_used == 0:
            continue

        # Normalize by metrics used and apply missing penalty
        raw_distance = np.sqrt(dist_sq / metrics_used) + (missing_count * MISSING_PENALTY)

        results.append({
            "name": row["name"],
            "position": row["position"],
            "raw_distance": raw_distance,
            "metrics_used": metrics_used,
            "missing_count": missing_count,
            # store raw values for display
            **{m: row.get("height_in" if m == "height" else m, np.nan) for m in METRICS}
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("raw_distance")

    # Convert raw distance to similarity %
    max_dist = results_df["raw_distance"].max()
    if max_dist == 0:
        results_df["similarity"] = 100.0
    else:
        results_df["similarity"] = ((1 - results_df["raw_distance"] / (max_dist + 1)) * 100).round(1)

    return results_df.head(top_n).reset_index(drop=True)

# ── Display helpers ────────────────────────────────────────────────────────────
def fmt(metric, val):
    if pd.isna(val):
        return "—"
    if metric == "height":
        return format_height(val)
    if metric in ("forty", "three_cone", "shuttle"):
        return f"{val:.2f}s"
    if metric == "arm_length":
        return f"{val:.3f}\""
    if metric in ("vertical", "broad_jump"):
        return f"{val:.1f}\""
    if metric == "weight":
        return f"{int(val)} lbs"
    return str(val)

# ── Admin authentication ───────────────────────────────────────────────────────
def check_admin():
    try:
        admin_pw = st.secrets["ADMIN_PASSWORD"]
    except:
        admin_pw = "admin123"  # fallback for local dev

    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False

    if not st.session_state.is_admin:
        with st.sidebar:
            st.markdown("---")
            st.subheader("Admin Login")
            pw = st.text_input("Password", type="password", key="admin_pw_input")
            if st.button("Login"):
                if pw == admin_pw:
                    st.session_state.is_admin = True
                    st.rerun()
                else:
                    st.error("Incorrect password")
    else:
        with st.sidebar:
            st.markdown("---")
            st.success("Logged in as Admin")
            if st.button("Logout"):
                st.session_state.is_admin = False
                st.rerun()

# ── Admin panel ────────────────────────────────────────────────────────────────
def admin_panel():
    st.header("⚙️ Admin Panel")

    tab1, tab2 = st.tabs(["📤 Upload Dataset", "📐 Edit Archetypes"])

    # ── Tab 1: Upload players CSV ──
    with tab1:
        st.subheader("Upload Player Database")
        st.caption("Upload a CSV with columns: name, position, height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle")

        uploaded = st.file_uploader("Choose CSV file", type="csv", key="player_upload")
        if uploaded:
            try:
                df = pd.read_csv(uploaded)
                required = {"name", "position", "height", "weight"}
                missing_cols = required - set(df.columns)
                if missing_cols:
                    st.error(f"Missing required columns: {missing_cols}")
                else:
                    st.success(f"Preview — {len(df)} players loaded")
                    st.dataframe(df.head(10), use_container_width=True)
                    if st.button("✅ Confirm & Save Dataset"):
                        save_players(df)
                        st.success("Dataset saved successfully.")
            except Exception as e:
                st.error(f"Error reading file: {e}")

    # ── Tab 2: Edit archetypes ──
    with tab2:
        st.subheader("Position Archetypes & Weights")
        st.caption("Edit position average measurables and metric priority rankings. rank1 = most important (weight 2.0), rank4 = 4th most important (weight 1.25). Use exact metric names: height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle")

        archetypes_df = load_archetypes()
        if archetypes_df.empty:
            st.warning("No archetypes file found.")
            return

        edited = st.data_editor(
            archetypes_df,
            use_container_width=True,
            num_rows="fixed",
            key="archetype_editor"
        )

        if st.button("💾 Save Archetypes"):
            save_archetypes(edited)
            st.success("Archetypes saved successfully.")

# ── User comparison tool ───────────────────────────────────────────────────────
def user_tool():
    players_df = load_players()
    archetypes_df = load_archetypes()

    if players_df.empty:
        st.warning("No player database loaded yet.")
        return

    st.header("🔍 Prospect Comparison")

    with st.form("comp_form"):
        col1, col2 = st.columns([1, 1])

        with col1:
            name = st.text_input("Prospect Name")
            position = st.selectbox("Position", POSITIONS)
            height = st.text_input(METRIC_LABELS["height"], placeholder="e.g. 6042")
            weight = st.number_input(METRIC_LABELS["weight"], min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            arm_length = st.number_input(METRIC_LABELS["arm_length"], min_value=0.0, max_value=40.0, value=0.0, step=0.125)

        with col2:
            forty = st.number_input(METRIC_LABELS["forty"], min_value=0.0, max_value=7.0, value=0.0, step=0.01)
            vertical = st.number_input(METRIC_LABELS["vertical"], min_value=0.0, max_value=60.0, value=0.0, step=0.5)
            broad_jump = st.number_input(METRIC_LABELS["broad_jump"], min_value=0.0, max_value=160.0, value=0.0, step=1.0)
            three_cone = st.number_input(METRIC_LABELS["three_cone"], min_value=0.0, max_value=10.0, value=0.0, step=0.01)
            shuttle = st.number_input(METRIC_LABELS["shuttle"], min_value=0.0, max_value=7.0, value=0.0, step=0.01)

        submitted = st.form_submit_button("🏈 Find Comps", use_container_width=True)

    if submitted:
        if not name:
            st.warning("Please enter a prospect name.")
            return

        # Build prospect dict — treat 0.0 as missing for optional metrics
        prospect = {
            "height": height if height.strip() else np.nan,
            "weight": weight if weight > 0 else np.nan,
            "arm_length": arm_length if arm_length > 0 else np.nan,
            "forty": forty if forty > 0 else np.nan,
            "vertical": vertical if vertical > 0 else np.nan,
            "broad_jump": broad_jump if broad_jump > 0 else np.nan,
            "three_cone": three_cone if three_cone > 0 else np.nan,
            "shuttle": shuttle if shuttle > 0 else np.nan,
        }

        # Convert prospect height for display
        prospect_height_in = parse_height(prospect["height"]) if pd.notna(prospect.get("height")) and str(prospect.get("height","")).strip() else np.nan

        with st.spinner("Running comparison..."):
            comps = compute_comps(prospect, position, players_df, archetypes_df)

        if comps.empty:
            st.warning(f"No comparable players found for position: {position}")
            return

        st.subheader(f"Top Comps for **{name}** ({position})")

        # Build display table
        display_rows = []
        for i, row in comps.iterrows():
            entry = {
                "Rank": i + 1,
                "Player": row["name"],
                "Position": row["position"],
                "Similarity %": f"{row['similarity']}%",
                "Raw Distance": f"{row['raw_distance']:.3f}",
            }
            for metric in METRICS:
                p_val = prospect_height_in if metric == "height" else prospect.get(metric, np.nan)
                db_val = row.get(metric, np.nan)
                label = metric.replace("_", " ").title()
                entry[f"{label} (You | DB)"] = f"{fmt(metric, p_val)} | {fmt(metric, db_val)}"

            display_rows.append(entry)

        display_df = pd.DataFrame(display_rows)

        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.caption(f"Similarity % is relative to this result set. Raw Distance: lower = closer match. Metrics used per comp shown in parentheses where data was available.")

# ── Main app ───────────────────────────────────────────────────────────────────
def main():
    # Sidebar branding
    with st.sidebar:
        st.image("https://img.icons8.com/emoji/96/american-football-emoji.png", width=60)
        st.title("SignalScout")
        st.caption("NFL Draft Prospect Comparison Tool")
        st.markdown("---")
        page = st.radio("Navigate", ["🔍 Find Comps", "⚙️ Admin"])

    check_admin()

    if page == "🔍 Find Comps":
        user_tool()
    elif page == "⚙️ Admin":
        if st.session_state.get("is_admin"):
            admin_panel()
        else:
            st.warning("Admin login required. Use the sidebar to log in.")

if __name__ == "__main__":
    main()
