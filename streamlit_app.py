import streamlit as st
import pandas as pd
import numpy as np
import json
import os
from io import StringIO, BytesIO

st.set_page_config(page_title="SignalScout", page_icon="🏈", layout="wide")

# ── Constants ──────────────────────────────────────────────────────────────────
DATA_DIR = "data"
PLAYERS_FILE = os.path.join(DATA_DIR, "players.csv")
ARCHETYPES_FILE = os.path.join(DATA_DIR, "archetypes.csv")

METRICS = ["height", "weight", "arm_length", "forty", "vertical", "broad_jump", "three_cone", "shuttle"]
LOWER_IS_BETTER = {"forty", "three_cone", "shuttle"}
POSITIONS = ["RB", "WR", "TE", "OT", "IOL", "DT", "EDGE", "LB", "CB", "S", "QB"]

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

COL_NAMES = {
    "height":     "Height",
    "weight":     "Weight",
    "arm_length": "Arm",
    "forty":      "40 YD",
    "vertical":   "Vertical",
    "broad_jump": "Broad Jump",
    "three_cone": "3 Cone",
    "shuttle":    "Short Shuttle",
}

WEIGHT_MAP = {"rank1": 2.0, "rank2": 1.75, "rank3": 1.5, "rank4": 1.25}
HW_WEIGHT = 1.1
DEFAULT_WEIGHT = 1.0
MISSING_PENALTY = 0.15

# ── Height helpers ─────────────────────────────────────────────────────────────
def parse_height(h):
    try:
        h = str(int(float(str(h)))).zfill(4)
        feet = int(h[0])
        inches = int(h[1:3])
        eighths = int(h[3])
        return feet * 12 + inches + eighths / 8
    except:
        return np.nan

def format_height_decimal(val_inches):
    try:
        val = float(val_inches)
        feet = int(val // 12)
        remaining = val - feet * 12
        return f"{feet}'{remaining:.1f}\""
    except:
        return "—"

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def load_players():
    try:
        df = pd.read_csv(PLAYERS_FILE)
        df["height_in"] = df["height"].apply(parse_height)
        if "draft_year" not in df.columns:
            df["draft_year"] = ""
        if "college" not in df.columns:
            df["college"] = ""
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
    df_save = df.drop(columns=["height_in"], errors="ignore")
    df_save.to_csv(PLAYERS_FILE, index=False)
    st.cache_data.clear()

def save_archetypes(df):
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(ARCHETYPES_FILE, index=False)
    st.cache_data.clear()

# ── Weighting ──────────────────────────────────────────────────────────────────
def get_weights(position, archetypes_df):
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

# ── Comparison engine ──────────────────────────────────────────────────────────
def compute_comps(prospect, positions, players_df, archetypes_df, top_n=10, year_range=None):
    prospect_in = prospect.copy()
    if pd.notna(prospect.get("height")):
        prospect_in["height"] = parse_height(prospect["height"])
    else:
        prospect_in["height"] = np.nan

    pos_df = players_df[players_df["position"].isin(positions)].copy()
    if pos_df.empty:
        return pd.DataFrame()

    if year_range:
        yr_df = pd.to_numeric(pos_df["draft_year"], errors="coerce")
        pos_df = pos_df[(yr_df >= year_range[0]) & (yr_df <= year_range[1])]
    if pos_df.empty:
        return pd.DataFrame()

    results = []
    for _, row in pos_df.iterrows():
        position = row["position"]
        weights = get_weights(position, archetypes_df)
        pos_group = players_df[players_df["position"] == position]

        dist_sq = 0.0
        metrics_used = 0
        missing_count = 0
        percentiles = {}

        for metric in METRICS:
            col = "height_in" if metric == "height" else metric
            p_val = prospect_in.get(metric, np.nan)
            r_val = row.get(col, np.nan)

            pos_vals = pos_group["height_in" if metric == "height" else metric].dropna()
            if len(pos_vals) < 2:
                missing_count += 1
                continue
            mean = pos_vals.mean()
            std = pos_vals.std()
            if std == 0:
                missing_count += 1
                continue

            if pd.notna(r_val):
                pct = (pos_vals < float(r_val)).mean() * 100
                if metric in LOWER_IS_BETTER:
                    pct = 100 - pct
                percentiles[metric] = round(pct)

            if pd.isna(p_val) or pd.isna(r_val):
                missing_count += 1
                continue

            z_p = (float(p_val) - mean) / std
            z_r = (float(r_val) - mean) / std
            if metric in LOWER_IS_BETTER:
                z_p = -z_p
                z_r = -z_r

            w = weights.get(metric, DEFAULT_WEIGHT)
            dist_sq += w * (z_p - z_r) ** 2
            metrics_used += 1

        if metrics_used == 0:
            continue

        raw_distance = np.sqrt(dist_sq / metrics_used) + (missing_count * MISSING_PENALTY)

        results.append({
            "name": row["name"],
            "position": row["position"],
            "college": row.get("college", ""),
            "draft_year": row.get("draft_year", ""),
            "raw_distance": raw_distance,
            "metrics_used": metrics_used,
            "percentiles": percentiles,
            **{m: row.get("height_in" if m == "height" else m, np.nan) for m in METRICS}
        })

    if not results:
        return pd.DataFrame()

    results_df = pd.DataFrame(results).sort_values("raw_distance")
    max_dist = results_df["raw_distance"].max()
    if max_dist == 0:
        results_df["similarity"] = 100
    else:
        results_df["similarity"] = ((1 - results_df["raw_distance"] / (max_dist + 1)) * 100).round(0).astype(int)

    return results_df.head(top_n).reset_index(drop=True)

# ── Formatting ─────────────────────────────────────────────────────────────────
def fmt(metric, val):
    if pd.isna(val) or val == "" or val is None:
        return "—"
    try:
        v = float(val)
    except:
        return str(val)
    if metric == "height":
        return format_height_decimal(v)
    if metric == "weight":
        return f"{int(v)}"
    if metric == "arm_length":
        return f"{v:.1f}"
    if metric in ("forty", "three_cone", "shuttle"):
        return f"{v:.2f}"
    if metric in ("vertical", "broad_jump"):
        return f"{v:.1f}"
    return str(val)

def metric_diff_pct(p_val, db_val, metric):
    try:
        p = float(p_val)
        d = float(db_val)
        if metric == "weight":
            spread = 50
        elif metric == "height":
            spread = 4
        elif metric in ("forty", "three_cone", "shuttle"):
            spread = 0.5
        else:
            spread = 10
        diff = abs(p - d) / spread
        return max(0.0, min(1.0, 1.0 - diff))
    except:
        return None

def green_bg(closeness):
    if closeness is None:
        return ""
    r = int(234 - closeness * 80)
    g = int(243 - closeness * 20)
    b = int(222 - closeness * 120)
    return f"background-color: rgb({r},{g},{b}); padding: 4px 6px; border-radius: 4px;"

# ── Green shading HTML table ───────────────────────────────────────────────────
def build_html_table(comps, prospect_vals):
    prospect_height_in = parse_height(prospect_vals.get("height")) if pd.notna(prospect_vals.get("height","")) and str(prospect_vals.get("height","")).strip() else np.nan

    col_order = ["similarity", "height", "weight", "arm_length", "forty", "vertical", "broad_jump", "three_cone", "shuttle", "raw_distance"]
    col_display = {
        "similarity": "Match",
        "height": "Height",
        "weight": "Weight",
        "arm_length": "Arm",
        "forty": "40 YD",
        "vertical": "Vertical",
        "broad_jump": "Broad Jump",
        "three_cone": "3 Cone",
        "shuttle": "Short Shuttle",
        "raw_distance": "Dist",
    }

    html = """
    <style>
    .ss-table { border-collapse: collapse; width: 100%; font-size: 13px; }
    .ss-table th { 
        background: transparent; 
        color: var(--color-text-secondary, #888); 
        font-weight: 500; 
        padding: 6px 8px; 
        text-align: center; 
        border-bottom: 1px solid rgba(136,135,128,0.2);
        white-space: nowrap;
    }
    .ss-table th.left { text-align: left; }
    .ss-table td { 
        padding: 5px 8px; 
        text-align: center; 
        border-bottom: 1px solid rgba(136,135,128,0.1);
        white-space: nowrap;
    }
    .ss-table td.left { text-align: left; }
    .ss-table tr:hover td { background: rgba(136,135,128,0.06); }
    .rank-num { color: #aaa; font-size: 12px; font-weight: 400; margin-right: 8px; }
    .player-name { font-weight: 500; }
    .college-year { font-size: 11px; color: #999; }
    .dist-cell { color: #aaa; font-size: 11px; }
    .metric-val { font-size: 12px; color: #888; }
    .prospect-val { font-weight: 500; font-size: 13px; }
    </style>
    <table class="ss-table">
    <thead><tr>
        <th class="left">Player</th>
    """
    for col in col_order:
        html += f'<th>{col_display[col]}</th>'
    html += "</tr></thead><tbody>"

    for i, row in comps.iterrows():
        html += f"<tr>"
        college = row.get("college", "")
        draft_year = row.get("draft_year", "")
        sub = ""
        if college or draft_year:
            sub = f"<br><span class='college-year'>{college}{' · ' if college and draft_year else ''}{draft_year}</span>"
        html += f"<td class='left'><span class='rank-num'>{i+1}</span><span class='player-name'>{row['name']}</span>{sub}</td>"

        for col in col_order:
            if col == "similarity":
                sim = int(row["similarity"])
                c = sim / 100
                style = green_bg(c)
                html += f"<td><span style='{style}'>{sim}%</span></td>"
            elif col == "raw_distance":
                html += f"<td class='dist-cell'>{row['raw_distance']:.3f}</td>"
            else:
                p_val = prospect_height_in if col == "height" else prospect_vals.get(col, np.nan)
                db_val = row.get(col, np.nan)
                closeness = metric_diff_pct(p_val, db_val, col)
                style = green_bg(closeness) if (pd.notna(p_val) and p_val not in ("", None) and str(p_val) != "nan") else ""
                p_fmt = fmt(col, p_val)
                db_fmt = fmt(col, db_val)
                pct = row.get("percentiles", {})
                pct_val = pct.get(col, None) if isinstance(pct, dict) else None
                tooltip = f"title='{pct_val}th percentile at position'" if pct_val is not None else ""
                html += f"<td {tooltip}><span style='{style}'><span class='prospect-val'>{p_fmt}</span> <span class='metric-val'>| {db_fmt}</span></span></td>"

        html += "</tr>"

    html += "</tbody></table>"
    return html

# ── Line chart for player card ─────────────────────────────────────────────────
def build_comparison_chart(prospect_vals, comp_row, positions, archetypes_df):
    prospect_height_in = parse_height(prospect_vals.get("height")) if pd.notna(prospect_vals.get("height","")) and str(prospect_vals.get("height","")).strip() else np.nan

    pos = comp_row["position"]
    arch_row = archetypes_df[archetypes_df["position"] == pos]

    metric_keys = []
    metric_labels = []
    p_raws = []
    db_raws = []
    avgs = []
    p_labels = []
    db_labels = []
    lower_flags = []

    for m in METRICS:
        p_val = prospect_height_in if m == "height" else prospect_vals.get(m, np.nan)
        db_val = comp_row.get(m, np.nan)
        if pd.isna(p_val) or pd.isna(db_val):
            continue
        try:
            p_f = float(p_val)
            db_f = float(db_val)
        except:
            continue

        avg = None
        if not arch_row.empty:
            avg_col = f"{m}_avg"
            if avg_col in arch_row.columns:
                try:
                    avg = float(arch_row.iloc[0][avg_col])
                except:
                    avg = None
        if avg is None:
            avg = (p_f + db_f) / 2

        metric_keys.append(m)
        metric_labels.append(COL_NAMES[m])
        p_raws.append(p_f)
        db_raws.append(db_f)
        avgs.append(avg)
        p_labels.append(fmt(m, p_f))
        db_labels.append(fmt(m, db_f))
        lower_flags.append(m in LOWER_IS_BETTER)

    p_norm = []
    db_norm = []
    for i in range(len(p_raws)):
        avg = avgs[i]
        spread = max(abs(p_raws[i] - avg), abs(db_raws[i] - avg), avg * 0.05)
        pn = (p_raws[i] - avg) / spread * 10
        dn = (db_raws[i] - avg) / spread * 10
        if lower_flags[i]:
            pn = -pn
            dn = -dn
        p_norm.append(round(pn, 3))
        db_norm.append(round(dn, 3))

    prospect_name = st.session_state.get("prospect_name", "Prospect")
    comp_name = comp_row["name"]

    chart_html = f"""
    <div style="padding: 1rem 0 0.5rem;">
      <div style="display:flex; gap:20px; margin-bottom:1rem; flex-wrap:wrap;">
        <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;">
          <span style="width:20px;height:3px;background:#378ADD;display:inline-block;border-radius:2px;"></span>
          {prospect_name}
        </span>
        <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#555;">
          <span style="width:20px;height:3px;background:#639922;display:inline-block;border-radius:2px;"></span>
          {comp_name}
        </span>
        <span style="display:flex;align-items:center;gap:6px;font-size:12px;color:#aaa;">
          <span style="width:8px;height:8px;border-radius:50%;background:#aaa;display:inline-block;"></span>
          Position avg = midpoint
        </span>
      </div>
      <div style="position:relative;width:100%;height:300px;">
        <canvas id="compChartModal"></canvas>
      </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.js"></script>
    <script>
    (function(){{
      const labels = {json.dumps(metric_labels)};
      const pNorm = {json.dumps(p_norm)};
      const dNorm = {json.dumps(db_norm)};
      const pLabels = {json.dumps(p_labels)};
      const dLabels = {json.dumps(db_labels)};

      const customLabels = {{
        id: 'customLabels',
        afterDatasetsDraw(chart) {{
          const ctx = chart.ctx;
          chart.data.datasets.forEach((ds, di) => {{
            const meta = chart.getDatasetMeta(di);
            meta.data.forEach((pt, pi) => {{
              const lbl = di === 0 ? pLabels[pi] : dLabels[pi];
              const col = di === 0 ? '#185FA5' : '#3B6D11';
              const yOff = di === 0 ? -14 : 16;
              ctx.save();
              ctx.font = '11px sans-serif';
              ctx.fillStyle = col;
              ctx.textAlign = 'center';
              ctx.fillText(lbl, pt.x, pt.y + yOff);
              ctx.restore();
            }});
          }});
        }}
      }};

      if (window._compChart) {{ window._compChart.destroy(); }}
      window._compChart = new Chart(document.getElementById('compChartModal'), {{
        type: 'line',
        plugins: [customLabels],
        data: {{
          labels: labels,
          datasets: [
            {{
              label: '{prospect_name}',
              data: pNorm,
              borderColor: '#378ADD',
              backgroundColor: '#378ADD',
              pointBackgroundColor: '#378ADD',
              pointRadius: 6,
              tension: 0.3,
              borderWidth: 2.5
            }},
            {{
              label: '{comp_name}',
              data: dNorm,
              borderColor: '#639922',
              backgroundColor: '#639922',
              pointBackgroundColor: '#639922',
              pointRadius: 6,
              tension: 0.3,
              borderWidth: 2.5
            }}
          ]
        }},
        options: {{
          responsive: true,
          maintainAspectRatio: false,
          layout: {{ padding: {{ top: 28, bottom: 28 }} }},
          plugins: {{
            legend: {{ display: false }},
            tooltip: {{
              callbacks: {{
                label: (ctx) => {{
                  const arr = ctx.datasetIndex === 0 ? pLabels : dLabels;
                  return ' ' + ctx.dataset.label + ': ' + arr[ctx.dataIndex];
                }}
              }}
            }}
          }},
          scales: {{
            x: {{
              grid: {{ color: 'rgba(136,135,128,0.15)' }},
              ticks: {{ font: {{ size: 11 }}, color: '#999' }}
            }},
            y: {{ display: false, min: -18, max: 18 }}
          }}
        }}
      }});
    }})();
    </script>
    """
    return chart_html

# ── Export CSV ─────────────────────────────────────────────────────────────────
def build_export_df(comps, prospect_vals, prospect_name, positions):
    prospect_height_in = parse_height(prospect_vals.get("height")) if pd.notna(prospect_vals.get("height","")) and str(prospect_vals.get("height","")).strip() else np.nan
    rows = []
    for i, row in comps.iterrows():
        entry = {
            "rank": i + 1,
            "player": row["name"],
            "position": row["position"],
            "college": row.get("college", ""),
            "draft_year": row.get("draft_year", ""),
            "similarity_pct": int(row["similarity"]),
            "raw_distance": round(row["raw_distance"], 4),
        }
        for m in METRICS:
            p_val = prospect_height_in if m == "height" else prospect_vals.get(m, np.nan)
            db_val = row.get(m, np.nan)
            entry[f"prospect_{m}"] = fmt(m, p_val)
            entry[f"db_{m}"] = fmt(m, db_val)
        rows.append(entry)
    return pd.DataFrame(rows)

# ── Admin auth ─────────────────────────────────────────────────────────────────
def check_admin():
    try:
        admin_pw = st.secrets["ADMIN_PASSWORD"]
    except:
        admin_pw = "admin123"
    if "is_admin" not in st.session_state:
        st.session_state.is_admin = False
    if not st.session_state.is_admin:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Admin Login")
        pw = st.sidebar.text_input("Password", type="password", key="admin_pw_input")
        if st.sidebar.button("Login"):
            if pw == admin_pw:
                st.session_state.is_admin = True
                st.rerun()
            else:
                st.sidebar.error("Incorrect password")
    else:
        st.sidebar.markdown("---")
        st.sidebar.success("Logged in as Admin")
        if st.sidebar.button("Logout"):
            st.session_state.is_admin = False
            st.rerun()

# ── Admin panel ────────────────────────────────────────────────────────────────
def admin_panel():
    st.header("⚙️ Admin Panel")
    tab1, tab2 = st.tabs(["📤 Upload Dataset", "📐 Edit Archetypes"])

    with tab1:
        st.subheader("Upload Player Database")
        st.caption("CSV columns: name, position, height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle, college, draft_year")
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
                        st.success("Dataset saved.")
            except Exception as e:
                st.error(f"Error: {e}")

    with tab2:
        st.subheader("Position Archetypes & Weights")
        st.caption("rank1–rank4 use exact metric names: height, weight, arm_length, forty, vertical, broad_jump, three_cone, shuttle")
        archetypes_df = load_archetypes()
        if not archetypes_df.empty:
            edited = st.data_editor(archetypes_df, use_container_width=True, num_rows="fixed", key="arch_editor")
            if st.button("💾 Save Archetypes"):
                save_archetypes(edited)
                st.success("Archetypes saved.")

# ── User tool ──────────────────────────────────────────────────────────────────
def user_tool():
    players_df = load_players()
    archetypes_df = load_archetypes()

    if players_df.empty:
        st.warning("No player database loaded yet.")
        return

    # ── Input form ──
    with st.form("comp_form"):
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            name = st.text_input("Prospect Name")
            positions = st.multiselect("Position(s)", POSITIONS, default=["RB"])
            height = st.text_input(METRIC_LABELS["height"], placeholder="e.g. 6042")
            weight = st.number_input(METRIC_LABELS["weight"], min_value=0.0, max_value=500.0, value=0.0, step=1.0)
            arm_length = st.number_input(METRIC_LABELS["arm_length"], min_value=0.0, max_value=40.0, value=0.0, step=0.125)

        with col2:
            forty = st.number_input(METRIC_LABELS["forty"], min_value=0.0, max_value=7.0, value=0.0, step=0.01)
            vertical = st.number_input(METRIC_LABELS["vertical"], min_value=0.0, max_value=60.0, value=0.0, step=0.5)
            broad_jump = st.number_input(METRIC_LABELS["broad_jump"], min_value=0.0, max_value=160.0, value=0.0, step=1.0)
            three_cone = st.number_input(METRIC_LABELS["three_cone"], min_value=0.0, max_value=10.0, value=0.0, step=0.01)
            shuttle = st.number_input(METRIC_LABELS["shuttle"], min_value=0.0, max_value=7.0, value=0.0, step=0.01)

        with col3:
            st.markdown("**Draft Year Filter**")
            use_year_filter = st.checkbox("Filter by draft year range")
            year_min = st.number_input("From year", min_value=1980, max_value=2030, value=2000, step=1)
            year_max = st.number_input("To year", min_value=1980, max_value=2030, value=2025, step=1)

        submitted = st.form_submit_button("🏈 Find Comps", use_container_width=True)

    if submitted:
        if not name:
            st.warning("Please enter a prospect name.")
            return
        if not positions:
            st.warning("Please select at least one position.")
            return

        st.session_state["prospect_name"] = name

        prospect = {
            "height": height if str(height).strip() else np.nan,
            "weight": weight if weight > 0 else np.nan,
            "arm_length": arm_length if arm_length > 0 else np.nan,
            "forty": forty if forty > 0 else np.nan,
            "vertical": vertical if vertical > 0 else np.nan,
            "broad_jump": broad_jump if broad_jump > 0 else np.nan,
            "three_cone": three_cone if three_cone > 0 else np.nan,
            "shuttle": shuttle if shuttle > 0 else np.nan,
        }

        year_range = (int(year_min), int(year_max)) if use_year_filter else None
        pos_label = ", ".join(positions)

        with st.spinner("Running comparison..."):
            comps = compute_comps(prospect, positions, players_df, archetypes_df, year_range=year_range)

        if comps.empty:
            st.warning(f"No comparable players found for position(s): {pos_label}")
            return

        st.subheader(f"Top Comps for **{name}** ({pos_label})")
        st.caption(f"{name} | Past Player Metric Comps — hover metric cells for percentile rank. Similarity % relative to this result set.")

        # ── HTML table ──
        html = build_html_table(comps, prospect)
        st.components.v1.html(html, height=420, scrolling=True)

        # ── Export ──
        export_df = build_export_df(comps, prospect, name, positions)
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Export Results (CSV)",
            data=csv_bytes,
            file_name=f"signalscout_{name.replace(' ','_')}.csv",
            mime="text/csv"
        )

        # ── Player card drill-down ──
        st.markdown("---")
        st.markdown("**Click a player to open comparison chart:**")
        btn_cols = st.columns(min(len(comps), 5))
        for i, row in comps.iterrows():
            col_idx = i % 5
            with btn_cols[col_idx]:
                if st.button(f"{i+1}. {row['name']}", key=f"card_{i}"):
                    st.session_state["selected_comp"] = row.to_dict()
                    st.session_state["prospect_vals"] = prospect
                    st.session_state["positions"] = positions

        if "selected_comp" in st.session_state and "prospect_vals" in st.session_state:
            comp_row = st.session_state["selected_comp"]
            p_vals = st.session_state["prospect_vals"]
            st.markdown(f"### {name} vs. {comp_row['name']} ({comp_row.get('position','')})")
            college = comp_row.get("college", "")
            draft_year = comp_row.get("draft_year", "")
            if college or draft_year:
                st.caption(f"{college}{' · ' if college and draft_year else ''}{draft_year}")
            chart_html = build_comparison_chart(p_vals, comp_row, positions, archetypes_df)
            st.components.v1.html(chart_html, height=380, scrolling=False)
            if st.button("✕ Close card"):
                del st.session_state["selected_comp"]
                del st.session_state["prospect_vals"]
                st.rerun()

# ── Main ───────────────────────────────────────────────────────────────────────
def main():
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
