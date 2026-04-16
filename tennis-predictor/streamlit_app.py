"""
Streamlit interface for tennis match prediction.

Run:
    streamlit run streamlit_app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

from models.predict import _build_player_stats_snapshot, build_match_feature_vector


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "best_model.pkl"
FEATURES_CSV = BASE_DIR / "data" / "processed" / "features.csv"

SURFACE_OPTIONS = ["Clay", "Hard", "Grass", "Carpet", "Unknown"]
ROUND_OPTIONS = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
LEVEL_OPTIONS = ["Grand Slam", "Masters", "ATP500", "Davis Cup", "Tour Finals", "Challenger"]


@st.cache_resource
def load_model_bundle() -> dict:
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_data
def load_features_history() -> pd.DataFrame:
    if not FEATURES_CSV.exists():
        return pd.DataFrame()
    return pd.read_csv(FEATURES_CSV, parse_dates=["date"])


def predict_one_match(
    model,
    feature_cols: list[str],
    features_df: pd.DataFrame,
    p1: str,
    p2: str,
    surface: str,
    round_name: str,
    level: str,
    match_date: pd.Timestamp,
) -> dict:
    player_stats = _build_player_stats_snapshot(features_df, match_date) if not features_df.empty else {}

    X = build_match_feature_vector(
        p1_name=p1,
        p2_name=p2,
        surface=surface,
        round_name=round_name,
        tourney_level=level,
        cutoff_date=match_date,
        player_stats=player_stats,
        features_df=features_df,
        feature_cols=feature_cols,
    )

    proba = model.predict_proba(X)[0]
    p1_win_prob = float(proba[1])
    p2_win_prob = 1.0 - p1_win_prob
    winner = p1 if p1_win_prob >= 0.5 else p2
    winner_prob = p1_win_prob if p1_win_prob >= 0.5 else p2_win_prob

    return {
        "player1": p1,
        "player2": p2,
        "surface": surface,
        "round": round_name,
        "level": level,
        "match_date": match_date.date().isoformat(),
        "predicted_winner": winner,
        "winner_confidence": winner_prob,
        "p1_win_probability": p1_win_prob,
        "p2_win_probability": p2_win_prob,
    }


def validate_batch_columns(df: pd.DataFrame) -> tuple[bool, str]:
    required = ["player1", "player2", "surface", "round", "level", "date"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {missing}"
    return True, ""


def render_single_match_tab(model, feature_cols: list[str], features_df: pd.DataFrame) -> None:
    st.subheader("Single Match Prediction")
    st.write("Fill in one matchup and get probabilities instantly.")

    with st.form("single_match_form"):
        col1, col2 = st.columns(2)
        with col1:
            p1 = st.text_input("Player 1", placeholder="Jannik Sinner")
            surface = st.selectbox("Surface", SURFACE_OPTIONS, index=0)
            round_name = st.selectbox("Round", ROUND_OPTIONS, index=4)
        with col2:
            p2 = st.text_input("Player 2", placeholder="Carlos Alcaraz")
            level = st.selectbox("Tournament Level", LEVEL_OPTIONS, index=0)
            date_value = st.date_input("Match Date", value=pd.Timestamp.today().date())

        submitted = st.form_submit_button("Predict Match")

    if not submitted:
        return

    p1 = p1.strip()
    p2 = p2.strip()

    if not p1 or not p2:
        st.error("Both players are required.")
        return
    if p1.lower() == p2.lower():
        st.error("Player 1 and Player 2 must be different.")
        return

    result = predict_one_match(
        model=model,
        feature_cols=feature_cols,
        features_df=features_df,
        p1=p1,
        p2=p2,
        surface=surface,
        round_name=round_name,
        level=level,
        match_date=pd.Timestamp(date_value),
    )

    st.success("Prediction completed.")
    metric_col1, metric_col2, metric_col3 = st.columns(3)
    metric_col1.metric("Predicted Winner", result["predicted_winner"])
    metric_col2.metric("Winner Confidence", f"{result['winner_confidence']:.1%}")
    metric_col3.metric("Match Date", result["match_date"])

    probs_df = pd.DataFrame(
        {
            "player": [result["player1"], result["player2"]],
            "win_probability": [result["p1_win_probability"], result["p2_win_probability"]],
        }
    )
    st.dataframe(
        probs_df.style.format({"win_probability": "{:.2%}"}),
        use_container_width=True,
        hide_index=True,
    )


def build_batch_template() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "player1": "Jannik Sinner",
                "player2": "Carlos Alcaraz",
                "surface": "Clay",
                "round": "QF",
                "level": "Grand Slam",
                "date": "2026-06-01",
            },
            {
                "player1": "Alexander Zverev",
                "player2": "Novak Djokovic",
                "surface": "Hard",
                "round": "SF",
                "level": "Masters",
                "date": "2026-03-29",
            },
        ]
    )


def render_batch_tab(model, feature_cols: list[str], features_df: pd.DataFrame) -> None:
    st.subheader("Batch Prediction")
    st.write("Upload a CSV or edit rows directly, then predict all matches in one click.")

    template_df = build_batch_template()
    st.download_button(
        "Download CSV Template",
        data=template_df.to_csv(index=False).encode("utf-8"),
        file_name="match_template.csv",
        mime="text/csv",
    )

    uploaded = st.file_uploader("Upload Matches CSV", type=["csv"])
    if uploaded is not None:
        input_df = pd.read_csv(uploaded)
    else:
        input_df = template_df.copy()

    edited_df = st.data_editor(
        input_df,
        use_container_width=True,
        num_rows="dynamic",
        column_config={
            "surface": st.column_config.SelectboxColumn("surface", options=SURFACE_OPTIONS),
            "round": st.column_config.SelectboxColumn("round", options=ROUND_OPTIONS),
            "level": st.column_config.SelectboxColumn("level", options=LEVEL_OPTIONS),
        },
    )

    if not st.button("Predict All Matches", type="primary"):
        return

    ok, error = validate_batch_columns(edited_df)
    if not ok:
        st.error(error)
        return

    run_df = edited_df.copy()
    run_df = run_df.dropna(subset=["player1", "player2", "surface", "round", "level", "date"])
    if run_df.empty:
        st.error("No valid rows to predict. Fill at least one complete row.")
        return

    outputs: list[dict] = []
    for i, row in run_df.iterrows():
        try:
            match_date = pd.Timestamp(row["date"])
            if pd.isna(match_date):
                raise ValueError("invalid date")

            prediction = predict_one_match(
                model=model,
                feature_cols=feature_cols,
                features_df=features_df,
                p1=str(row["player1"]).strip(),
                p2=str(row["player2"]).strip(),
                surface=str(row["surface"]).strip(),
                round_name=str(row["round"]).strip(),
                level=str(row["level"]).strip(),
                match_date=match_date,
            )
            prediction["row_index"] = int(i)
            outputs.append(prediction)
        except Exception as exc:
            outputs.append(
                {
                    "row_index": int(i),
                    "player1": row.get("player1", ""),
                    "player2": row.get("player2", ""),
                    "predicted_winner": "ERROR",
                    "winner_confidence": None,
                    "p1_win_probability": None,
                    "p2_win_probability": None,
                    "error": str(exc),
                }
            )

    out_df = pd.DataFrame(outputs)
    st.success(f"Predicted {len(out_df)} rows.")

    display_cols = [
        "row_index",
        "player1",
        "player2",
        "predicted_winner",
        "winner_confidence",
        "p1_win_probability",
        "p2_win_probability",
    ]
    if "error" in out_df.columns:
        display_cols.append("error")

    styled = out_df[display_cols].style.format(
        {
            "winner_confidence": "{:.2%}",
            "p1_win_probability": "{:.2%}",
            "p2_win_probability": "{:.2%}",
        },
        na_rep="",
    )
    st.dataframe(styled, use_container_width=True, hide_index=True)

    st.download_button(
        "Download Predictions CSV",
        data=out_df.to_csv(index=False).encode("utf-8"),
        file_name="match_predictions.csv",
        mime="text/csv",
    )


def main() -> None:
    st.set_page_config(page_title="Tennis Match Predictor", page_icon="T", layout="wide")
    st.title("Tennis Match Predictor")
    st.caption("Predict one match or many matches with your trained model bundle.")

    if not MODEL_PATH.exists():
        st.error("Model not found at models/best_model.pkl. Train first: python models/train.py")
        st.stop()

    try:
        bundle = load_model_bundle()
    except Exception as exc:
        st.error(f"Could not load model bundle: {exc}")
        st.stop()

    model = bundle.get("model")
    feature_cols = bundle.get("feature_cols")
    model_name = bundle.get("name", "Unknown")

    if model is None or feature_cols is None:
        st.error("Invalid model bundle format. Expected keys: model, feature_cols, name.")
        st.stop()

    features_df = load_features_history()

    info_col1, info_col2 = st.columns(2)
    info_col1.info(f"Loaded model: {model_name}")
    info_col2.info(
        f"Feature history rows: {len(features_df)}"
        if not features_df.empty
        else "Feature history not found. Unknown players will use neutral defaults."
    )

    tab_single, tab_batch = st.tabs(["Single Match", "Batch Matches"])
    with tab_single:
        render_single_match_tab(model, feature_cols, features_df)
    with tab_batch:
        render_batch_tab(model, feature_cols, features_df)


if __name__ == "__main__":
    main()
