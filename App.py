import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

OUT = Path(__file__).parent / "output"

# Norske navn og korte forklaringer til skyvene
FEATURE_INFO = {
    "fixed_acidity": ("Fast syre (fixed acidity)", "g/L, syrer som ikke fordamper lett"),
    "volatile_acidity": ("Flyktig syre (volatile acidity)", "g/L, for mye gir eddiksmak"),
    "citric_acid": ("Sitronsyre (citric acid)", "g/L, gir friskhet"),
    "residual_sugar": ("Restsukker (residual sugar)", "g/L, sukker igjen etter gjæring"),
    "chlorides": ("Klorider (chlorides)", "g/L, saltinnhold"),
    "free_sulfur_dioxide": ("Fritt svoveldioksid", "mg/L"),
    "total_sulfur_dioxide": ("Totalt svoveldioksid", "mg/L"),
    "density": ("Tetthet (density)", "g/cm³"),
    "pH": ("pH", "surhetsgrad"),
    "sulphates": ("Sulfater (sulphates)", "g/L"),
    "alcohol": ("Alkohol", "volumprosent"),
}
COLOR_MAP = {"Rød": 0, "Hvit": 1}
CLASS_COLORS = {"lav": "#d9534f", "middels": "#f0ad4e", "høy": "#5cb85c"}


@st.cache_resource
def load_artifacts():
    bundle = joblib.load(OUT / "best_model.joblib")
    ranges = json.loads((OUT / "feature_ranges.json").read_text())
    return bundle, ranges


def get_importances(model, features):
    """Feature importance for trebaserte modeller, ellers |koeffisient| for lineære."""
    if hasattr(model, "feature_importances_"):
        vals = model.feature_importances_
    elif hasattr(model, "named_steps") and hasattr(model.named_steps["clf"], "coef_"):
        vals = np.abs(model.named_steps["clf"].coef_).mean(axis=0)
        vals = vals / vals.sum()
    else:
        return None
    return pd.Series(vals, index=features).sort_values()


st.set_page_config(page_title="Vinkvalitet", page_icon="🍷", layout="wide")
st.title("🍷 Hvor god er denne vinen?")
st.write(
    "Juster de kjemiske målingene til venstre. En maskinlæringsmodell trent på "
    "ca. 5300 portugisiske «vinho verde»-viner anslår om vinen blir vurdert som "
    "**lav**, **middels** eller **høy** kvalitet."
)

try:
    bundle, ranges = load_artifacts()
except FileNotFoundError:
    st.error(
        "Fant ikke `output/best_model.joblib`. Kjør `train_wine.py` først, "
        "slik at modellen og feature_ranges.json blir laget."
    )
    st.stop()

model = bundle["model"]
features = bundle["features"]
labels = bundle["labels"]

# ---------- Sidepanel: input ----------
st.sidebar.header("Målinger")
color_name = st.sidebar.radio("Type vin", list(COLOR_MAP), horizontal=True)

numeric_features = [f for f in features if f != "color"]


def reset_all():
    """Sett alle skyver og tallfelt tilbake til gjennomsnittet."""
    for name in numeric_features:
        m = float(round(ranges[name]["mean"], 4))
        st.session_state[f"sl_{name}"] = m
        st.session_state[f"ni_{name}"] = m


def from_slider(name):
    st.session_state[f"ni_{name}"] = st.session_state[f"sl_{name}"]


def from_number(name):
    lo, hi = float(ranges[name]["min"]), float(ranges[name]["max"])
    # Hold verdien innenfor området modellen er trent på
    v = min(max(st.session_state[f"ni_{name}"], lo), hi)
    st.session_state[f"ni_{name}"] = v
    st.session_state[f"sl_{name}"] = v


st.sidebar.button("Tilbakestill til gjennomsnitt", on_click=reset_all)
st.sidebar.caption("Dra i skyven eller skriv inn tallet i feltet ved siden av.")

values = {}
for f in features:
    if f == "color":
        values[f] = COLOR_MAP[color_name]
        continue
    lo, mean, hi = float(ranges[f]["min"]), float(ranges[f]["mean"]), float(ranges[f]["max"])
    title, help_text = FEATURE_INFO.get(f, (f, ""))
    step = float((hi - lo) / 200) or 0.01
    start = float(round(mean, 4))
    # Første gang: sett startverdi i begge widgets
    st.session_state.setdefault(f"sl_{f}", start)
    st.session_state.setdefault(f"ni_{f}", start)

    st.sidebar.markdown(f"**{title}**", help=help_text)
    col_slider, col_number = st.sidebar.columns([3, 2])
    with col_slider:
        st.slider(
            title,
            min_value=lo,
            max_value=hi,
            step=step,
            key=f"sl_{f}",
            on_change=from_slider,
            args=(f,),
            label_visibility="collapsed",
        )
    with col_number:
        st.number_input(
            title,
            min_value=lo,
            max_value=hi,
            step=step,
            format="%.5g",
            key=f"ni_{f}",
            on_change=from_number,
            args=(f,),
            label_visibility="collapsed",
        )
    values[f] = st.session_state[f"sl_{f}"]

X_input = pd.DataFrame([values])[features]

# ---------- Prediksjon ----------
proba = model.predict_proba(X_input)[0]
pred_idx = int(np.argmax(proba))
pred_label = labels[pred_idx]

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Prediksjon")
    st.markdown(
        f"<div style='font-size:2.2rem;font-weight:700;color:{CLASS_COLORS[pred_label]}'>"
        f"{pred_label.upper()} kvalitet</div>",
        unsafe_allow_html=True,
    )
    st.caption(f"Modellens sikkerhet: {proba[pred_idx]:.0%}")
    st.bar_chart(
        pd.DataFrame({"Sannsynlighet": proba}, index=labels),
        horizontal=True,
    )
    st.caption(
        "Lav = kvalitet 3-5, middels = 6, høy = 7-9 (på skalaen 0-10 fra "
        "smakspaneler). Modellen er ikke en fasit, men et anslag basert på "
        "kjemien alene."
    )

with col2:
    st.subheader("Hva betyr mest for modellen?")
    imp = get_importances(model, features)
    if imp is not None:
        imp = imp.drop(labels=["color"], errors="ignore")
        imp.index = [FEATURE_INFO.get(i, (i,))[0].split(" (")[0] for i in imp.index]
        st.bar_chart(imp, horizontal=True)
        st.caption(
            "Generell viktighet i hele modellen, ikke for akkurat denne vinen."
        )

# ---------- Hva-hvis ----------
st.divider()
st.subheader("Hva hvis...?")
st.write(
    "Velg én egenskap og se hvordan sannsynlighetene endrer seg når bare den "
    "varierer, mens alt annet står som i skyvene."
)
numeric = [f for f in features if f != "color"]
what_if = st.selectbox(
    "Egenskap",
    numeric,
    format_func=lambda f: FEATURE_INFO.get(f, (f,))[0],
    index=numeric.index("alcohol") if "alcohol" in numeric else 0,
)
grid = np.linspace(ranges[what_if]["min"], ranges[what_if]["max"], 60)
sweep = pd.concat([X_input] * len(grid), ignore_index=True)
sweep[what_if] = grid
sweep_proba = pd.DataFrame(model.predict_proba(sweep), columns=labels)
sweep_proba.index = np.round(grid, 3)
sweep_proba.index.name = FEATURE_INFO.get(what_if, (what_if,))[0]
st.line_chart(sweep_proba)
st.caption(
    "Merk at egenskapene henger sammen i virkeligheten (f.eks. alkohol og "
    "tetthet). Her endres bare én om gangen, så ekstreme kombinasjoner kan "
    "være urealistiske."
)

with st.expander("Om modellen og dataene"):
    st.write(
        "- Data: Wine Quality (UCI, Cortez et al. 2009), rød og hvit vinho verde.\n"
        "- Duplikater er fjernet. Klassene er ubalanserte (flest «middels»).\n"
        "- Modell: den beste av fire testede, valgt på macro-F1 på et "
        "uavhengig testsett.\n"
        "- Begrensninger: smakspoeng er subjektive, og dataene gjelder bare "
        "portugisiske viner."
    )