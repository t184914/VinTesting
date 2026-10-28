# Er vinen god? Prediksjon av vinkvalitet med maskinlæring

Sluttprosjekt i maskinlæring. Vi har laget en nettside der brukeren stiller inn 11 kjemiske målinger for en vin og får et anslag på om vinen blir vurdert som **lav**, **middels** eller **høy** kvalitet.

**Nettside:** (https://vintestinggit.streamlit.app/)
**Rapport:** [legg inn filnavnet på rapporten i repoet, f.eks. `Rapport.pdf`]

**Gruppe:** [navn 1], [navn 2]

## Hva prosjektet gjør

- Data: Wine Quality fra UCI (rød og hvit vinho verde), 6497 viner, 5320 etter at duplikater er fjernet.
- Oppgave: klassifisering i tre klasser (lav = kvalitet 3-5, middels = 6, høy = 7-9).
- Modeller: dummy-baseline, logistisk regresjon, Random Forest og Gradient Boosting, tunet med kryssvalidering.
- Beste modell: Random Forest.
- Nettside (Streamlit): skyveknapper og tallfelt for hver egenskap, predikert klasse med sannsynligheter, feature importance og en «hva hvis»-analyse.

## Resultater (testsett, 1064 viner)

| Modell | Test accuracy | Test macro-F1 |
|---|---|---|
| Random Forest | 0,595 | **0,595** |
| Gradient Boosting | 0,583 | 0,566 |
| Logistisk regresjon | 0,550 | 0,546 |
| Dummy (vanligste klasse) | 0,437 | 0,203 |

Alkohol, tetthet og flyktig syre er de viktigste egenskapene for modellen. Se rapporten for mer detaljert evaluering og feilanalyse.

## Innhold i repoet

| Fil / mappe | Innhold |
|---|---|
| `getData.py` | Henter datasettet fra UCI og lagrer det som CSV i `data/` |
| `Train Wine.py` | Renser data, trener og tuner modellene, evaluerer dem og lagrer den beste |
| `Test.py` | Skriver ut feature importance for den lagrede modellen |
| `App.py` | Streamlit-nettsiden |
| `data/` | `winequality-red.csv` og `winequality-white.csv` |
| `output/` | Lagret modell (`best_model.joblib`), `feature_ranges.json`, `model_comparison.csv` og confusion matrix-figurer |
| `requirements.txt` | Python-biblioteker som trengs |

## Slik kjører du prosjektet

Krever Python 3.10 eller nyere.

**1. Last ned repoet og installer bibliotekene**

```bash
git clone https://github.com/t184914/VinTesting.git
cd VinTesting
pip install -r requirements.txt
```

**2. (Valgfritt) Hent dataene på nytt**

CSV-filene ligger allerede i `data/`. Vil du hente dem fra UCI selv:

```bash
python getData.py
```

**3. (Valgfritt) Tren modellene på nytt**

Den ferdigtrente modellen ligger i `output/`. For å gjenskape resultatene:

```bash
python "Train Wine.py" --red data/winequality-red.csv --white data/winequality-white.csv
```

Dette skriver ut modellsammenligningen og lagrer `best_model.joblib`, `feature_ranges.json`, `model_comparison.csv` og confusion matrix-figurene i `output/`. Tuningen kan ta noen minutter.

**4. Start nettsiden**

```bash
streamlit run App.py
```

Nettleseren åpner seg automatisk (vanligvis på `http://localhost:8501`).

## Datakilde og referanser

- Cortez, P., Cerdeira, A., Almeida, F., Matos, T. og Reis, J. (2009). *Modeling wine preferences by data mining from physicochemical properties.* Decision Support Systems.
- UCI Machine Learning Repository: [Wine Quality (id 186)](https://archive.ics.uci.edu/dataset/186/wine+quality)
- Verktøy: pandas, scikit-learn, matplotlib, joblib, ucimlrepo og Streamlit.

## Bruk av AI-verktøy

Vi har brukt Claude (Anthropic) som hjelp til idéutvikling og til å skrive utkast til kode og rapport. [Beskriv kort hva dere selv gjorde og forsto, og hva dere endret eller sjekket.]

## Begrensninger

Kvalitetspoengene er subjektive vurderinger fra smakspaneler, og dataene gjelder bare portugisiske vinho verde-viner. Modellen er et anslag og ikke en erstatning for en smaksvurdering.
