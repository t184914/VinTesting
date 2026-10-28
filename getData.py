# pip install ucimlrepo
from pathlib import Path
from ucimlrepo import fetch_ucirepo

wine = fetch_ucirepo(id=186)
df = wine.data.original          # features + quality + color i én tabell
print(df["color"].unique())      # sjekk hvilke verdier som finnes (red / white)

Path("data").mkdir(exist_ok=True)
for color in ["red", "white"]:
    part = df[df["color"] == color].drop(columns="color")
    part.to_csv(f"data/winequality-{color}.csv", index=False)
    print(color, len(part))