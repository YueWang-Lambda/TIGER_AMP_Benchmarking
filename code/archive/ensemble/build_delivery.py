from pathlib import Path
import json
DATA=json.loads((Path(__file__).resolve().parents[1]/"figure_legends.json").read_text())
LEGENDS=DATA["figures"]
DISCUSSION=DATA["shared_mechanistic_legend"]
INTRO=DATA["intro"]
REF1,REF2=DATA["references"]
