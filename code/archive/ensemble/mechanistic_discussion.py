"""V17 discussion is generated from current results by refresh_ensemble.py."""
from pathlib import Path
import json
def discussion(*args):
 return json.loads((Path(__file__).resolve().parents[1]/"figure_legends.json").read_text())["shared_mechanistic_legend"]
