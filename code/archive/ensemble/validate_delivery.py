"""Data-dependent V17 ensemble acceptance; no fixed historical score targets."""
from pathlib import Path
import json,subprocess
import numpy as np
import pandas as pd
from docx import Document
OUT=Path(__file__).resolve().parents[1];PARENT=OUT.parent
load=lambda p:json.loads(Path(p).read_text())
nb=load(PARENT/'S4_complete_analysis_and_figures_Fig5_style.ipynb')
added=[c for c in nb['cells'] if c.get('metadata',{}).get('tiger_ensemble_addendum')]
standalone=load(OUT/'TIGER_ensemble_figures.ipynb');assert added==standalone['cells']
for c in added:
 if c['cell_type']=='code':assert c['execution_count'] is not None and not any(o['output_type']=='error' for o in c['outputs'])
images=sum('image/png' in o.get('data',{}) for c in added if c['cell_type']=='code' for o in c['outputs']);assert images==14
for stem in ['benchmark_per_seed','benchmark_family_per_seed','benchmark_summary']:
 a=pd.read_csv(PARENT/(stem+'.csv'));b=pd.read_csv(OUT/'tables'/(stem+'.csv'));b=b[b.method!='TIGER-ensemble']
 pd.testing.assert_frame_equal(a,b[a.columns].reset_index(drop=True),check_exact=False,atol=1e-12,rtol=0)
for seed in range(120,130):
 a=pd.read_csv(OUT/f'predictions/TIGER_ensemble_seed{seed}.csv',float_precision='round_trip')
 b=pd.read_csv(PARENT/f'source_snapshot/predictions/tiger/s4_single_n1n2_seed{seed}.csv',float_precision='round_trip')
 assert np.array_equal(a['TIGER-single'],b.y_pred)
 assert np.allclose(a['TIGER-ensemble'],(a['TIGER-pair']+a['TIGER-single'])/2,atol=1e-14,rtol=0)
timing=load(OUT/'timing/result.json');assert timing['status']=='PASS' and len(timing['checks'])==21 and len(timing['repeats_s'])==20
assert np.isclose(timing['median_s'],np.median(timing['repeats_s']),atol=1e-12,rtol=0)
fonts={}
for p in sorted((OUT/'figures').rglob('*.pdf')):
 names=[line.split()[0] for line in subprocess.check_output(['pdffonts',str(p)],text=True).splitlines()[2:] if line.strip()]
 assert names and all('Arial' in n for n in names)
 for ext in ['png','tiff','svg']:assert p.with_suffix('.'+ext).exists()
 fonts[str(p.relative_to(OUT))]=names
assert len(fonts)==14
assert all(not x['off_canvas_text'] for x in load(OUT/'figures/figure_layout_checks.json').values())
assert all(x['label_or_marker_collisions']==0 for x in load(OUT/'figures/cost_label_layout_checks.json').values())
for lang in ['EN','ZH']:
 p=OUT/f'TIGER_ensemble_addendum_{lang}.docx';assert p.with_suffix('.md').exists()
 assert len(Document(p).inline_shapes)==6
result=dict(status='PASS',version='V17',validation_only=load(PARENT/'reconstruction_acceptance.json')['validation_only'],images=images,matched_seed_ensemble=True,all_original_benchmark_rows_preserved=True,fonts=fonts)
(OUT/'delivery_validation.json').write_text(json.dumps(result,indent=2));print('V17 ensemble delivery PASS')
