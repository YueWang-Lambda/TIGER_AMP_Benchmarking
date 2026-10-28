"""Portable reviewer figures from the independently reconstructed result tables.

The publication's original PDF exports are preserved in figures/reference.
These editable plots regenerate the same data with explicit condition labels.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.lines import Line2D

from evaluate import ROOT, METRICS, pareto, read, require

DISPLAY={'Das':'Das2021','SenseXAMP':'SenseXAMP-Ecoli','tsAMP':'tsAMP-CS'}
CATEGORIES={
    'BERT-AmPEP60':'PLM fine-tuning','LLAMP':'PLM fine-tuning',
    **{m:'Frozen PLM + head' for m in ['HyperAMP','SenseXAMP','Deep-PIE','AMPGen','AMPpredictor','tsAMP']},
    'AMP-Designer':'Generative LM scorer',
    **{m:'Sequence DL' for m in ['AMPlify','Das','APEX','DeepAMP','HydrAMP','EvoGradient','ANIA']},
    **{m:'Physicochemical models' for m in ['MBC-Attention','EIPpred','AMPActiPred']},
    'AMPLiT':'Sequence + physicochemical DL',
    'TIGER-single':'Multimodal regression','GGN-ABPMIC':'Multimodal regression',
    'TIGER-pair':'Multimodal metric learning','TIGER-ensemble':'TIGER ensemble',
}
COLORS=dict(zip(['PLM fine-tuning','Frozen PLM + head','Generative LM scorer','Sequence DL','Physicochemical models','Sequence + physicochemical DL','Multimodal regression','Multimodal metric learning','TIGER ensemble'],['#7C6BD9','#43A047','#F97316','#318BFA','#009E9A','#C15D9E','#202020','#202020','#D55E00']))
for font in Path('/usr/share/fonts/truetype/msttcorefonts').glob('Arial*.ttf'):
    font_manager.fontManager.addfont(str(font))
FONT='Arial' if any(f.name=='Arial' for f in font_manager.fontManager.ttflist) else 'DejaVu Sans'
plt.rcParams.update({'font.family':FONT,'font.size':11,'axes.labelsize':12,'axes.titlesize':14,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})


def name(m): return DISPLAY.get(m,m)
def color(m): return COLORS[CATEGORIES[m]]
def marker(m): return 'D' if m=='TIGER-pair' else ('s' if m=='TIGER-single' else ('*' if m=='TIGER-ensemble' else 'o'))


def export(fig, path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(path.with_suffix('.'+ext),dpi=200,facecolor='white')
    return fig


def benchmark(data, metric='PCC', selected=False, ensemble=False):
    summary=read(data/'benchmark_summary.csv')
    if selected:
        choices=read(data/'pcc_selected.csv')[['method','condition']]
        d=summary.merge(choices,on=['method','condition'],validate='many_to_one')
        heading='PCC-selected training condition'
    else:
        d=summary[summary.condition=='with_negatives'];heading='With-negative training condition'
    if ensemble:
        d=pd.concat([d,read(data/'ensemble_summary.csv')],ignore_index=True)
    d=d[d.metric==metric].sort_values('mean',ascending=metric in ['RSE','log10MAE'],na_position='last')
    fig,ax=plt.subplots(figsize=(9.5,10.7));fig.subplots_adjust(left=.28,right=.93,bottom=.12,top=.90)
    for y,r in enumerate(d.itertuples()):
        if np.isfinite(r.mean):
            ax.errorbar(r.mean,y,xerr=[[r.mean-r.ci_low],[r.ci_high-r.mean]],fmt=marker(r.method),color=color(r.method),ms=7,capsize=3,lw=1.3)
        else:
            ax.text(.98,y,'NA*',transform=ax.get_yaxis_transform(),va='center',ha='right')
        if r.method.startswith('TIGER'):ax.axhspan(y-.4,y+.4,color='#E4F9DF',zorder=-1)
    ax.set(yticks=range(len(d)),yticklabels=[name(m) for m in d.method],ylim=(len(d)-.4,-.6),xlabel=metric+(' (lower is better)' if metric in ['RSE','log10MAE'] else ' (higher is better)'))
    if metric=='RSE':ax.set_xscale('log')
    ax.set_title(f'{heading}\nFamily-macro {metric}',loc='left',pad=16)
    fig.text(.05,.054,'Mean and 95% t interval across ten seeds; 11 equally weighted families.',fontsize=10)
    fig.text(.05,.028,'* Constant AMPLiT predictions make one seed correlation undefined; all seeds are retained.',fontsize=9)
    return fig


def place_labels(ax, points):
    """Place numeric method IDs in display coordinates and avoid label collisions."""
    fig=ax.figure;fig.canvas.draw();renderer=fig.canvas.get_renderer();boxes=[]
    for x,y,label in points:
        placed=None
        for radius in [6,13,20,29,39,50]:
            for dx,dy in [(radius,radius),(-radius,radius),(radius,-radius),(-radius,-radius),(0,radius),(radius,0),(-radius,0),(0,-radius)]:
                t=ax.annotate(label,(x,y),xytext=(dx,dy),textcoords='offset points',fontsize=8,ha='center',va='center',color='#414141',arrowprops=dict(arrowstyle='-',color='#AAAAAA',lw=.45) if radius>=20 else None)
                box=t.get_window_extent(renderer).expanded(1.08,1.12)
                # Arrow bounding boxes include the point, so use the text box only.
                from matplotlib.text import Text
                box=Text.get_window_extent(t,renderer).expanded(1.10,1.18)
                if ax.bbox.contains(box.x0,box.y0) and ax.bbox.contains(box.x1,box.y1) and not any(box.overlaps(b) for b in boxes):
                    boxes.append(box);placed=t;break
                t.remove()
            if placed is not None:break
        if placed is None:
            # Caller can increase figsize; fail instead of exporting hidden labels.
            raise RuntimeError(f'No non-overlapping position for method ID {label}')


def cost(data, metric='PCC', both=False, ensemble=False):
    d=read(data/('all_conditions.csv' if both else 'pcc_selected.csv'))
    if ensemble:
        e=read(data/'ensemble_summary.csv').set_index('metric')['mean']
        t=json.loads((ROOT/'results/timing/TIGER-ensemble.json').read_text())
        require(len(t['repeats_s'])==20,'Incomplete ensemble timing')
        seconds=float(np.median(t['repeats_s']))
        require(np.isclose(seconds,t['median_s'],atol=1e-12,rtol=0),'Ensemble timing differs')
        d=pd.concat([d,pd.DataFrame([dict(method='TIGER-ensemble',condition='with_negatives',**e.to_dict(),time_s_no_struct=seconds,structure_increment_s=328.12,time_s_with_struct=seconds+328.12)])],ignore_index=True)
    methods=sorted(d.method.unique());ids={m:i+1 for i,m in enumerate(methods)}
    fig=plt.figure(figsize=(12.6,8.8));ax=fig.add_axes([.08,.32,.61,.58])
    ax.set_xscale('log');ax.set_xlim(.001,1000)
    if metric=='PCC':ax.set_ylim(.18,.73)
    else:ax.set_ylim(0,max(1.5,float(d.RSE.max())*1.06))
    ax.set(xlabel='Time for external workload (s)',ylabel=f'Family-macro {metric}')
    ax.set_title(('Both training conditions' if both else 'PCC-selected condition')+(' + ensemble' if ensemble else ''),loc='left',pad=12)
    labels=[]
    if both:
        for method,g in d.groupby('method'):
            valid=g[np.isfinite(g[metric])]
            ax.plot(valid.time_s_with_struct,valid[metric],color=color(method),lw=1,alpha=.6,zorder=1)
    for r in d.itertuples():
        value=getattr(r,metric)
        if not np.isfinite(value):continue
        filled=not both or r.condition=='with_negatives'
        ax.scatter(r.time_s_with_struct,value,s=80,marker=marker(r.method),edgecolor=color(r.method),facecolor=color(r.method) if filled else 'white',linewidth=1.2,zorder=4)
        # In the two-condition plot label one endpoint per method; connectors identify the other.
        g=d[d.method==r.method]
        label_condition=g.loc[g[metric].notna()].sort_values(metric,ascending=False).iloc[0].condition
        if not both or r.condition==label_condition:labels.append((r.time_s_with_struct,value,str(ids[r.method])))
        if not both and r.structure_increment_s>0:
            ax.scatter(r.time_s_no_struct,value,s=80,marker=marker(r.method),facecolor='white',edgecolor=color(r.method),linewidth=1.2,zorder=3)
            ax.plot([r.time_s_no_struct,r.time_s_with_struct],[value,value],ls='--',lw=.8,color=color(r.method),alpha=.55)
    front=pareto(d,metric)
    ax.step(front.time_s_with_struct,front[metric],where='post',color='#737373',lw=1.1,zorder=0)
    place_labels(ax,labels)
    for i,m in enumerate(methods):fig.text(.74,.90-i*.026,f'{ids[m]:2d}  {name(m)}',fontsize=9,color=color(m))
    handles=[Line2D([],[],marker='o',lw=0,mfc=c,mec=c,label=k) for k,c in COLORS.items() if k in set(CATEGORIES[m] for m in methods)]
    fig.legend(handles=handles,loc='lower left',bbox_to_anchor=(.05,.115),ncol=3,frameon=False,fontsize=9)
    explanation='Filled: with negatives; hollow: without negatives. Lines connect conditions; structures-ready points omitted.' if both else 'Filled: includes structure generation where needed; hollow: structures ready. Training condition chosen by PCC.'
    fig.text(.06,.10,explanation,fontsize=9)
    fig.text(.06,.072,'Both conditions use the measured with-negative seed-123 time coordinate; 20 timed repetitions.',fontsize=9)
    fig.text(.06,.044,'Structure increment: 328.12 s for 208 peptides. Serial estimate from 30 measured peptides.',fontsize=9)
    fig.text(.06,.016,'Gray line: nondominated displayed full-workload points. Sub-10-ms time differences may reflect timing noise.',fontsize=9)
    return fig


def run(data, output):
    data,output=Path(data),Path(output)
    require((data/'validation.json').exists(),'Run code/evaluate.py first')
    require(json.loads((data/'validation.json').read_text())['status']=='PASS','Evaluation must pass')
    for metric in METRICS:
        fig=benchmark(data,metric);export(fig,output/f'benchmark_{metric}');plt.close(fig)
        fig=benchmark(data,metric,ensemble=True);export(fig,output/f'benchmark_ensemble_{metric}');plt.close(fig)
    for metric in ['PCC','RSE']:
        for label,both,ensemble in [('main',False,False),('conditions',True,False),('ensemble',False,True)]:
            fig=cost(data,metric,both,ensemble);export(fig,output/f'pareto_{label}_{metric}');plt.close(fig)
    (output/'plotting_metadata.json').write_text(json.dumps(dict(font=FONT,figure_count=14,pcc_selection_shared_with_rse=True,pareto_x_min=.001,point_size=80),indent=2)+'\n')
    print(f'Wrote 14 figures in PDF, SVG and PNG; font={FONT}')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'reproduced')
    p.add_argument('--output',type=Path,default=ROOT/'reproduced/figures')
    args=p.parse_args();run(args.data,args.output)
