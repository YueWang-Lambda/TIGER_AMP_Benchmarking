"""V17 plotting functions. This complete source is also embedded in the notebook.

Each axis function accepts an Axes, so independent panels expose physical canvas
and axis dimensions without hidden rescaling. No training occurs here.
"""
from pathlib import Path
import os,json
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors,font_manager
from matplotlib.lines import Line2D
from matplotlib.patches import Patch,FancyBboxPatch
from matplotlib.backends.backend_pdf import PdfPages

OUT=Path.cwd() if (Path.cwd()/'reconstruction_acceptance.json').exists() else Path(__file__).resolve().parent
T=pd.read_csv(OUT/'tiger_per_seed.csv');TS=pd.read_csv(OUT/'tiger_summary.csv')
B=pd.read_csv(OUT/'benchmark_per_seed.csv');BS=pd.read_csv(OUT/'benchmark_summary.csv')
BF=pd.read_csv(OUT/'benchmark_family_per_seed.csv');LABELS=json.loads((OUT/'labels.json').read_text())
GROUPS=json.loads((OUT/'ablation_groups.json').read_text())
PEP=pd.read_csv(OUT/'external_peptide_source_data.csv');EDGES=pd.read_csv(OUT/'external_edge_source_data.csv')
SIM=pd.read_csv(OUT/'external_training_similarity_v15.csv')
FAMILY_ORDER=sorted(PEP.family.unique())
METRICS=['PCC','KCC','log10MAE','RSE']
ML_METRICS=['log10MAE','RSE','PCC','KCC']
COLORS=['#318BFA','#43A047','#F97316','#7C6BD9','#009E9A','#C15D9E','#202020']
INK='#202020';MUTED='#697581';BLUE=COLORS[0];GREEN=COLORS[1]
FAMILY_COLORS=dict(zip(FAMILY_ORDER,COLORS+['#D55E00','#D0A300','#56B4E9','#8C564B']))
CATEGORY_COLORS={'plm_finetune':'#7C6BD9','frozen_plm':'#43A047','generative_lm':'#F97316','dl_sequence':'#318BFA','dl_physicochemical':'#009E9A','dl_sequence_physicochemical':'#C15D9E','multimodal_regression':'#202020','multimodal_metric_learning':'#202020'}
CATEGORIES={'BERT-AmPEP60':'plm_finetune','LLAMP':'plm_finetune','HyperAMP':'frozen_plm','SenseXAMP':'frozen_plm','Deep-PIE':'frozen_plm','AMPGen':'frozen_plm','AMPpredictor':'frozen_plm','tsAMP':'frozen_plm','AMP-Designer':'generative_lm','AMPlify':'dl_sequence','Das':'dl_sequence','APEX':'dl_sequence','DeepAMP':'dl_sequence','HydrAMP':'dl_sequence','EvoGradient':'dl_sequence','ANIA':'dl_sequence','MBC-Attention':'dl_physicochemical','EIPpred':'dl_physicochemical','AMPActiPred':'dl_physicochemical','AMPLiT':'dl_sequence_physicochemical','TIGER-single':'multimodal_regression','GGN-ABPMIC':'multimodal_regression','TIGER-pair':'multimodal_metric_learning'}
assert set(B.method)==set(CATEGORIES)
DISPLAY={'Das':'Das2021','BERT-AmPEP60':'BERT-AmPEP60','SenseXAMP':'SenseXAMP-Ecoli','tsAMP':'tsAMP-CS'}
def display_name(m):return DISPLAY.get(m,m)
def colour(m):return CATEGORY_COLORS[CATEGORIES[m]]
def marker(m):return 'D' if m=='TIGER-pair' else ('s' if m=='TIGER-single' else 'o')
for p in Path('/usr/share/fonts/truetype/msttcorefonts').glob('Arial*.ttf'):font_manager.fontManager.addfont(str(p))
plt.rcParams.update({'font.family':'Arial','font.size':11,'axes.labelsize':12,'xtick.labelsize':10,'ytick.labelsize':11,'axes.titlesize':13,'pdf.fonttype':42,'svg.fonttype':'none','axes.grid':False,'savefig.facecolor':'white'})
DPI=int(os.environ.get('V15_EXPORT_DPI','600'));SPECS={};LAYOUT={};MAIN_FILES=[]

def clean(ax):
    ax.spines[:].set_visible(False);ax.tick_params(length=3,color='#8C949D',labelcolor='#626B75');ax.grid(False)
def vals(arm,metric):return T[T.arm==arm].sort_values('seed')[metric].to_numpy()
def mean_ci(x):
    x=np.asarray(x,float)
    if not np.isfinite(x).all():return np.nan,np.nan
    return x.mean(),stats.t.ppf(.975,len(x)-1)*stats.sem(x)
def layout_check(fig):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();outside=[]
    from matplotlib.text import Text
    ignored=set()
    for ax in fig.axes:
        for axis,lim in [(ax.xaxis,ax.get_xlim()),(ax.yaxis,ax.get_ylim())]:
            for tick in axis.get_major_ticks()+axis.get_minor_ticks():
                if not min(lim)<=tick.get_loc()<=max(lim):ignored.update([id(tick.label1),id(tick.label2)])
    for t in fig.findobj(Text):
        if id(t) in ignored:continue
        if not t.get_visible() or not t.get_text().strip():continue
        if t.axes is not None and (not t.axes.axison):continue
        # Axis locators may keep invisible off-limit tick objects; rendered labels only.
        box=t.get_window_extent(renderer)
        if box.width==0 or box.height==0:continue
        if box.x1<0 or box.x0>w or box.y1<0 or box.y0>h:continue
        if box.x0<-2 or box.y0<-2 or box.x1>w+2 or box.y1>h+2:outside.append(t.get_text())
    return {'off_canvas_text':outside}
def save(fig,name,width=None,height=None,description='',main=True):
    directory=OUT if main else OUT/'panels';directory.mkdir(exist_ok=True)
    check=layout_check(fig)
    if check['off_canvas_text']:raise RuntimeError((name,check))
    for ext in ['pdf','svg','png','tiff']:
        kw={'pil_kwargs':{'compression':'tiff_lzw'}} if ext=='tiff' else {}
        fig.savefig(directory/f'{name}.{ext}',dpi=DPI,**kw)
    fig.savefig(directory/f'{name}_preview.png',dpi=140)
    SPECS[name]={'width_mm':fig.get_figwidth()*25.4,'height_mm':fig.get_figheight()*25.4,'dpi':DPI,'description':description,'main':main}
    LAYOUT[name]=check
    if main:MAIN_FILES.append(str(directory/f'{name}.pdf'))
    (OUT/'figure_specifications.json').write_text(json.dumps(SPECS,indent=2))
    (OUT/'figure_layout_checks.json').write_text(json.dumps(LAYOUT,indent=2))
    return directory/f'{name}_preview.png'

def training_axis(ax,panel):
    clean(ax)
    if panel=='a':
        names=['Source observations','Eligible observations','Sequence × CFU records','Unique sequences','External-method subset']
        ns=[11516,7401,7217,6847,4929]
        ax.barh(range(5),ns,color=[BLUE]*3+[INK,GREEN],height=.60)
        ax.set(yticks=range(5),yticklabels=names,xlim=(0,13700),xlabel='Number of observations / sequences');ax.invert_yaxis()
        for y,n in enumerate(ns):ax.text(n+160,y,f'{n:,}',va='center',fontsize=11)
    elif panel=='b':
        x=pd.read_csv(OUT/'training_cleaned_replay.csv');y=np.log2(x.groupby('sequence').MIC.min())
        ax.hist(y,bins=np.arange(-10,17,.5),color=BLUE,alpha=.85);ax.axvline(7,color=INK,ls='--',lw=1)
        ax.set(xlabel='Sequence-level log2 MIC (µg/ml)',ylabel='Training sequences',xlim=(-10,16))
    else:
        d=pd.read_csv(OUT/'training_delta_distributions.csv');x=d.bin_center
        curves=[('theoretical_full_pool','Full-pool reference','#8C949D')]
        if np.array_equal(d.count_before_balancing_candidates,d.count_after_balancing_final):
            curves.append(('count_after_balancing_final','Candidates = balanced pairs',INK))
        else:
            curves += [('count_before_balancing_candidates','Candidate pairs',BLUE),('count_after_balancing_final','Balanced pairs',INK)]
        for key,label,col in curves:
            y=d[key].to_numpy();ax.plot(x,y/y.sum(),label=label,color=col,lw=1.7)
        ax.set(xlabel='Δlog2 MIC (anchor − query)',ylabel='Fraction per 0.5-log2 bin');ax.legend(frameon=False,fontsize=10)
def training_figure():
    fig=plt.figure(figsize=(250/25.4,170/25.4));fig.suptitle('Training data and local activity differences',x=.03,ha='left',fontsize=16)
    for p,r in [('a',[.32,.58,.62,.29]),('b',[.09,.14,.37,.29]),('c',[.59,.14,.37,.29])]:
        ax=fig.add_axes(r);training_axis(ax,p);ax.set_title(p,loc='left',fontweight='bold')
    fig.text(.03,.035,'P11 uses classification negatives and joint regression-negative pairs; panel c describes AMP–AMP pairs.',fontsize=10,color=MUTED)
    save(fig,'Fig1_training_data')
    return fig

def external_axis(ax,panel):
    clean(ax)
    if panel=='a':
        ns=PEP.groupby('family').size().reindex(FAMILY_ORDER);pc=EDGES.groupby('family').size().reindex(FAMILY_ORDER)
        ax.barh(range(11),ns,color=[FAMILY_COLORS[f] for f in FAMILY_ORDER],height=.62)
        ax.set(yticks=range(11),yticklabels=FAMILY_ORDER,xlim=(0,100),xlabel='Peptides');ax.invert_yaxis()
        for i,f in enumerate(FAMILY_ORDER):ax.text(ns[f]+1,i,f'{ns[f]}  ({pc[f]:,} pairs)',fontsize=10,va='center')
    else:
        for i,f in enumerate(FAMILY_ORDER):
            x=np.sort(SIM.loc[SIM.family==f,'symmetric_max'] if panel=='b' else EDGES.loc[EDGES.family==f,'similarity_03'])
            ax.step(x,np.arange(1,len(x)+1)/len(x),where='post',color=FAMILY_COLORS[f],lw=1.6,ls='-' if i<6 else '--',label=f)
        ax.set(xlabel='Symmetric maximum NW score' if panel=='b' else 'Within-family directed NW score',ylabel='Cumulative fraction',ylim=(0,1.02),xlim=(0,1))
def external_figure():
    fig=plt.figure(figsize=(260/25.4,245/25.4));fig.suptitle('External test set: 11 peptide families',x=.03,ha='left',fontsize=16)
    for p,r in [('a',[.24,.56,.63,.34]),('b',[.09,.20,.36,.23]),('c',[.60,.20,.36,.23])]:
        ax=fig.add_axes(r);external_axis(ax,p);ax.set_title(p,loc='left',fontweight='bold')
    fig.legend([Line2D([],[],color=FAMILY_COLORS[f],ls='-' if i<6 else '--',lw=2) for i,f in enumerate(FAMILY_ORDER)],FAMILY_ORDER,loc='lower center',bbox_to_anchor=(.5,.055),ncol=4,frameon=False,fontsize=10)
    fig.text(.03,.025,'208 peptides; 2,211 directed pairs. Training proximity recomputed against 6,847 frozen training sequences.',fontsize=10,color=MUTED)
    save(fig,'Fig2_external_testset');return fig

def ablation_axis(ax,arms,metric='PCC',labels=None):
    clean(ax)
    for i,a in enumerate(arms):
        v=vals(a,metric);m,h=mean_ci(v);c=INK if a=='factor_P11' else COLORS[i%6]
        ax.scatter(v,i+np.linspace(-.13,.13,len(v)),s=20,c=c,alpha=.5,lw=0)
        ax.errorbar(m,i,xerr=h,fmt='D' if a=='factor_P11' else 'o',color=c,ms=6,lw=1.4,capsize=3)
    ax.set(yticks=range(len(arms)),yticklabels=labels or [LABELS[a] for a in arms],xlabel=metric,ylim=(len(arms)-.55,-.55))
def core_figure():
    fig=plt.figure(figsize=(285/25.4,230/25.4));fig.suptitle('Ablation evidence relative to TIGER-pair (unsigned quota 20,000)',x=.03,ha='left',fontsize=16)
    panels=[('a',['factor_P11','s4_mod_ps','s4_mod_pg','s4_mod_sg','s4_mod_p','s4_mod_s','s4_mod_g']),('b',['factor_P11','S4_gat','S4_gcn','S4_sage']),('c',['factor_P11','S4_se','S4_sp','S4_sep','S4_nodeemb','S4_nodezero']),('d',['s4_gate_off','S4_sim00','S4_sim03','S4_sim05','factor_P11','S4_sim09'])]
    for (p,arms),r in zip(panels,[[.22,.58,.25,.31],[.70,.58,.26,.31],[.22,.13,.25,.31],[.70,.13,.26,.31]]):
        ax=fig.add_axes(r);ablation_axis(ax,arms);ax.set_title(p,loc='left',fontweight='bold')
    fig.text(.03,.025,'Dots: all ten seeds. Intervals: 95% confidence intervals across seeds. Validation gate fixed at 0.7 in panel d.',fontsize=10,color=MUTED)
    save(fig,'Fig3_core_ablation');return fig

def manuscript_axis(ax,metric,kind='C',font_scale=1.):
    arms=['factor_P11','s4_gate_off'] if kind=='C' else ['factor_P11','s4_single_n1n2']
    palette=['#4DBDED','#A1D799'] if kind=='C' else ['#E3C2FB','#BFBFBF']
    upper={'log10MAE':.56,'RSE':1.05,'PCC':.8,'KCC':.65}[metric]
    spreads=[(vals(a,metric).mean()-vals(a,metric).std(ddof=1), vals(a,metric).mean()+vals(a,metric).std(ddof=1)) for a in arms]
    lower=min(.2,np.floor((min(x[0] for x in spreads)-.025)*20)/20)
    upper=max(upper,np.ceil((max(x[1] for x in spreads)+.06)*20)/20)
    for i,(a,c) in enumerate(zip(arms,palette)):
        v=vals(a,metric);mean=v.mean();sd=v.std(ddof=1)
        assert mean-sd>=lower and mean+sd<upper
        ax.bar(i,mean,width=.57,color=c,ec='#666666',lw=.7)
        ax.errorbar(i,mean,yerr=sd,fmt='none',color='#777777',lw=1,capsize=3)
        ax.text(i,mean+sd+.04*(upper-.2),f'{mean:.3f}',ha='center',fontsize=11*font_scale)
    ax.spines[['right','top']].set_visible(False);ax.spines[['left','bottom']].set_color('#777777')
    ax.set(ylim=(lower,upper),xlim=(-.6,1.6),xticks=[],ylabel=metric);ax.tick_params(labelsize=11*font_scale)
def manuscript_figure(kind='C'):
    fig=plt.figure(figsize=(175/25.4,165/25.4));title='Similarity-gate comparison' if kind=='C' else 'Pair supervision versus single-peptide regression'
    fig.text(.045,.965,f'{kind}  |  {title}',va='top',fontsize=13,fontweight='bold')
    pal=['#4DBDED','#A1D799'] if kind=='C' else ['#E3C2FB','#BFBFBF']
    names=['TIGER-pair (τ = 0.7)','No training gate'] if kind=='C' else ['TIGER-pair (P11)','Single-peptide head']
    fig.legend([Patch(fc=c,ec='#666666') for c in pal],names,loc='upper center',bbox_to_anchor=(.52,.91),ncol=2,frameon=False,fontsize=10)
    for m,r in zip(ML_METRICS,[[.14,.54,.30,.25],[.64,.54,.30,.25],[.14,.17,.30,.25],[.64,.17,.30,.25]]):manuscript_axis(fig.add_axes(r),m,kind)
    fig.text(.05,.065,'Ten matched seeds; mean ± sample SD; all 11 families.',fontsize=10,color=MUTED)
    fig.text(.05,.030,'Only the training gate differs; validation remains at 0.7.' if kind=='C' else 'Both heads retain classification and regression negatives; the existing single head is reused.',fontsize=9,color=MUTED)
    save(fig,'Fig3C_similarity_gate' if kind=='C' else 'Fig3E_metric_learning');return fig

def benchmark_order(metric):
    d=BS[(BS.metric==metric)&(BS.condition=='with_negatives')]
    return d.sort_values('mean',ascending=metric in ['RSE','log10MAE'],na_position='last').method.tolist()
def benchmark_axis(ax,metric,heatmap=False,font_scale=1.):
    order=benchmark_order(metric);clean(ax)
    if not heatmap:
        for i,m in enumerate(order):
            x=B[(B.method==m)&(B.condition=='with_negatives')][metric].to_numpy();mu,h=mean_ci(x)
            if m.startswith('TIGER'):ax.axhspan(i-.44,i+.44,color='#E4F9DF',zorder=0)
            ax.scatter(x,i+np.linspace(-.15,.15,10),color=colour(m),s=25,alpha=.50,marker=marker(m),lw=0)
            if np.isfinite(mu):ax.errorbar(mu,i,xerr=h,color=colour(m),fmt=marker(m),ms=6,capsize=3,lw=1.3)
            else:ax.text(.99,i,'NA*',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=9,color=colour(m))
        ax.set(xlabel=metric,yticks=range(23),yticklabels=[display_name(m) for m in order],ylim=(22.7,-.7))
        if metric=='RSE':ax.set_xscale('log');ax.set_xlabel('RSE (log scale; lower is better)')
        for tick,m in zip(ax.get_yticklabels(),order):tick.set_color(colour(m))
    else:
        # Means propagate any undefined seed; no changing of the seed denominator.
        mat=BF[BF.condition=='with_negatives'].groupby(['method','family'])[metric].agg(lambda s:np.mean(s.to_numpy())).unstack().reindex(index=order,columns=FAMILY_ORDER)
        if metric in ['PCC','KCC']:norm=colors.Normalize(-.3,1);cmap=colors.LinearSegmentedColormap.from_list('corr',['#F97316','#FAFAFA','#318BFA'])
        else:
            norm=colors.LogNorm(max(.01,np.nanmin(mat.to_numpy())),np.nanmax(mat.to_numpy()));cmap=plt.get_cmap('Blues')
        im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,norm=norm,aspect='auto')
        for i in range(23):
            for j in range(11):
                v=mat.iloc[i,j];rgba=cmap(norm(v)) if np.isfinite(v) else (1,1,1,1)
                tc='white' if np.mean(rgba[:3])<.55 else INK
                ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=8.7*font_scale,color=tc)
        ax.set(yticks=range(23),yticklabels=[display_name(m) for m in order],xticks=range(11),xticklabels=FAMILY_ORDER)
        ax.tick_params(length=0);plt.setp(ax.get_xticklabels(),rotation=45,ha='right',fontsize=9*font_scale)
    return order
def benchmark_figure(metric='PCC'):
    fig=plt.figure(figsize=(430/25.4,240/25.4));fig.suptitle(f'Twenty-three-method benchmark | {metric}',x=.03,ha='left',fontsize=17)
    benchmark_axis(fig.add_axes([.13,.16,.29,.72]),metric)
    benchmark_axis(fig.add_axes([.57,.16,.41,.72]),metric,True)
    fig.text(.03,.065,'With-negative condition; TIGER uses P11 / S11. Ten seeds, each evaluated on 11 families and 2,211 directed pairs.',fontsize=11,color=MUTED)
    fig.text(.03,.035,'95% seed CIs; faint points retain available seeds. * AMPLiT: one undefined macro (constant predictions in one family).',fontsize=10,color=MUTED)
    save(fig,'Fig4_benchmark'+('' if metric=='PCC' else '_'+metric));return fig

def controls_axis(ax,metric='PCC',condition_offset=.16):
    # V14 layout: filled above, hollow below each method's row centre.
    assert 0 < condition_offset < .4
    order=benchmark_order(metric);clean(ax)
    for i,m in enumerate(order):
        by=BS[(BS.method==m)&(BS.metric==metric)].set_index('condition');c=colour(m)
        a,b=[by.loc[k,'mean'] for k in ['without_negatives','with_negatives']]
        if np.isfinite([a,b]).all():ax.plot([a,b],[i+condition_offset,i-condition_offset],color=c,lw=.7,alpha=.45,zorder=1)
        for cond,fill,dy in [('without_negatives','white',condition_offset),('with_negatives',c,-condition_offset)]:
            r=by.loc[cond]
            if np.isfinite(r['mean']):ax.errorbar(r['mean'],i+dy,xerr=[[r['mean']-r.ci_low],[r.ci_high-r['mean']]],fmt=marker(m),mfc=fill,mec=c,color=c,ms=6,lw=1,capsize=0,zorder=3)
            else:ax.text(.99,i+dy,'NA*',ha='right',va='center',transform=ax.get_yaxis_transform(),fontsize=9,color=c)
    ax.set(yticks=range(23),yticklabels=[display_name(m) for m in order],ylim=(22.7,-.7),xlabel=metric)
    if metric=='RSE':ax.set_xscale('log')
def controls_figure():
    fig=plt.figure(figsize=(340/25.4,245/25.4));fig.suptitle('Negative-sample training controls',x=.03,ha='left',fontsize=17)
    for m,r in [('PCC',[.20,.14,.28,.72]),('RSE',[.69,.14,.28,.72])]:controls_axis(fig.add_axes(r),m)
    fig.legend([Line2D([],[],marker='o',mfc='white',mec=INK,lw=0),Line2D([],[],marker='o',color=INK,lw=0)],['Without negatives','With negatives'],loc='upper center',bbox_to_anchor=(.54,.935),ncol=2,frameon=False)
    fig.text(.03,.05,'TIGER compares registered 00 vs 11 cells. EvoGradient also changes classification pretraining; this is a protocol comparison.',fontsize=10,color=MUTED)
    save(fig,'Fig7_negative_sample_controls');return fig

def factorial_axis(ax,metric='PCC'):
    """Registered two-condition pair-model comparison; no factorial effect claim."""
    clean(ax)
    for i,arm in enumerate(['factor_P00','factor_P11']):
        v=vals(arm,metric);m,h=mean_ci(v)
        ax.scatter(i+np.linspace(-.05,.05,10),v,s=23,color=INK,alpha=.45,lw=0)
        ax.errorbar(i,m,yerr=h,fmt='D',color=INK,mfc='white' if i==0 else INK,ms=7,capsize=3,lw=1.3)
    for seed in range(120,130):
        y=[T[(T.arm==a)&(T.seed==seed)][metric].iloc[0] for a in ['factor_P00','factor_P11']]
        ax.plot([0,1],y,color=INK,alpha=.18,lw=.7,zorder=0)
    ax.set(xticks=[0,1],xticklabels=['P00\nNeither stage','P11\nBoth stages'],ylabel=metric,xlim=(-.4,1.4))
def factorial_figure():
    fig=plt.figure(figsize=(245/25.4,195/25.4));fig.suptitle('Negative exposure under unsigned quota 20,000',x=.03,ha='left',fontsize=15)
    for m,r in zip(METRICS,[[.10,.57,.36,.27],[.61,.57,.36,.27],[.10,.16,.36,.27],[.61,.16,.36,.27]]):factorial_axis(fig.add_axes(r),m)
    fig.text(.03,.055,'TIGER-pair: P00 versus P11; ten matched seeds; mean ± 95% seed CI.',fontsize=10,color=MUTED)
    fig.text(.03,.025,'P00 disables the complete classification-pretraining stage and regression-negative injection.',fontsize=10,color=MUTED)
    save(fig,'Fig8_negative_conditions');return fig

def atlas_axis(ax,metric):
    arms=[a for _,aa in GROUPS for a in aa];ablation_axis(ax,arms,metric)
    ax.set_title(metric,loc='left',fontsize=13)
    if metric=='RSE':ax.set_xscale('log')
def atlas_figure():
    fig=plt.figure(figsize=(440/25.4,340/25.4));fig.suptitle('Complete ablation atlas | Unsigned-20,000 reference and 41 controls',x=.03,ha='left',fontsize=17)
    arms=[a for _,aa in GROUPS for a in aa]
    for j,m in enumerate(METRICS):
        ax=fig.add_axes([.23+j*.19,.10,.16,.80]);atlas_axis(ax,m)
        if j:ax.set_yticklabels([])
    fig.text(.03,.040,'Zero node features belongs to Node attributes. Sequence-chain edges is an independent Edge geometry intervention.',fontsize=11,color=MUTED)
    fig.text(.03,.018,'Raw metrics; no composite renormalization or seed selection. Four threshold arms hold validation at 0.7.',fontsize=11,color=MUTED)
    save(fig,'ExtendedData1_metric_atlas');return fig

def graph_figure():
    fig=plt.figure(figsize=(255/25.4,125/25.4));fig.suptitle('Node information and contact geometry',x=.03,ha='left',fontsize=16)
    arms=['factor_P11','S4_nodezero','S4_edge_chain','s4_mod_ps']
    for m,r in [('PCC',[.28,.20,.25,.56]),('RSE',[.70,.20,.25,.56])]:
        ax=fig.add_axes(r);ablation_axis(ax,arms,m)
        if m=='RSE':ax.set_yticklabels([])
    fig.text(.03,.05,'Zeroed nodes retain contact adjacency. Sequence-chain edges alter adjacency while preserving node attributes.',fontsize=10,color=MUTED)
    save(fig,'Fig6_graph_information');return fig

def independent_panel(draw,name,width=200.,height=165.,left=50.,bottom=28.,axes_width=140.,axes_height=115.,font_scale=1.):
    """All physical lengths are in mm. Axis lengths are independent of canvas."""
    assert left>=0 and bottom>=0 and left+axes_width<=width and bottom+axes_height<=height
    with plt.rc_context({'font.size':11*font_scale,'axes.labelsize':12*font_scale,'ytick.labelsize':11*font_scale,'xtick.labelsize':10*font_scale}):
        fig=plt.figure(figsize=(width/25.4,height/25.4));ax=fig.add_axes([left/width,bottom/height,axes_width/width,axes_height/height]);draw(ax)
        save(fig,name,main=False);return fig
