"""V15 plotting functions. This complete source is also embedded in the notebook.

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

BASE = Path(__file__).resolve().parents[2]
ADDON = BASE / 'TIGER_ensemble_addendum'
OUT = ADDON / 'figures'
DATA = ADDON / 'tables'
B = pd.read_csv(DATA/'benchmark_per_seed.csv')
BS = pd.read_csv(DATA/'benchmark_summary.csv')
BF = pd.read_csv(DATA/'benchmark_family_per_seed.csv')
FAMILY_ORDER = sorted(BF.family.unique())
METRICS = ['PCC','KCC','log10MAE','RSE']
COLORS=['#318BFA','#43A047','#F97316','#7C6BD9','#009E9A','#C15D9E','#202020']
INK='#202020';MUTED='#697581';BLUE=COLORS[0];GREEN=COLORS[1]
FAMILY_COLORS=dict(zip(FAMILY_ORDER,COLORS+['#D55E00','#D0A300','#56B4E9','#8C564B']))
CATEGORY_COLORS={'plm_finetune':'#7C6BD9','frozen_plm':'#43A047','generative_lm':'#F97316','dl_sequence':'#318BFA','dl_physicochemical':'#009E9A','dl_sequence_physicochemical':'#C15D9E','multimodal_regression':'#202020','multimodal_metric_learning':'#202020'}
CATEGORIES={'BERT-AmPEP60':'plm_finetune','LLAMP':'plm_finetune','HyperAMP':'frozen_plm','SenseXAMP':'frozen_plm','Deep-PIE':'frozen_plm','AMPGen':'frozen_plm','AMPpredictor':'frozen_plm','tsAMP':'frozen_plm','AMP-Designer':'generative_lm','AMPlify':'dl_sequence','Das':'dl_sequence','APEX':'dl_sequence','DeepAMP':'dl_sequence','HydrAMP':'dl_sequence','EvoGradient':'dl_sequence','ANIA':'dl_sequence','MBC-Attention':'dl_physicochemical','EIPpred':'dl_physicochemical','AMPActiPred':'dl_physicochemical','AMPLiT':'dl_sequence_physicochemical','TIGER-single':'multimodal_regression','GGN-ABPMIC':'multimodal_regression','TIGER-pair':'multimodal_metric_learning'}
CATEGORIES['TIGER-ensemble']='ensemble'
CATEGORY_COLORS['ensemble']='#202020'
assert set(B.method)==set(CATEGORIES)
DISPLAY={'Das':'Das2021','BERT-AmPEP60':'BERT-AmPEP60','SenseXAMP':'SenseXAMP-Ecoli','tsAMP':'tsAMP-CS'}
def display_name(m):return DISPLAY.get(m,m)
def colour(m):return CATEGORY_COLORS[CATEGORIES[m]]
def marker(m):return '*' if m=='TIGER-ensemble' else 'D' if m=='TIGER-pair' else ('s' if m=='TIGER-single' else 'o')
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
        ax.set(xlabel=metric,yticks=range(len(order)),yticklabels=[display_name(m) for m in order],ylim=(len(order)-.3,-.7))
        if metric=='RSE':ax.set_xscale('log');ax.set_xlabel('RSE (log scale; lower is better)')
        for tick,m in zip(ax.get_yticklabels(),order):tick.set_color(colour(m))
    else:
        # Means propagate any undefined seed; no changing of the seed denominator.
        mat=BF[BF.condition=='with_negatives'].groupby(['method','family'])[metric].agg(lambda s:np.mean(s.to_numpy())).unstack().reindex(index=order,columns=FAMILY_ORDER)
        if metric in ['PCC','KCC']:norm=colors.Normalize(-.3,1);cmap=colors.LinearSegmentedColormap.from_list('corr',['#F97316','#FAFAFA','#318BFA'])
        else:
            norm=colors.LogNorm(max(.01,np.nanmin(mat.to_numpy())),np.nanmax(mat.to_numpy()));cmap=plt.get_cmap('Blues')
        im=ax.imshow(np.ma.masked_invalid(mat),cmap=cmap,norm=norm,aspect='auto')
        for i in range(len(order)):
            for j in range(11):
                v=mat.iloc[i,j];rgba=cmap(norm(v)) if np.isfinite(v) else (1,1,1,1)
                tc='white' if np.mean(rgba[:3])<.55 else INK
                ax.text(j,i,f'{v:.2f}' if np.isfinite(v) else 'NA',ha='center',va='center',fontsize=8.7*font_scale,color=tc)
        ax.set(yticks=range(len(order)),yticklabels=[display_name(m) for m in order],xticks=range(11),xticklabels=FAMILY_ORDER)
        ax.tick_params(length=0);plt.setp(ax.get_xticklabels(),rotation=45,ha='right',fontsize=9*font_scale)
    return order
def benchmark_figure(metric='PCC'):
    fig=plt.figure(figsize=(430/25.4,240/25.4));fig.suptitle(f'Benchmark including TIGER-ensemble | {metric}',x=.03,ha='left',fontsize=17)
    benchmark_axis(fig.add_axes([.13,.16,.29,.72]),metric)
    benchmark_axis(fig.add_axes([.57,.16,.41,.72]),metric,True)
    fig.text(.03,.065,'23 backbones + one equal-weight P11/S11 ensemble; with-negative condition. Ten seeds; 11 families; 2,211 pairs.',fontsize=11,color=MUTED)
    fig.text(.03,.035,'95% seed CIs; faint points retain available seeds. * AMPLiT: one undefined macro (constant predictions in one family).',fontsize=10,color=MUTED)
    save(fig,'Fig4_benchmark_TIGER_ensemble'+('' if metric=='PCC' else '_'+metric));return fig


def independent_panel(draw,name,width=200.,height=165.,left=50.,bottom=28.,axes_width=140.,axes_height=115.,font_scale=1.):
    """All physical lengths are in mm. Axis lengths are independent of canvas."""
    assert left>=0 and bottom>=0 and left+axes_width<=width and bottom+axes_height<=height
    with plt.rc_context({'font.size':11*font_scale,'axes.labelsize':12*font_scale,'ytick.labelsize':11*font_scale,'xtick.labelsize':10*font_scale}):
        fig=plt.figure(figsize=(width/25.4,height/25.4));ax=fig.add_axes([left/width,bottom/height,axes_width/width,axes_height/height]);draw(ax)
        save(fig,name,main=False);return fig


# TIGER-single and ensemble use V16 measured times; 22 other backbones retain V15 coordinates.
assert json.loads((ADDON/'timing/result.json').read_text())['status']=='PASS'
control_best={m:pd.read_csv(DATA/f'cost_best_{m}.csv') for m in ['PCC','RSE']}
control_fronts={m:{tag:pd.read_csv(DATA/f'pareto_{m}_{tag}.csv') for tag in ['new_structures','structures_ready']} for m in ['PCC','RSE']}
METHOD_CATEGORY=CATEGORIES
CATEGORY_ORDER=['plm_finetune','frozen_plm','generative_lm','dl_sequence','dl_physicochemical','dl_sequence_physicochemical','multimodal_regression','multimodal_metric_learning']
CATEGORY_LABELS={'plm_finetune':'PLM fine-tuning','frozen_plm':'Frozen PLM + head','generative_lm':'Generative LM scorer','dl_sequence':'Sequence DL (no PLM)','dl_physicochemical':'Physicochemical models (no PLM)','dl_sequence_physicochemical':'Sequence + physicochemical DL (no PLM)','multimodal_regression':'3D multimodal regression','multimodal_metric_learning':'Multimodal metric learning'}
FIG5_CATEGORY_COLORS=CATEGORY_COLORS
FIG5_CATEGORY_TEXT_COLORS={'ensemble':'#202020','plm_finetune':'#5B478C','frozen_plm':'#287538','generative_lm':'#A54F00','dl_sequence':'#173D63','dl_physicochemical':'#006F6C','dl_sequence_physicochemical':'#88406F','multimodal_regression':'#202020','multimodal_metric_learning':'#202020'}
FIG5_THEME=dict(ink='#202020',muted='#8C949D',faded_text='#AEB4BA',frontier='#454545',green='#E4F9DF',gray='#FAFAFA')
method_marker=marker
noneg_display_name=display_name

def resolve_cost_annotation_overlaps(fig, ax, font_scale=1.):
    """Deterministic label placement in screen space after time coordinates change.

    Keeps the user's colours, fonts, marker sizes and axes. Only annotation offsets
    change. Candidate placements avoid other labels and all plotted marker centres;
    distant labels receive thin leader lines. Works on log or linear ordinate axes.
    """
    from matplotlib.text import Annotation
    from matplotlib.transforms import Bbox
    from matplotlib.collections import PathCollection
    original_dpi=fig.dpi
    fig.set_dpi(200)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    from matplotlib.backends.backend_agg import RendererAgg
    w,h=fig.canvas.get_width_height()
    data_renderer=RendererAgg(w,h,fig.dpi);data_renderer.clear()
    for artist in list(ax.lines)+list(ax.collections):
        if artist.get_visible():artist.draw(data_renderer)
    data_mask=np.asarray(data_renderer.buffer_rgba())[:,:,3]>24
    labels=[t for t in ax.texts if isinstance(t,Annotation) and t.get_visible()]
    axis_box=ax.get_window_extent(renderer)
    # Marker centres, including both structure-cost scenarios.
    points=[]
    for col in ax.collections:
        if isinstance(col,PathCollection):
            points.extend(col.get_offset_transform().transform(col.get_offsets()))
    fixed=[t.get_window_extent(renderer).expanded(1.01,1.04) for t in ax.texts if not isinstance(t,Annotation) and t.get_visible()]
    placed=[]
    # Preserve the existing high-value/TIGER labels first.
    labels.sort(key=lambda a:(not a.get_text().startswith('TIGER'),a.xy[1],a.xy[0]))
    overlaps=[]
    for label in labels:
        initial=tuple(label.get_position());initial_ha=label.get_ha()
        candidates=[(initial[0],initial[1],initial_ha)]
        for radius in [9,14,20,27,35,45,58,74]:
            candidates.extend([(radius,0,'left'),(-radius,0,'right'),(0,radius,'center'),(0,-radius,'center'),
                               (radius,radius,'left'),(-radius,radius,'right'),(radius,-radius,'left'),(-radius,-radius,'right')])
        best=None;best_score=float('inf')
        for dx,dy,ha in candidates:
            label.set_position((dx,dy));label.set_ha(ha)
            box=label.get_window_extent(renderer).expanded(1.04,1.20)
            intersections=sum(box.overlaps(b) for b in placed+fixed)
            x0=max(0,int(box.x0)-3);x1=min(w,int(box.x1)+4)
            y0=max(0,h-int(box.y1)-4);y1=min(h,h-int(box.y0)+3)
            marker_hits=int(data_mask[y0:y1,x0:x1].any())
            outside=max(0,axis_box.x0-box.x0)+max(0,box.x1-axis_box.x1)+max(0,axis_box.y0-box.y0)+max(0,box.y1-axis_box.y1)
            shift=((dx-initial[0])**2+(dy-initial[1])**2)**.5
            score=100000*(intersections+marker_hits)+1000*outside+shift
            if score<best_score:best_score=score;best=(dx,dy,ha,box,intersections,marker_hits)
            if score==0:break
        dx,dy,ha,box,intersections,marker_hits=best
        label.set_position((dx,dy));label.set_ha(ha);placed.append(box)
        if intersections or marker_hits:overlaps.append(label.get_text())
        if abs(dx)+abs(dy)>34:
            # Leader ends at the label edge; it never covers the label itself.
            xy=ax.transData.transform(label.xy);target=(box.x0 if ha=='left' else box.x1 if ha=='right' else (box.x0+box.x1)/2,(box.y0+box.y1)/2)
            ax.annotate('',xy=label.xy,xytext=ax.transData.inverted().transform(target),
                        arrowprops=dict(arrowstyle='-',color=label.get_color(),lw=.45,alpha=.45),zorder=2)
    assert not overlaps,('Unresolved cost label collisions',overlaps)
    if not hasattr(fig,'_v12_label_audit'):fig._v12_label_audit={}
    fig._v12_label_audit=dict(labels=len(labels),label_or_marker_collisions=0,placement='deterministic screen-space search')

    fig.set_dpi(original_dpi)
    fig.canvas.draw()

"""v15 PCC Pareto plot; full drawing code embedded in the notebook."""

def ensemble_cost_figure(metric='PCC', canvas_width_mm=325.12, canvas_height_mm=190.5,
                         axes_left_mm=26.01, axes_bottom_mm=34.0,
                         axes_width_mm=292.60, axes_height_mm=100.0, font_scale=1.0):

    # ===== EDITABLE GEOMETRY AND TYPOGRAPHY / 尺寸与字体接口 =====
    CANVAS_WIDTH_MM = canvas_width_mm
    CANVAS_HEIGHT_MM = canvas_height_mm
    AXES_LEFT_MM = axes_left_mm
    AXES_BOTTOM_MM = axes_bottom_mm
    AXES_WIDTH_MM = axes_width_mm
    AXES_HEIGHT_MM = axes_height_mm
    FONT_SCALE = font_scale
    FONT_FAMILY = "Arial"
    TITLE_SIZE = 18.0
    LABEL_SIZE = 10.0
    TICK_SIZE = 9.0
    LEGEND_SIZE = 10.0
    NOTE_SIZE = 9.0
    POINT_SIZE = 48.0
    EXPORT_DPI = 600
    X_LIM = (0.001, 1500.0)
    METRIC = metric
    SHOW_BOTH_CONDITIONS = False
    Y_LIM = ((0.18, 0.72) if METRIC == "PCC" else
             ((0.58, 4.2) if SHOW_BOTH_CONDITIONS else (0.58, 2.10)))
    X_CUT, Y_CUT = 1.0, (0.50 if METRIC == "PCC" else 1.0)
    fs=lambda x: x*FONT_SCALE
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter
    AA=FIG5_THEME.copy()
    data=control_cost[METRIC] if SHOW_BOTH_CONDITIONS else control_best[METRIC]
    front=control_fronts[METRIC]['new_structures']
    assert len(data)==(46 if SHOW_BOTH_CONDITIONS else 24)
    assert set(data.method)==set(METHOD_CATEGORY)
    assert data.time_s_with_struct.min()>=X_LIM[0]

    with plt.rc_context({'font.family':FONT_FAMILY,'font.size':fs(LABEL_SIZE),
                         'text.color':AA['ink'],'axes.labelcolor':AA['ink'],
                         'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(CANVAS_WIDTH_MM/25.4,CANVAS_HEIGHT_MM/25.4),facecolor='white')
        noun='Prediction accuracy' if METRIC=='PCC' else 'Prediction error'
        fig.text(.015,.970,noun+' vs. inference cost, including TIGER-ensemble',fontsize=fs(TITLE_SIZE),va='top')
        mode='Both training conditions' if SHOW_BOTH_CONDITIONS else ('PCC-selected condition per backbone; fixed P11/S11 ensemble' if METRIC=='RSE' else 'Best condition per backbone; fixed P11/S11 ensemble')
        fig.text(.015,.916,mode+' · ten-seed mean '+METRIC+' · 208 peptides / 2,211 pairs',
                 fontsize=fs(LABEL_SIZE),color=AA['muted'],va='top')
        fig.text(.015,.886,'Seed-123 medians, 20 repeats: pair and ensemble retimed; other backbones retained',
                 fontsize=fs(LABEL_SIZE),color=AA['muted'],va='top')
        ax=fig.add_axes([AXES_LEFT_MM/CANVAS_WIDTH_MM,AXES_BOTTOM_MM/CANVAS_HEIGHT_MM,
                         AXES_WIDTH_MM/CANVAS_WIDTH_MM,AXES_HEIGHT_MM/CANVAS_HEIGHT_MM])
        ax.set_xscale('log')
        if METRIC=='RSE':ax.set_yscale('log')
        ax.set(xlim=X_LIM,ylim=Y_LIM,xlabel='Time for external workload (s, log scale)',
               ylabel=('External family-macro Pearson r' if METRIC=='PCC' else 'External family-macro RSE (log scale; lower is better)'))
        ax.spines[:].set_visible(False);ax.grid(False)
        ax.set_xticks([.001,.01,.1,1,10,100,1000])
        ax.set_xticklabels(['0.001','0.01','0.1','1','10','100','1,000'])
        ax.xaxis.set_minor_locator(LogLocator(base=10,subs=[2,5],numticks=30))
        ax.xaxis.set_minor_formatter(NullFormatter())
        if METRIC=='RSE':
            ticks=[.6,.8,1,1.5,2,3,4] if SHOW_BOTH_CONDITIONS else [.6,.8,1,1.5,2]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.yaxis.set_major_formatter(FixedFormatter([f'{v:.1f}' for v in ticks]))
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.tick_params(axis='both',which='major',labelsize=fs(TICK_SIZE),length=3.5,width=.7,color='#777777',pad=5)
        ax.tick_params(axis='x',which='minor',length=2,width=.6,color='#888888')
        ax.xaxis.label.set_fontsize(fs(LABEL_SIZE));ax.yaxis.label.set_fontsize(fs(LABEL_SIZE))
        ax.xaxis.labelpad=10;ax.yaxis.labelpad=12
        low,high=(Y_CUT,Y_LIM[1]) if METRIC=='PCC' else (Y_LIM[0],Y_CUT)
        ax.add_patch(Rectangle((X_LIM[0],low),X_CUT-X_LIM[0],high-low,fc=AA['green'],ec='none',zorder=0))
        low,high=(Y_LIM[0],Y_CUT) if METRIC=='PCC' else (Y_CUT,Y_LIM[1])
        ax.add_patch(Rectangle((X_CUT,low),X_LIM[1]-X_CUT,high-low,fc=AA['gray'],ec='none',zorder=0))
        ax.plot(front.time_s_with_struct,front['mean'],color=AA['frontier'],lw=1.4,
                linestyle=(0,(.1,2.7)),dash_capstyle='round',zorder=2)
        # Only one method label per backbone; condition is encoded by marker fill.
        for method,g in data.groupby('method',sort=False):
            category=METHOD_CATEGORY[method];colour=FIG5_CATEGORY_COLORS[category]
            textcolour=FIG5_CATEGORY_TEXT_COLORS[category]
            marker=method_marker(method);size=POINT_SIZE*(2.2 if method=='TIGER-ensemble' else 53/43 if method.startswith('TIGER') else 1.)
            label=noneg_display_name(method)
            if method=='AMPLiT' and METRIC=='PCC':label+='†'
            if method in ['SenseXAMP','AMPpredictor']:label+='*'
            if SHOW_BOTH_CONDITIONS:
                x=float(g.time_s_with_struct.iloc[0])
                assert g.time_s_with_struct.nunique()==1
                assert set(g.condition)=={'with_negatives','without_negatives'}
                by=g.set_index('condition');a=float(by.loc['with_negatives','mean']);b=float(by.loc['without_negatives','mean'])
                ax.plot([x,x],[a,b],color=colour,lw=1.1,alpha=.65,zorder=3)
                ax.scatter(x,a,s=size,marker=marker,fc=colour,ec='white',lw=.5,zorder=5)
                ax.scatter(x,b,s=size,marker=marker,fc='white',ec=colour,lw=1.4,zorder=5)
                y=(float(np.nanmean([a,b])) if METRIC=='PCC' else (a*b)**.5)
            else:
                r=g.iloc[0];x=float(r.time_s_with_struct);y=float(r['mean'])
                if method in ['TIGER-pair','TIGER-single','GGN-ABPMIC','TIGER-ensemble']:
                    ax.plot([r.time_s_no_struct,x],[y,y],color=colour,lw=.9,ls=(0,(5,4)),alpha=.55,zorder=1)
                    ax.scatter(r.time_s_no_struct,y,s=size,marker=marker,fc='white',ec=colour,lw=1.2,zorder=4)
                ax.scatter(x,y,s=size,marker=marker,fc=colour,ec='white' if method.startswith('TIGER') else colour,lw=.5,zorder=4)
            offset=(-6,11) if method.startswith('TIGER') else (6,10)
            ha='right' if method.startswith('TIGER') else 'left'
            if method in ['TIGER-pair','TIGER-single']:
                offset=(9,0);ha='left'
            if method=='TIGER-ensemble':
                offset=(-10,12);ha='right'
            if method=='DeepAMP':offset=(0,-14);ha='center'
            if method=='EvoGradient':offset=(5,-14)
            if method=='APEX' and SHOW_BOTH_CONDITIONS:
                if METRIC=='PCC':y=max(a,b);offset=(6,10)
                else:offset=(12,-9)
            ax.annotate(label,(x,y),xytext=offset,textcoords='offset points',ha=ha,va='center',
                        fontsize=fs(LABEL_SIZE),color=textcolour,zorder=6)
        resolve_cost_annotation_overlaps(fig,ax,font_scale=FONT_SCALE)
        audit_path=OUT/'cost_label_layout_checks.json'
        audits=json.loads(audit_path.read_text()) if audit_path.exists() else {}
        audits['Fig5_accuracy_cost_pareto_TIGER_ensemble_'+METRIC]=fig._v12_label_audit
        audit_path.write_text(json.dumps(audits,indent=2))

        region=Patch(fc=AA['green'],ec='#BBEDB4',lw=.5,label=('Illustrative region (≤1 s, r ≥0.50)' if METRIC=='PCC' else 'Illustrative region (≤1 s, RSE ≤1)'))
        pareto=Line2D([],[],color=AA['frontier'],lw=1.4,ls=(0,(.1,2.7)),dash_capstyle='round',label='Point-estimate Pareto line')
        if SHOW_BOTH_CONDITIONS:
            handles=[region,pareto,Line2D([],[],marker='o',mfc='#626B75',mec='#626B75',lw=0,label='With negatives'),
                     Line2D([],[],marker='o',mfc='white',mec='#626B75',lw=0,label='Without negatives'),
                     Line2D([],[],marker='D',color='#626B75',lw=0,label='TIGER-pair'),
                     Line2D([],[],marker='s',color='#626B75',lw=0,label='TIGER-single')]
        else:
            handles=[region,pareto,Line2D([],[],marker='D',color='#626B75',lw=0,label='TIGER-pair'),
                     Line2D([],[],marker='s',color='#626B75',lw=0,label='TIGER-single'),
                     Line2D([],[],marker='*',markersize=10,color='#202020',lw=0,label='TIGER-ensemble'),
                     Line2D([],[],marker='o',color='#626B75',lw=0,label='External methods'),
                     Line2D([],[],marker='o',mfc='white',mec='#626B75',lw=0,label='Structures ready')]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.015,.841),ncol=4,frameon=False,
                   fontsize=fs(LEGEND_SIZE),handlelength=1.2,handletextpad=.4,columnspacing=1.2,borderaxespad=0)
        category_handles=[]
        for category in CATEGORY_ORDER:
            if category in ['multimodal_regression','multimodal_metric_learning']:
                handle=Line2D([],[],marker='s' if category=='multimodal_regression' else 'D',lw=0,
                              color=FIG5_CATEGORY_COLORS[category],label=CATEGORY_LABELS[category])
            else:handle=Patch(fc=FIG5_CATEGORY_COLORS[category],ec='none',label=CATEGORY_LABELS[category])
            category_handles.append(handle)
        fig.legend(handles=[category_handles[i] for i in [0,4,1,5,2,6,3,7]],loc='upper left',bbox_to_anchor=(.015,.775),
                   ncol=4,frameon=False,fontsize=fs(LEGEND_SIZE),handlelength=.8,handletextpad=.4,columnspacing=1.4,labelspacing=.7,borderaxespad=0)
        if SHOW_BOTH_CONDITIONS:
            notes=['Vertical lines join conditions at the new with-negative timing coordinate; same architecture and parameter count.',
                   ('Structural costs include generation. TIGER compares 00 / 11.' + (' †AMPLiT: with-negative PCC is undefined.' if METRIC=='PCC' else ''))]
        else:
            notes=[('Backbone conditions selected by mean PCC; descriptive comparison; ensemble weights fixed at 0.5 / 0.5.' + (' †AMPLiT: with-negative PCC is undefined.' if METRIC=='PCC' else '')),
                   'Filled: structure generation included; hollow: structures ready. The ensemble shares one structure library.']
        notes+=['*SenseXAMP-Ecoli and AMPpredictor also use physicochemical inputs. Pareto segments are visual guides, not evaluated models.']
        notes += ['Structure increment: 328.12 s for 208 peptides; Serial estimate from 30 measured peptides.']
        for y,text in zip([.084,.062,.040,.018],notes):fig.text(.015,y,text,fontsize=fs(NOTE_SIZE),color=AA['muted'])
        save(fig,'Fig5_accuracy_cost_pareto_TIGER_ensemble'+('' if METRIC=='PCC' else '_RSE'),CANVAS_WIDTH_MM,CANVAS_HEIGHT_MM,'Preserved geometry; V16 single and matched-seed ensemble timing.')
        return fig

