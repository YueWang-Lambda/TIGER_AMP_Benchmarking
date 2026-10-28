"""v15 RSE Pareto plot; full drawing code embedded in the notebook."""

def cost_conditions_rse_figure():

    # ===== EDITABLE GEOMETRY AND TYPOGRAPHY / 尺寸与字体接口 =====
    CANVAS_WIDTH_MM = 325.12
    CANVAS_HEIGHT_MM = 190.5
    AXES_LEFT_MM = 26.01
    AXES_BOTTOM_MM = 34.0
    AXES_WIDTH_MM = 292.60
    AXES_HEIGHT_MM = 100.0
    FONT_SCALE = 1.0
    FONT_FAMILY = "Arial"
    TITLE_SIZE = 18.0
    LABEL_SIZE = 10.0
    TICK_SIZE = 9.0
    LEGEND_SIZE = 10.0
    NOTE_SIZE = 9.0
    POINT_SIZE = 48.0
    EXPORT_DPI = 600
    X_LIM = (0.001, 1500.0)
    METRIC = 'RSE'
    SHOW_BOTH_CONDITIONS = True
    Y_LIM = ((0.12, 0.72) if METRIC == "PCC" else
             ((0.58, 4.2) if SHOW_BOTH_CONDITIONS else (0.58, 1.40)))
    X_CUT, Y_CUT = 1.0, (0.50 if METRIC == "PCC" else 1.0)
    fs=lambda x: x*FONT_SCALE
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib.ticker import LogLocator, NullFormatter, FixedLocator, FixedFormatter
    AA=FIG5_THEME.copy()
    data=control_cost[METRIC] if SHOW_BOTH_CONDITIONS else control_best[METRIC]
    front=control_condition_fronts[METRIC]['new_structures']
    assert len(data)==(46 if SHOW_BOTH_CONDITIONS else 23)
    assert set(data.method)==set(METHOD_CATEGORY)
    assert data.time_s_with_struct.min()>=X_LIM[0]

    with plt.rc_context({'font.family':FONT_FAMILY,'font.size':fs(LABEL_SIZE),
                         'text.color':AA['ink'],'axes.labelcolor':AA['ink'],
                         'pdf.fonttype':42,'svg.fonttype':'none'}):
        fig=plt.figure(figsize=(CANVAS_WIDTH_MM/25.4,CANVAS_HEIGHT_MM/25.4),facecolor='white')
        noun='Prediction accuracy' if METRIC=='PCC' else 'Prediction error'
        fig.text(.015,.970,noun+' vs. inference cost, by method',fontsize=fs(TITLE_SIZE),va='top')
        mode='Both training conditions' if SHOW_BOTH_CONDITIONS else 'Best training condition per backbone'
        fig.text(.015,.916,mode+' · ten-seed mean '+METRIC+' · 208 peptides / 2,211 pairs',
                 fontsize=fs(LABEL_SIZE),color=AA['muted'],va='top')
        fig.text(.015,.886,'Shared time coordinates: accepted seed-123 medians; 20 repeats per method',
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
            ticks=[.6,.8,1,1.5,2,3,4] if SHOW_BOTH_CONDITIONS else [.6,.8,1,1.2,1.4]
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
            marker=method_marker(method);size=POINT_SIZE*(53/43 if method.startswith('TIGER') else 1.)
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
                if method in ['TIGER-pair','TIGER-single','GGN-ABPMIC']:
                    ax.plot([r.time_s_no_struct,x],[y,y],color=colour,lw=.9,ls=(0,(5,4)),alpha=.55,zorder=1)
                    ax.scatter(r.time_s_no_struct,y,s=size,marker=marker,fc='white',ec=colour,lw=1.2,zorder=4)
                ax.scatter(x,y,s=size,marker=marker,fc=colour,ec='white' if method.startswith('TIGER') else colour,lw=.5,zorder=4)
            offset=(-6,11) if method.startswith('TIGER') else (6,10)
            ha='right' if method.startswith('TIGER') else 'left'
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
        audits['ExtendedData5_accuracy_cost_conditions_RSE']=fig._v12_label_audit
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
                     Line2D([],[],marker='o',color='#626B75',lw=0,label='External methods'),
                     Line2D([],[],marker='o',mfc='white',mec='#626B75',lw=0,label='Structures ready')]
        fig.legend(handles=handles,loc='upper left',bbox_to_anchor=(.015,.841),ncol=len(handles),frameon=False,
                   fontsize=fs(LEGEND_SIZE),handlelength=1.2,handletextpad=.4,columnspacing=1.2,borderaxespad=0)
        category_handles=[]
        for category in CATEGORY_ORDER:
            if category in ['multimodal_regression','multimodal_metric_learning']:
                handle=Line2D([],[],marker='s' if category=='multimodal_regression' else 'D',lw=0,
                              color=FIG5_CATEGORY_COLORS[category],label=CATEGORY_LABELS[category])
            else:handle=Patch(fc=FIG5_CATEGORY_COLORS[category],ec='none',label=CATEGORY_LABELS[category])
            category_handles.append(handle)
        fig.legend(handles=[category_handles[i] for i in [0,4,1,5,2,6,3,7]],loc='upper left',bbox_to_anchor=(.015,.791),
                   ncol=4,frameon=False,fontsize=fs(LEGEND_SIZE),handlelength=.8,handletextpad=.4,columnspacing=1.4,labelspacing=.7,borderaxespad=0)
        if SHOW_BOTH_CONDITIONS:
            notes=['Vertical lines join conditions at the accepted with-negative timing coordinate; same architecture and parameter count.',
                   ('Structural costs include generation. TIGER compares 00 / 11.' + (' †AMPLiT: with-negative PCC is undefined.' if METRIC=='PCC' else ''))]
        else:
            notes=[('Condition selected by mean PCC; the RSE view uses the same condition. Test-set selection is descriptive.' + (' †AMPLiT: with-negative PCC is undefined.' if METRIC=='PCC' else '')),
                   'For structural models, filled points include structure generation and hollow points assume structures are ready.']
        notes+=['*SenseXAMP-Ecoli and AMPpredictor also use physicochemical inputs. Pareto segments are visual guides, not evaluated models.']
        notes += ['Structure increment: 328.12 s for 208 peptides; Serial estimate from 30 measured peptides.']
        for y,text in zip([.084,.062,.040,.018],notes):fig.text(.015,y,text,fontsize=fs(NOTE_SIZE),color=AA['muted'])
        save(fig,'ExtendedData5_accuracy_cost_conditions_RSE',CANVAS_WIDTH_MM,CANVAS_HEIGHT_MM,'RSE: both training conditions; retained Figure 5 design, with new timing and accuracy.')

