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
