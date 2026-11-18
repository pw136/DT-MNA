from figure_common import *
import figure_common as fc

def fixed_legend(a,handles=None,ncol=1,size=6.6):
 le=a.legend(handles=handles,ncol=ncol,fontsize=size,loc='upper left',bbox_to_anchor=(.018,.982),borderaxespad=0,borderpad=.18,labelspacing=.3,handlelength=1.05,handletextpad=.3,columnspacing=.62,frameon=True,framealpha=1,facecolor='white',edgecolor='none')
 a.figure.canvas.draw();area=a.get_window_extent();bb=le.get_window_extent();overlap=fc.hits(a,bb.padded(1.5))
 inside=area.contains(bb.x0,bb.y0) and area.contains(bb.x1,bb.y1)
 fc.LAYOUT.append(dict(figure=a.figure.number,panel=a.get_title(loc='left'),inside_axes=bool(inside),curve_or_value_overlaps=overlap))
 if not inside or overlap:raise AssertionError(('Legend collision',a.get_title(loc='left'),inside,overlap))
 return le

def figure9():
    legend0 = fc.legend
    save0 = fc.save
    def replace_legend(a, *args, **kw):
        t = a.get_title(loc='left')
        if t == 'Representation composition':
            return fixed_legend(a, ncol=4, size=6.4)
        if t == 'Accuracy across networks':
            return fixed_legend(a, handles=kw.get('handles'), ncol=2, size=6.7)
        return legend0(a, *args, **kw)
    def save9(f, n):
        f.canvas.draw()
        layout = []
        for i in (1, 2):
            a = f.axes[i]
            le = a.get_legend()
            bb = le.get_window_extent()
            box = a.get_window_extent()
            layout.append(dict(ncols=le._ncols, left_fraction=(bb.x0 - box.x0) / box.width))
        assert layout[0]['ncols'] == 4 and layout[1]['ncols'] == 2
        assert all(abs(r['left_fraction'] - .018) < .01 for r in layout)
        save0(f, n)
    fc.legend = replace_legend
    fc.save = save9
    fc.figure7()
    fc.legend = legend0
    fc.save = save0

def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (figure1, figure4, figure5, figure9):
        fn()
        print(fn.__name__, flush=True)

if __name__ == '__main__':
    main()
