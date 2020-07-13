
## MILLIFY
import math
millnames = ['',' Thousand',' Million']
def millify(n):
    n = float(n)
    millidx = max(0,min(len(millnames)-1,
                        int(math.floor(0 if n == 0 else math.log10(abs(n))/3))))
    return '{:.0f}{}'.format(n / 10**(3 * millidx), millnames[millidx])



## CORR PLOT
corr = df[ind_vars + [depvar]].corr()
cmap=sns.diverging_palette(5, 250, as_cmap=True)
def magnify():
    return [dict(selector="th",
                 props=[("font-size", "7pt")]),
            dict(selector="td",
                 props=[('padding', "0em 0em")]),
            dict(selector="th:hover",
                 props=[("font-size", "12pt")]),
            dict(selector="tr:hover td:hover",
                 props=[('max-width', '50px'),
                        ('font-size', '12pt')])
]
display(corr.style.background_gradient(cmap, axis=1)\
    .set_properties(**{'max-width': '40px', 'font-size': '8pt', 'max-height' : '200px'})\
    .set_caption("Hover to magnify")\
    .set_precision(2)\
    .set_table_styles(magnify()))


def std_scaler(v):
    scaler = StandardScaler()
    return scaler.fit_transform(v.reshape(-1, 1))

                
def anova_fn(self, v1, v2):
    stat, p = f_oneway(v1, v2)
    if p < 0.05:
        return 'group means are similar. the binary feature adds no value', p, ('stat=%.3f, p=%.4f' % (stat, p))
    else:
        return 'group means are dissimilar. feature has relationship with dependant variable', p, ('stat=%.3f, p=%.4f' % (stat, p))    

