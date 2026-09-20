from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2]/'06_实验结果与日志/constraint_round2/analysis'
d=pd.read_csv(R/'summary.csv');d=d[(d.site=='hill_of_towie')&(d.scenario=='clean')&(d.ratio==2)&(d.period=='all')]
names=['none_f1.0_plain','curve_f1.0_plain','distribution_f1.0_plain','ridge'];labels=['DLinear','+ Curve prior','+ Distribution prior','Ridge']
fig,axs=plt.subplots(1,2,figsize=(10,4.2),layout='constrained')
for policy,color,marker,label in [('uncalibrated','#8996a4','o','Before calibration'),('quantile_calibrated','#156b76','s','After calibration')]:
    vals=[d[(d.group==n)&(d.policy==policy)].cost_mean.iloc[0] for n in names]
    axs[0].plot(range(4),vals,marker=marker,color=color,label=label,lw=1.6)
axs[0].set_xticks(range(4),labels,rotation=15,ha='right');axs[0].set_ylabel('Normalized next-hour imbalance loss')
axs[0].legend(frameon=False,fontsize=9);axs[0].set_title('A  Same decision calibration for every model',loc='left',fontsize=10)
c=pd.read_csv(R/'paired_cost_intervals.csv');c=c[(c.site=='hill_of_towie')&(c.policy=='quantile_calibrated')]
for i,n in enumerate(['bounds_f1.0_plain','curve_f1.0_plain','distribution_f1.0_plain']):
    r=c[c.method==n].iloc[0];lo=r.three_constraint_fwer_low;hi=r.three_constraint_fwer_high;point=r.delta_cost
    axs[1].errorbar(point*1000,i,xerr=np.array([[point-lo],[hi-point]])*1000,fmt='o',color='#156b76',capsize=4)
axs[1].axvline(0,color='#888888',lw=.8);axs[1].set_yticks(range(3),['Rated bound','Curve prior','Distribution prior']);axs[1].invert_yaxis()
axs[1].set_xlabel('Loss change vs calibrated DLinear (x 1,000)\nNegative favors constraint')
axs[1].set_title('B  Three-contrast adjusted paired intervals',loc='left',fontsize=10)
for ax in axs:
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.15)
fig.suptitle('Hill of Towie: new external test, 2017',fontsize=12)
fig.savefig(R/'external_calibration.png',dpi=180);fig.savefig(R/'external_calibration.svg')
