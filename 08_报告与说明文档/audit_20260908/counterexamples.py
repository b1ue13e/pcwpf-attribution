"""Deterministic counterexamples; does not change project results."""
import json
from pathlib import Path
import numpy as np

def uniform_loss(p,a,b,r=2):
    assert a<=p<=b
    return (r*(p-a)**2+(b-p)**2)/(2*(b-a))

# Strongly convex MSE, nonzero prior gradient, but calibration removes all constant bias.
mu=.5;target=.6;lam=1.;theta=(mu+lam*target)/(1+lam);q=.25+.5/3
base=uniform_loss(q,.25,.75);constrained=uniform_loss(theta+(q-theta),.25,.75)
assert abs(base-constrained)<1e-14

# Conditional quantile shift depends on X: a biased predictor can outperform mean+constant.
means=np.array([.2,.5]);widths=np.array([.2,.8]);quantiles=np.array([1/6,11/30])
weighted=lambda v:np.sum(v/widths)/np.sum(1/widths)
delta0=weighted(quantiles-means)
f=(means+quantiles)/2;delta1=weighted(quantiles-f)
cost0=np.mean([uniform_loss(p,.1,.1+w) for p,w in zip(means+delta0,widths)])
cost1=np.mean([uniform_loss(p,.1,.1+w) for p,w in zip(f+delta1,widths)])
assert cost1<cost0
report={'constant_shift':{'mse_hessian':2,'prior_gradient_at_mean':-.2,'lambda':lam,
    'unconstrained_parameter':mu,'regularized_parameter':theta,'optimal_decisions_equal':True,
    'calibrated_risk_base':base,'calibrated_risk_regularized':constrained,'risk_gap':constrained-base},
    'heteroscedastic':{'mean_predictor':means.tolist(),'regularized_predictor':f.tolist(),
    'delta_mean':float(delta0),'delta_regularized':float(delta1),'base_calibrated_risk':cost0,
    'regularized_calibrated_risk':cost1,'excess_mse':float(np.mean((f-means)**2))}}
Path(__file__).with_name('counterexamples.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
