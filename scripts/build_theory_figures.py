"""Draw original vector teaching figures from analytic examples (no model run)."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'assets/theory'
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'axes.labelcolor': '#334155',
                     'text.color': '#1e293b', 'axes.titleweight': 'semibold'})
blue, orange = '#3173bc', '#db793b'

def save(fig, name):
    fig.savefig(OUT/(name+'.pdf'), bbox_inches='tight', metadata={'Creator': 'AIGC analytic teaching figures'})
    plt.close(fig)

# A stationary OU example. Use its exact finite-interval transition, not Euler.
rng = np.random.default_rng(20261124)
t = np.linspace(0, 2, 201)
x0 = rng.normal(size=10)
samples = np.empty((len(t), len(x0))); samples[0] = x0
decay = np.exp(-(t[1]-t[0]))
for i in range(1, len(t)):
    samples[i] = decay*samples[i-1]+np.sqrt(1-decay*decay)*rng.normal(size=len(x0))
fig, ax = plt.subplots(1, 2, figsize=(9, 2.5), sharex=True, sharey=True, layout='constrained')
colors = plt.cm.viridis(np.linspace(.1, .9, len(x0)))
for j,c in enumerate(colors):
    ax[0].plot(t, np.full_like(t, x0[j]), color=c, lw=1.2)
    ax[1].plot(t, samples[:, j], color=c, lw=.8, alpha=.85)
for a in ax:
    a.set(xlabel='Forward time', ylim=(-3.5, 3.5));a.grid(alpha=.15)
ax[0].set(ylabel='State x', title='PF-ODE: dx / dt = 0')
ax[1].set(title='OU SDE: dx = -x dt + sqrt(2) dW')
fig.suptitle('Both have the same analytic marginal N(0, 1) at every time', fontsize=10)
save(fig, 'same-marginals')

# Independent Gaussian endpoints under linear interpolation.
t = np.linspace(0, 1, 301)
c = (1-t)**2+t*t
xx = np.linspace(-3, 3, 401)
fig, ax = plt.subplots(1, 3, figsize=(10.3, 2.7), layout='constrained')
for tm,color in zip([0., .25, .5], [blue, orange, '#3b9b76']):
    variance = (1-tm)**2+tm*tm
    density = np.exp(-xx*xx/(2*variance))/np.sqrt(2*np.pi*variance)
    ax[0].plot(xx, density, color=color, label=f't={tm:g} (also {1-tm:g})')
ax[0].set(xlabel='State x', ylabel='Density', title='Analytic marginals');ax[0].legend(fontsize=7)
for z in [-2., -1., -.5, .5, 1., 2.]:
    ax[1].plot(t, np.sqrt(c)*z, lw=1.2)
ax[1].set(xlabel='Forward time t', ylabel='State x', title='Exact ODE paths')
ax[2].plot(t, np.sqrt(c), color=blue, lw=2, label='Exact std: sqrt(c(t))')
ax[2].scatter([0, 1], [0, 1], color=orange, zorder=4)
ax[2].plot([0, 1], [0, 1], '--', color=orange, label='One reverse Euler step')
ax[2].set(xlabel='Noise time t (generation: 1 to 0)', ylabel='Standard deviation', title='A one-step failure', ylim=(-.05, 1.15))
ax[2].legend(fontsize=7, loc='lower right')
for a in ax:a.grid(alpha=.15)
save(fig, 'gaussian-path')

# End-to-end scalar errors on an analytic field, counted by NFE.
def field(x,t): return (2*t-1)/((1-t)**2+t*t)*x
steps = np.array([1, 2, 4, 8, 16, 32, 64, 128])
errors = {'Euler': [], 'Heun': []}
for k in steps:
    h = -1/k
    for method in errors:
        x=1.
        for i in range(k):
            tt=1-i/k
            v=field(x,tt)
            x += h*v if method=='Euler' else .5*h*(v+field(x+h*v,tt+h))
        errors[method].append(abs(x-1))
fig, ax = plt.subplots(figsize=(7.4, 2.6), layout='constrained')
ax.loglog(steps, errors['Euler'], 'o-', color=blue, label='Euler (K evaluations)')
ax.loglog(2*steps, errors['Heun'], 's-', color=orange, label='Heun (2K evaluations)')
ax.set(xlabel='Number of field evaluations (NFE)', ylabel='Absolute endpoint scale error',
       title='Numerical error only: exact endpoint scale = 1')
ax.grid(which='both', alpha=.15);ax.legend(fontsize=8)
save(fig, 'solver-error')

record = {'date': '2026-11-24', 'kind': 'original analytic teaching figures',
          'ou_random_seed': 20261124, 'ou_transition': 'exact Gaussian OU transition',
          'neural_model_run': False, 'figures': ['same-marginals.pdf','gaussian-path.pdf','solver-error.pdf'],
          'solver_steps': steps.tolist(), 'solver_errors': errors}
(ROOT/'research/theory-update/figure-record.json').write_text(json.dumps(record, indent=2)+'\n',encoding='utf8')
print(json.dumps({'figures':record['figures']}))
