"""Numerical substitutions for newly added formulas, not theorem/model validation."""
import itertools
import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rng = random.Random(20261124)
residuals = {}

def check(name, lhs, rhs, tolerance=1e-9):
    error = abs(lhs-rhs)
    residuals[name] = max(residuals.get(name, 0.0), error)
    assert error <= tolerance * max(1.0, abs(lhs), abs(rhs)), (name,lhs,rhs,error)

def normal_log(x, mean, variance):
    return -0.5*(math.log(2*math.pi*variance)+(x-mean)**2/variance)

def normal_kl(m1,v1,m2,v2):
    return 0.5*(math.log(v2/v1)+(v1+(m1-m2)**2)/v2-1)

for _ in range(100):
    x = rng.uniform(-2,2)
    noise_var = rng.uniform(.2,2)
    qm = rng.uniform(-2,2)
    qv = rng.uniform(.2,2)
    posterior_var = noise_var/(1+noise_var)
    posterior_mean = x/(1+noise_var)
    elbo = -0.5*(math.log(2*math.pi*noise_var)+((x-qm)**2+qv)/noise_var)
    elbo -= normal_kl(qm,qv,0,1)
    check('vae_gap',normal_log(x,0,1+noise_var)-elbo,
          normal_kl(qm,qv,posterior_mean,posterior_var))

    precision = rng.uniform(.3,3)
    data_second_moment = rng.uniform(.2,3)
    def ebm_loss(q):
        return q*data_second_moment/2+math.log(2*math.pi/q)/2
    step=1e-5
    check('ebm_negative_phase',(ebm_loss(precision+step)-ebm_loss(precision-step))/(2*step),
          data_second_moment/2-1/(2*precision),1e-8)

    a,b = rng.uniform(.2,1.5),rng.uniform(.2,1.5)
    sigma2 = rng.uniform(.2,2)
    marginal_var=a*a*sigma2+b*b
    score_derivative=-1/marginal_var
    check('second_tweedie',b*b/(a*a)*(1+b*b*score_derivative),
          sigma2*b*b/marginal_var)

    signal_derivative = rng.uniform(-2,-.1)
    diffusion = rng.uniform(.1,2)
    variance_derivative=2*signal_derivative*b*b+diffusion*diffusion
    logsnr_derivative=2*signal_derivative-variance_derivative/(b*b)
    check('logsnr_weight',logsnr_derivative,-diffusion*diffusion/(b*b))

    time=rng.uniform(.05,.95)
    drift=-1/(1-time)
    recovered_diffusion2=2*time/(1-time)
    check('linear_flow_sde_reconstruction',2*drift*time*time+recovered_diffusion2,2*time)

    k=rng.uniform(-1.5,1.5)
    delta=rng.uniform(.05,1)
    exponential=math.exp(-k*delta)
    average=(1-exponential)/delta
    average_derivative=(k*delta*exponential-1+exponential)/(delta*delta)
    check('linear_meanflow_identity',average+delta*(average_derivative+k*average),k)
    check('linear_meanflow_map',x-delta*average*x,exponential*x)

    # Independently expand the state-dependent 1D Fokker--Planck terms.
    p=math.exp(-x*x/2)/math.sqrt(2*math.pi)
    dp=-x*p
    ddp=(x*x-1)*p
    A=1+.2*x*x
    dA=.4*x
    ddA=.4
    drift=-.3*x
    ddrift=-.3
    velocity=drift-.5*dA+.5*A*x
    dvelocity=ddrift-.5*ddA+.5*(dA*x+A)
    check('state_dependent_probability_flow',-(dp*velocity+p*dvelocity),
          -(ddrift*p+drift*dp)+.5*(ddA*p+2*dA*dp+A*ddp))

    # Verify the generic tilted-density PDE residual with a known Gaussian tilt.
    t=rng.uniform(.1,1)
    k=rng.uniform(.2,1)
    variance=1+t
    tilt_slope=k*t
    mean=tilt_slope*variance
    mean_derivative=k*(1+2*t)
    logz_derivative=k*k*t*variance+.5*k*k*t*t
    rhs=k*x-(-x/variance)*tilt_slope-.5*tilt_slope**2-logz_derivative
    lhs=(x-mean)*mean_derivative/variance
    check('tilted_path_residual',lhs,rhs)

    d=rng.uniform(.2,2)
    sigma=rng.uniform(.2,2)
    skip=d*d/(d*d+sigma*sigma)
    out=sigma*d/math.sqrt(sigma*sigma+d*d)
    residual_variance=(1-skip)**2*d*d+skip*skip*sigma*sigma
    check('edm_residual_scaling',residual_variance,out*out)

    # Exact integrals for an affine noise prediction in log-SNR.
    h=rng.uniform(.01,1)
    lam=rng.uniform(-1,1)
    E=rng.uniform(-2,2)
    derivative=rng.uniform(-2,2)
    scale_a=rng.uniform(.2,2)
    scale_b=scale_a*math.exp(-(lam+h))
    primitive=lambda u: -math.exp(-u)*(E+derivative*(u+1-lam))
    exact=scale_a*(primitive(lam+h)-primitive(lam))
    expansion=scale_b*(math.expm1(h)*E+(math.expm1(h)-h)*derivative)
    check('dpm_second_order_integral',exact,expansion)

    primitive_data=lambda u: math.exp(u)*(E+derivative*(u-lam-1))
    exact_data=scale_b*(primitive_data(lam+h)-primitive_data(lam))
    expansion_data=scale_a*((1-math.exp(-h))*E+(h-1+math.exp(-h))*derivative)
    check('dpmpp_second_order_integral',exact_data,expansion_data)

    start=rng.uniform(-2,2)
    target=rng.uniform(-2,2)
    at,bt=.3,.9
    az,bz=.9,.2
    pseudo=(target-bz/bt*start)/(az-bz*at/bt)
    check('progressive_distillation_target',bz/bt*start+(az-bz*at/bt)*pseudo,target)

    rate=rng.uniform(.1,2)
    step=rng.uniform(.01,.2)
    sigma=rng.uniform(.1,2)
    stable_variance=sigma*sigma/(2*rate-rate*rate*step)
    check('ou_euler_stationarity',(1-rate*step)**2*stable_variance+sigma*sigma*step,
          stable_variance)

    s,t=rng.uniform(.1,.3),rng.uniform(.6,1)
    u,ustar=rng.uniform(-2,2),rng.uniform(-2,2)
    g=start-t*u
    gstar=start-t*ustar
    G=s/t*start+(t-s)/t*g
    Gstar=start-(t-s)*ustar
    check('ctm_meanflow_error',(G-Gstar)**2,(t-s)**2*(u-ustar)**2)
    check('ctm_residual_error',(G-Gstar)**2,((t-s)/t)**2*(g-gstar)**2)

matrix=[[.4,-.8],[1.2,-.3]]
estimate=0
for vec in itertools.product([-1,1],repeat=2):
    estimate+=sum(vec[i]*matrix[i][j]*vec[j] for i in range(2) for j in range(2))/4
check('hutchinson_rademacher',estimate,matrix[0][0]+matrix[1][1])

# A finite-state forward chain checked by enumeration, separately from matrix formulas.
prior=[.2,.5,.3]
Q1=[[.7,.2,.1],[.1,.7,.2],[.3,.2,.5]]
Q2=[[.6,.1,.3],[.2,.7,.1],[.1,.3,.6]]
joint=[[[prior[l]*Q1[l][j]*Q2[j][i] for i in range(3)]
         for j in range(3)] for l in range(3)]
p2=[sum(joint[l][j][i] for l in range(3) for j in range(3)) for i in range(3)]
for i in range(3):
    reconstructed=[0,0,0]
    for l in range(3):
        conditional=sum(Q1[l][j]*Q2[j][i] for j in range(3))
        clean_posterior=sum(joint[l][j][i] for j in range(3))/p2[i]
        for j in range(3):
            reconstructed[j]+=Q1[l][j]*Q2[j][i]/conditional*clean_posterior
    for j in range(3):
        check('discrete_reverse_mixture',reconstructed[j],sum(joint[l][j][i] for l in range(3))/p2[i])

rates=[[0, .4,.2],[.3,0,.1],[.5,.2,0]]
p=[.2,.3,.5]
dp=[sum(p[j]*rates[j][i] for j in range(3) if j!=i)
    -p[i]*sum(rates[i]) for i in range(3)]
reverse=[[rates[j][i]*p[j]/p[i] if j!=i else 0 for j in range(3)] for i in range(3)]
for i in range(3):
    reverse_dp=sum(p[j]*reverse[j][i] for j in range(3) if j!=i)-p[i]*sum(reverse[i])
    check('ctmc_reverse_clock',reverse_dp,-dp[i])

# Nonlinear tangent residual has a parameter-dependent variance term.
values=[-1,2]
weights=[.4,.6]
mean=sum(w*v for w,v in zip(weights,values))
variance=sum(w*(v-mean)**2 for w,v in zip(weights,values))
time_derivative,jacobian=.7,1.3
raw=sum(w*(time_derivative+jacobian*v)**2 for w,v in zip(weights,values))
check('consistency_conditional_variance',raw,(time_derivative+jacobian*mean)**2+jacobian**2*variance)

alpha=lambda t:math.exp(-t)
for t in [.2,.5,.9]:
    h=1e-6
    finite_rate=(alpha(t-h)-alpha(t))/(1-alpha(t))/h
    rate=alpha(t)/(1-alpha(t))
    check('mask_unmask_rate',finite_rate,rate,1e-6)

# Marginalize two-point conditional discrete flows over independent endpoint pairs.
source=[.5,.3,.2]
target=[.1,.4,.5]
coupling=[[source[i]*target[j] for j in range(3)] for i in range(3)]
for time in [.2,.7]:
    marginal=[(1-time)*source[i]+time*target[i] for i in range(3)]
    marginal_rates=[[coupling[i][j]/marginal[i] if i!=j else 0
                     for j in range(3)] for i in range(3)]
    for j in range(3):
        change=sum(marginal[i]*marginal_rates[i][j] for i in range(3) if i!=j)
        change-=marginal[j]*sum(marginal_rates[j])
        check('discrete_flow_conditional_current',change,target[j]-source[j])

record={'date':'2026-11-24','status':'passed','random_cases':100,'seed':20261124,
        'maximum_absolute_residuals':residuals,
        'scope':'Numerical substitutions and finite-state enumerations for new formulas; no general theorem proof, model training or inference.'}
(ROOT/'research/theory-update/supplement-math-checks.json').write_text(
    json.dumps(record,indent=2)+'\n',encoding='utf-8')
print(json.dumps(record))
