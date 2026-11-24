"""Deterministic numerical sanity checks for identities in the theory chapters.

These checks target transcription/sign errors in the manuscript, not trained models.
They are not a proof of the general theorems cited by the manuscript.
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    rng = random.Random(20261124)
    residuals = {}

    def check(name, actual, expected, tolerance=1e-9):
        error = abs(actual - expected)
        residuals[name] = max(error, residuals.get(name, 0.0))
        if error > tolerance:
            raise AssertionError((name, actual, expected, error))

    for _ in range(100):
        # Gaussian conditioning and Tweedie's formula, away from singular endpoints.
        a, b = rng.uniform(0.1, 1), rng.uniform(0.1, 2)
        mean, variance, x = rng.uniform(-2, 2), rng.uniform(0.1, 3), rng.uniform(-3, 3)
        cov = a*a*variance+b*b
        score = -(x-a*mean)/cov
        posterior = mean+a*variance/cov*(x-a*mean)
        check('tweedie_gaussian', a*posterior, x+b*b*score)

        # DDPM conditional posterior: precision addition versus closed form.
        alpha, abar_previous = rng.uniform(0.1, 0.99), rng.uniform(0.1, 0.99)
        beta = 1-alpha
        abar = alpha*abar_previous
        x0, eps = rng.uniform(-3, 3), rng.uniform(-3, 3)
        noisy = math.sqrt(abar)*x0+math.sqrt(1-abar)*eps
        posterior_var = beta*(1-abar_previous)/(1-abar)
        precision_var = 1/(alpha/beta+1/(1-abar_previous))
        posterior_mean = (math.sqrt(abar_previous)*beta*x0+
                          math.sqrt(alpha)*(1-abar_previous)*noisy)/(1-abar)
        precision_mean = precision_var*(math.sqrt(alpha)*noisy/beta+
                                        math.sqrt(abar_previous)*x0/(1-abar_previous))
        epsilon_mean = (noisy-beta/math.sqrt(1-abar)*eps)/math.sqrt(alpha)
        check('ddpm_variance', posterior_var, precision_var)
        check('ddpm_mean_precision', posterior_mean, precision_mean)
        check('ddpm_mean_noise', posterior_mean, epsilon_mean)

        # Affine velocity inversion, using a nonlinear angular schedule phi=t^2.
        t = rng.uniform(0.1, 0.9)
        a, b = math.cos(t*t), math.sin(t*t)
        da, db = -2*t*b, 2*t*a
        noisy, velocity = a*x0+b*eps, da*x0+db*eps
        determinant = a*db-b*da
        check('velocity_inverse_clean', (db*noisy-b*velocity)/determinant, x0)
        check('velocity_inverse_noise', (-da*noisy+a*velocity)/determinant, eps)
        check('rotation_not_time_velocity', velocity, 2*t*(a*eps-b*x0))

        # Integrating-factor first-order updates both equal deterministic DDIM.
        phi_t, phi_s = rng.uniform(0.7, 1.3), rng.uniform(0.05, 0.6)
        at, bt, ass, bs = math.cos(phi_t), math.sin(phi_t), math.cos(phi_s), math.sin(phi_s)
        xt, prediction = rng.uniform(-2, 2), rng.uniform(-2, 2)
        clean = (xt-bt*prediction)/at
        ddim = ass*clean+bs*prediction
        lt, ls = math.log(at/bt), math.log(ass/bs)
        dpm = ass/at*xt-ass*(math.exp(-lt)-math.exp(-ls))*prediction
        dpmpp = bs/bt*xt+bs*(math.exp(ls)-math.exp(lt))*clean
        check('ddim_equals_noise_integral', ddim, dpm)
        check('ddim_equals_data_integral', ddim, dpmpp)

        # OU variance solves V'=-2*k*V+sigma^2 and is positive.
        k, sigma = rng.uniform(0.1, 2), rng.uniform(0.1, 2)
        var = sigma*sigma/(2*k)*(1-math.exp(-2*k*t))
        derivative = sigma*sigma*math.exp(-2*k*t)
        check('ou_variance_differential', derivative, -2*k*var+sigma*sigma)
        assert var >= 0

        # MeanFlow identity for dx/dt=k*x, derivative in start time s.
        s, dest = rng.uniform(0.6, 1), rng.uniform(0, 0.4)
        delta = dest-s
        exponential = math.exp(k*delta)
        coefficient = math.expm1(k*delta)/delta
        derivative_start = -(k*exponential*delta-math.expm1(k*delta))/(delta*delta)
        average = coefficient*x
        rhs = k*x-(s-dest)*(derivative_start*x+coefficient*k*x)
        check('meanflow_identity', average, rhs)

        # Absorbing-mask reverse probabilities stay normalized and nonnegative.
        a_dest, a_start = rng.uniform(0.05, 0.4), rng.uniform(0.5, 0.95)
        unmask = (a_start-a_dest)/(1-a_dest)
        keep = (1-a_start)/(1-a_dest)
        check('mask_reverse_normalization', unmask+keep, 1)
        assert 0 <= keep <= 1 and 0 <= unmask <= 1

    # Analytic counterexample: independent Gaussian linear CFM is not one-step exact.
    for t in (0.1, 0.3, 0.5, 0.8, 0.99):
        c = (1-t)**2+t*t
        slope = (2*t-1)/c
        check('gaussian_cfm_variance_ode', 2*slope*c, 4*t-2)
    check('gaussian_cfm_exact_endpoint_scale', math.sqrt(1/1), 1)
    check('gaussian_cfm_euler_endpoint_scale', 1-(2-1)/1, 0)

    result = {'date': '2026-11-24', 'status': 'passed', 'random_cases_per_identity': 100,
              'seed': 20261124, 'maximum_absolute_residuals': residuals,
              'scope': 'Numerical transcription checks for analytic identities, not a general proof or model experiment.'}
    output = ROOT/'research/theory-update/math-checks.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
