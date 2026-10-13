#!/usr/bin/env python3
import argparse
import numpy as np

# JAX imports for pointwise model evaluation and safe math
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)



# Reuse your notebook-consistent utilities & model loader
import test_xc_nn_jax_pyscf as t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-model", required=True, help="Path without extension to Fx .eqx/.json")
    ap.add_argument("--fc-model", required=True, help="Path without extension to Fc .eqx/.json")
    ap.add_argument("--n", type=int, default=2001)
    ap.add_argument("--L", type=float, default=5.0)
    ap.add_argument("--rho0", type=float, default=0.8)
    ap.add_argument("--out", default="jax_1d_fx_fc.csv")
    args = ap.parse_args()

    n = args.n
    L = float(args.L)
    rho0 = float(args.rho0)

    x = np.linspace(-L, L, n)
    rho = rho0 * (1.0 - (x / L) ** 2)
    rho = np.where(rho > 0.0, rho, 0.0)

    # analytic derivative of parabola, zeroed where rho clipped
    drdx = np.where(rho > 0.0, -2.0 * rho0 * x / (L * L), 0.0)

    # Unpolarized: zeta = 0
    zeta = np.zeros_like(rho)
    grad_mod = np.abs(drdx)

    eps = 1e-5
    mask = rho > 0.0

    # x0 = log(rho^(1/3) + eps)
    x0 = np.zeros_like(rho)
    x0[mask] = np.log(rho[mask] ** (1.0 / 3.0) + eps)

    # x1 = log(zeta' + eps), with zeta'=1 when zeta=0
    x1 = np.zeros_like(rho)
    zeta_prime = 0.5 * ((1 + zeta) ** (4.0 / 3.0) + (1 - zeta) ** (4.0 / 3.0))
    x1[mask] = np.log(zeta_prime[mask] + eps)

    # x2 = (1-exp(-s^2))*log(1+s), s = |grad rho|/(2*kF*rho)
    x2 = np.zeros_like(rho)
    kF = np.zeros_like(rho)
    kF[mask] = (3.0 * np.pi**2 * rho[mask]) ** (1.0 / 3.0)
    s = np.zeros_like(rho)
    s[mask] = grad_mod[mask] / (2.0 * kF[mask] * rho[mask])
    x2[mask] = (1.0 - np.exp(-(s[mask] ** 2))) * np.log(1.0 + s[mask])

    tinputs = np.stack([x0, x1, x2], axis=0)  # shape (3, N)

    # Load Equinox models
    model_fx = t.load_pretrained_model(args.fx_model)
    model_fc = t.load_pretrained_model(args.fc_model)

    # Ensure model parameters are float64 (match Fortran dp)
    import jax.tree_util as jtu

    def to_f64(x):
        return x.astype(jnp.float64) if isinstance(x, jnp.ndarray) else x

    model_fx = jtu.tree_map(to_f64, model_fx)
    model_fc = jtu.tree_map(to_f64, model_fc)

    # Move to JAX arrays
    tinputs_j = jnp.asarray(tinputs)

    # Evaluate pointwise: vmap over columns (each column is a length-3 vector)
    Fx = np.asarray(jax.vmap(lambda ti: model_fx(ti), in_axes=1)(tinputs_j)).reshape(-1)
    Fc = np.asarray(jax.vmap(lambda ti: model_fc(ti), in_axes=1)(tinputs_j)).reshape(-1)

    # For rho==0 points, define neutral factors to avoid any edge behavior
    Fx = np.where(mask, Fx, 1.0)
    Fc = np.where(mask, Fc, 1.0)

    out = np.column_stack([np.arange(1, n + 1), x, rho, drdx, Fx, Fc])
    np.savetxt(
        args.out,
        out,
        delimiter=",",
        header="i,x,rho,drdx,Fx,Fc",
        comments="",
    )
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()