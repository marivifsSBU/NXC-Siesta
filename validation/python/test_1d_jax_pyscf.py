#!/usr/bin/env python3
import argparse
import numpy as np

# import from your already-working tester
import test_xc_nn_jax_pyscf as t  # uses your notebook-consistent eval path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-model", required=True)
    ap.add_argument("--fc-model", required=True)
    ap.add_argument("--n", type=int, default=2001)
    ap.add_argument("--L", type=float, default=5.0)
    ap.add_argument("--rho0", type=float, default=0.8)
    args = ap.parse_args()

    n = args.n
    L = args.L
    rho0 = args.rho0

    x = np.linspace(-L, L, n)
    dx = x[1] - x[0]

    rho = rho0 * (1.0 - (x/L)**2)
    rho = np.where(rho > 0.0, rho, 0.0)
    drdx = np.where(rho > 0.0, -2.0*rho0*x/(L*L), 0.0)

    # unpolarized represented as spin=0 path in your eval_xc implementation
    # PySCF expects rho array shape (4,N): [rho, dx, dy, dz]
    rho_block = np.vstack([rho, drdx, np.zeros_like(drdx), np.zeros_like(drdx)])

    model_fx = t.load_pretrained_model(args.fx_model)
    model_fc = t.load_pretrained_model(args.fc_model)
    my_eval = t.get_my_eval_xc_ft(model_fx=model_fx, model_fc=model_fc)

    # spin=0 for unpolarized
    exc, vxc, _, _ = my_eval("GGA", rho_block, spin=0, deriv=1)
    vrho, vsigma = vxc  # shapes: (N,1) and (N,1)

    exc = np.asarray(exc).reshape(-1)
    vrho = np.asarray(vrho).reshape(-1)      # d e_xc / d rho
    vsigma = np.asarray(vsigma).reshape(-1)  # d e_xc / d sigma, sigma=|grad rho|^2

    sigma = drdx * drdx
    edens = rho * exc

    # Clean vsigma near vacuum / numerical edge cases.
    # Even if drdx==0, NaNs can propagate through multiplication.
    mask = (rho > 0.0)
    vsigma = np.where(mask, vsigma, 0.0)
    vsigma = np.nan_to_num(vsigma, nan=0.0, posinf=0.0, neginf=0.0)

    # convert vsigma -> derivative wrt grad component:
    # de/d(drdx) = de/dsigma * d(sigma)/d(drdx) = vsigma * 2*drdx
    v_gradx = vsigma * (2.0 * drdx)

    Eint = np.trapezoid(edens, x)

    out = np.column_stack([np.arange(1, n+1), x, rho, drdx, exc, edens, vrho, v_gradx])
    np.savetxt("jax_1d.csv", out, delimiter=",",
               header="i,x,rho,drdx,exc,edens,vrho,vgradx", comments="")
    print(f"Wrote jax_1d.csv; 1D integral E_xc={Eint:.16e}")

if __name__ == "__main__":
    main()