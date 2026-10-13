#!/usr/bin/env python3
import argparse
import numpy as np
import jax.numpy as jnp

import test_xc_nn_jax_pyscf as t  # your notebook-consistent eval path

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fx-model", required=True)
    ap.add_argument("--fc-model", required=True)
    ap.add_argument("--n", type=int, default=2001)
    ap.add_argument("--L", type=float, default=5.0)
    ap.add_argument("--rho0", type=float, default=0.8)
    ap.add_argument("--out", default="jax_1d_spin1_equiv.csv")
    args = ap.parse_args()

    n = args.n
    L = float(args.L)
    rho0 = float(args.rho0)

    x = np.linspace(-L, L, n)
    rho = rho0 * (1.0 - (x/L)**2)
    rho = np.where(rho > 0.0, rho, 0.0)
    drdx = np.where(rho > 0.0, -2.0*rho0*x/(L*L), 0.0)

    model_fx = t.load_pretrained_model(args.fx_model)
    model_fc = t.load_pretrained_model(args.fc_model)
    my_eval = t.get_my_eval_xc_ft(model_fx=model_fx, model_fc=model_fc)

    # SIESTA nSpin=1 emulation: rho_u=rho_d=rho/2 and dxu=dxd=drdx/2
    rho_u = 0.5 * rho
    rho_d = 0.5 * rho
    dxu = 0.5 * drdx
    dxd = 0.5 * drdx

    rho_u_block = jnp.asarray(np.vstack([rho_u, dxu, np.zeros_like(dxu), np.zeros_like(dxu)]))
    rho_d_block = jnp.asarray(np.vstack([rho_d, dxd, np.zeros_like(dxd), np.zeros_like(dxd)]))
    rho_spin = (rho_u_block, rho_d_block)

    exc, vxc, _, _ = my_eval("GGA", rho_spin, spin=1, deriv=1)
    vrho, vsigma = vxc   # vrho: (N,2), vsigma: (N,3) = (vsuu, vsud, vsdd)

    exc = np.asarray(exc).reshape(-1)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)

    edens = rho * exc

    # Total-density vrho for SIESTA nSpin=1 convention:
    vrho_tot = 0.5 * (vrho[:,0] + vrho[:,1])

    # Convert vsigma -> dE/d(drdx) using spin-gradient formula:
    # dE/d(grad_u) = 2*vsuu*grad_u + vsud*grad_d
    # dE/d(grad_d) = 2*vsdd*grad_d + vsud*grad_u
    vsuu = vsigma[:,0]
    vsud = vsigma[:,1]
    vsdd = vsigma[:,2]

    dE_ddxu = 2.0*vsuu*dxu + vsud*dxd
    dE_ddxd = 2.0*vsdd*dxd + vsud*dxu

    # drdx = dxu + dxd, with dxu=dxd=drdx/2 => vgradx = 0.5*(dE/ddxu + dE/ddxd)
    vgradx = 0.5 * (dE_ddxu + dE_ddxd)
    vgradx = np.nan_to_num(vgradx, nan=0.0, posinf=0.0, neginf=0.0)

    out = np.column_stack([np.arange(1, n+1), x, rho, drdx, exc, edens, vrho_tot, vgradx])
    np.savetxt(args.out, out, delimiter=",",
               header="i,x,rho,drdx,exc,edens,vrho,vgradx", comments="")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()