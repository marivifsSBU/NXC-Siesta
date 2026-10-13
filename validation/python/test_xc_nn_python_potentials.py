#!/usr/bin/env python3
"""
test_xc_nn_python_potentials.py
===============================

Python reference script to evaluate NN-GGA exchange–correlation energies and
potentials on a fixed real-space grid using JAX/Equinox.

WHAT THIS SCRIPT DOES
---------------------
This script provides a *Python/JAX reference* for the NN-GGA functional used
in PySCF and SIESTA. Given a real-space grid file with columns:

  rho, gx, gy, gz, w

it:

  • loads the trained Fx and Fc Equinox models,
  • constructs the PySCF-compatible custom XC evaluator
    (get_my_eval_xc_ft_example),
  • evaluates the NN XC energy density eps_xc(r) and XC potentials
    using JAX automatic differentiation,
  • prints the first few grid-point values for inspection,
  • writes the full per-point results to an output file.

The PySCF-style GGA input array has shape (4, N):
  [rho, d rho/dx, d rho/dy, d rho/dz]

WHAT IT IS USED FOR
-------------------
This script is a *validation and debugging tool*. It was used to:

  • verify that the JAX-based NN XC energy densities and potentials are
    well defined on a given grid,
  • generate a trusted Python reference for comparison against the
    Fortran NN implementation in SIESTA,
  • confirm that analytic derivatives coded in Fortran match the
    JAX automatic-differentiation results point by point.

It is particularly useful when diagnosing discrepancies between
PySCF- and SIESTA-based NN calculations.

OUTPUT FILES
------------
The script writes:

  python_xc_potentials.dat

with columns:
  rho, gx, gy, gz, exc_py, vrho_py, vsigma_py

where:
  exc_py    = per-electron XC energy density
  vrho_py   = d( rho * eps_xc ) / d rho
  vsigma_py = d( rho * eps_xc ) / d sigma, with sigma = |grad rho|^2

The gradient derivative with respect to |grad rho| can be reconstructed as:

  d( rho*eps_xc )/d|grad rho| = 2 |grad rho| * vsigma

NOTES
-----
• This script does not perform self-consistent calculations.
• Integration weights are not used here; this is a pointwise comparison.
• Vacuum points (rho ≈ 0) may require masking, depending on the grid.

This script is intended for testing and validation, not for production use.
"""

import numpy as np
import jax
import jax.numpy as jnp

from modules.models import (
    GGA_Fx_G_transf_lin,
    GGA_Fc_G_transf_lin,
    load_eqx_model,
)
from customXC_good_NN import (
    eUEG_LDA_x,
    eUEG_LDA_c,
    get_my_eval_xc_ft_example,
)

jax.config.update("jax_enable_x64", True)


def load_models():
    params = {"depth": 3, "nodes": 16, "seed": 42}
    name_fx = f"{GGA_Fx_G_transf_lin.name}_d{params['depth']}_n{params['nodes']}_s{params['seed']}_v_20000"
    name_fc = f"{GGA_Fc_G_transf_lin.name}_d{params['depth']}_n{params['nodes']}_s{params['seed']}_v_20000"

    fx_path = "../pretrained_models/" + name_fx
    fc_path = "../pretrained_models/" + name_fc

    print("Loading Fx model:", fx_path)
    fx = load_eqx_model(fx_path)
    print("Loading Fc model:", fc_path)
    fc = load_eqx_model(fc_path)
    return fx, fc


if __name__ == "__main__":
    fx_model, fc_model = load_models()
    my_eval_xc = get_my_eval_xc_ft_example(fx_model, fc_model)

    data = np.loadtxt("grid_rho_grad_w.dat")
    rho = data[:, 0]
    gx = data[:, 1]
    gy = data[:, 2]
    gz = data[:, 3]
    w = data[:, 4]

    # PySCF-style GGA rho array: shape (4, N)
    rho_xc = np.stack([rho, gx, gy, gz], axis=0)

    exc, vxc, _, _ = my_eval_xc("NN", jnp.asarray(rho_xc),
                                spin=0, relativity=0, deriv=1)
    vrho, vsigma, _, _ = vxc

    print("i   rho         |grad|        exc_py       vrho_py      dedg_py")
    for i in range(10):
        g = np.sqrt(gx[i]**2 + gy[i]**2 + gz[i]**2)
        dedg_py = float(2*g * vsigma[i])
        print(f"{i:2d} {rho[i]:10.3e} {g:10.3e} {float(exc[i]):10.3e} "
              f"{float(vrho[i]):10.3e} {dedg_py:10.3e}")

    out = np.column_stack([rho, gx, gy, gz,
                           np.asarray(exc), np.asarray(vrho), np.asarray(vsigma)])
    np.savetxt("python_xc_potentials.dat",
               out,
               header="rho gx gy gz exc_py vrho_py vsigma_py")