#!/usr/bin/env python3
"""JAX/PySCF-side NN-XC grid tester (spin-polarized).

Goal: evaluate the NN-XC *exactly through the same functions you use in PySCF*
(i.e. the eval_xc callback returned by get_my_eval_xc_ft), on the same
spin-resolved grid that the Fortran tester reads.

It prints the first few points and an integrated E_xc = sum w * rho_tot * eps_xc.

Input grid columns (one point per line):
  rho_u rho_d  gxu gyu gzu  gxd gyd gzd  w

Typical use:
  python test_xc_nn_jax_pyscf.py \
    --grid nn_rho_grad_w.dat \
    --fx-model /path/to/Fx_model_file_without_ext \
    --fc-model /path/to/Fc_model_file_without_ext

The model paths should point to the Equinox checkpoints you already use in PySCF,
expected to have both:
  <model>.eqx   and   <model>.json
"""

from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path
from typing import Union

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

jax.config.update("jax_enable_x64", True)


# -------------------- BEGIN: minimal imports from your notebook --------------------
# If you prefer, you can instead `import` these from your existing notebook-exported
# module. I kept them here so the tester is a single file.

class GGA_Fc_xcdiff(eqx.Module):
    """NN model for the correlation enhancement factor with LOB bound.

    Assumes the tinputs are transformed (see data preprocessing)

    These tinputs should be:
    - x0: log(rho^{1/3}+eps)
    - x1: log(zeta'+eps), zeta'=1/2*((1+zeta)^{4/3}+ (1-zeta)^{4/3})
    - x2: (1-exp(-s^2))*log(1+s)
    """

    name: str = 'GGA_Fc_xcdiff'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        if lob_lim is None:
            lob_lim = 2.0
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=3,
                              out_size=1,
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, tinputs):
        x0, x1, x2 = tinputs
        netterm = self.net(tinputs)
        lobterm = self.lobf(x2*netterm)
        return 1+lobterm.squeeze()

    def full_name(self, tail_info: Union[str, None] = None):
        if tail_info is None:
            return f'{self.name}_d{self.depth}_n{self.nodes}_s{self.seed}'
        else:
            return f'{self.name}_d{self.depth}_n{self.nodes}_s{self.seed}_{tail_info}'


class GGA_Fx_xcdiff(eqx.Module):
    name: str = 'GGA_Fx_xcdiff'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=1.804):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        if lob_lim is None:
            lob_lim = 1.804
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=1,
                              out_size=1,
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, tinputs):
        x2 = tinputs[2].flatten()
        netterm = self.net(x2)
        lobterm = self.lobf(x2*netterm)
        return 1+lobterm.squeeze()

    def full_name(self, tail_info: Union[str, None] = None):
        if tail_info is None:
            return f'{self.name}_d{self.depth}_n{self.nodes}_s{self.seed}'
        else:
            return f'{self.name}_d{self.depth}_n{self.nodes}_s{self.seed}_{tail_info}'


class LOB(eqx.Module):
    limit: float

    def __init__(self, limit: float):
        self.limit = limit

    def __call__(self, x):
        return self.limit * jax.nn.sigmoid(x-jnp.log(self.limit - 1))-1


def eUEG_LDA_x_unpol(rho):
    # energy per particle
    c = -0.75 * (3.0 / jnp.pi) ** (1.0 / 3.0)
    return jnp.where(rho <= 0.0, 0.0, c * rho ** (1.0 / 3.0))


def pw92_eps_c_wospin(rho_tot, zeta):
    if jnp.ndim(rho_tot) == 0:
        return __pw92_eps_c_wospin_point(rho_tot, zeta)
    pw92_eps_c_vectorized = jax.vmap(__pw92_eps_c_wospin_point, in_axes=(0, 0))
    return pw92_eps_c_vectorized(rho_tot, zeta)


def __pw92_eps_c_wospin_point(rho_tot, zeta):
    A = jnp.array([0.031090690869654895, 0.015545, 0.016887])
    ALPHA1 = jnp.array([0.21370, 0.20548, 0.11125])
    BETA1 = jnp.array([7.5957, 14.1189, 10.357])
    BETA2 = jnp.array([3.5876, 6.1977, 3.6231])
    BETA3 = jnp.array([1.6382, 3.3662, 0.88026])
    BETA4 = jnp.array([0.49294, 0.62517, 0.49671])

    rs = (3.0 / (4.0 * jnp.pi * rho_tot)) ** (1/3)

    def compute_g(rs_val):
        G = jnp.zeros((3,))
        for k in range(3):
            B = (BETA1[k] * jnp.sqrt(rs_val) +
                 BETA2[k] * rs_val +
                 BETA3[k] * rs_val**1.5 +
                 BETA4[k] * rs_val**2)
            C = 1 + 1 / (2 * A[k] * B)
            G = G.at[k].set(-2 * A[k] * (1 + ALPHA1[k] * rs_val) * jnp.log(C))
        return G

    G = compute_g(rs)

    C0 = 1.0 / (2.0**(4/3) - 2.0)
    F = ((1+zeta)**(4/3) + (1-zeta)**(4/3) - 2.0) * C0

    Fpp_0 = C0*8.0/9.0
    G0, G1, G2 = G

    eps_c = G0 - G2 * F / Fpp_0*(1 - zeta**4) + (G1 - G0)*F*zeta**4
    return eps_c


def transform_inputs_for_pyscf_grad_rho(rho_tot, zeta, grad_rho_mod):
    eps_log = 1e-5
    x0 = jnp.log(rho_tot**(1/3) + eps_log)
    zeta_prime = 0.5 * ((1 + zeta)**(4/3) + (1 - zeta)**(4/3))
    x1 = jnp.log(zeta_prime + eps_log)
    k_F = (3 * jnp.pi**2 * rho_tot)**(1/3)
    s = grad_rho_mod / (2 * k_F * rho_tot)
    x2 = (1 - jnp.exp(-s**2)) * jnp.log(1 + s)
    return x0, x1, x2


def transform_inputs_for_pyscf_sigma(rho_tot, zeta, sigma_tot):
    grad_rho_mod = jnp.sqrt(sigma_tot)
    return transform_inputs_for_pyscf_grad_rho(rho_tot, zeta, grad_rho_mod)


def custom_potential_ft_ex_sc_tinputs_pol(rho0u, rho0d, sigma_u, sigma_d, model_fx):
    zeta = jnp.zeros_like(rho0u)
    grad_rho_u_mod = jnp.sqrt(sigma_u)
    grad_rho_d_mod = jnp.sqrt(sigma_d)

    x0u, x1u, x2u = transform_inputs_for_pyscf_grad_rho(2*rho0u, zeta, 2*grad_rho_u_mod)
    tinputs_u = jnp.stack([x0u, x1u, x2u], axis=0)
    fxu = model_fx(tinputs_u)
    eUEG_u = eUEG_LDA_x_unpol(2*rho0u)

    x0d, x1d, x2d = transform_inputs_for_pyscf_grad_rho(2*rho0d, zeta, 2*grad_rho_d_mod)
    tinputs_d = jnp.stack([x0d, x1d, x2d], axis=0)
    fxd = model_fx(tinputs_d)
    eUEG_d = eUEG_LDA_x_unpol(2*rho0d)

    return rho0u * eUEG_u * fxu + rho0d * eUEG_d * fxd


def custom_potential_ft_ex_sc_tinputs_unpol(rho0_tot, zeta, sigma_tot, model_fx):
    x0, x1, x2 = transform_inputs_for_pyscf_sigma(rho0_tot, zeta, sigma_tot)
    tinputs = jnp.stack([x0, x1, x2], axis=-1)
    fx = model_fx(tinputs)
    eUEGx = eUEG_LDA_x_unpol(rho0_tot)
    return rho0_tot*fx*eUEGx


def custom_potential_ft_ec_sc_tinputs(rho0_tot, zeta, sigma_tot, model_fc):
    x0, x1, x2 = transform_inputs_for_pyscf_sigma(rho0_tot, zeta, sigma_tot)
    tinputs = jnp.stack([x0, x1, x2], axis=0)
    fc = model_fc(tinputs)
    return rho0_tot*fc*pw92_eps_c_wospin(rho0_tot, zeta)


def get_rho_sigma_polarized(rho):
    rho_u, rho_d = rho
    rho0u, dxu, dyu, dzu = rho_u[:4]
    rho0d, dxd, dyd, dzd = rho_d[:4]

    grad_rho_u = jnp.stack([dxu, dyu, dzu], axis=0)
    grad_rho_d = jnp.stack([dxd, dyd, dzd], axis=0)

    sigma_uu = jnp.sum(grad_rho_u**2, axis=0)
    sigma_dd = jnp.sum(grad_rho_d**2, axis=0)
    sigma_ud = jnp.sum(grad_rho_u*grad_rho_d, axis=0)
    return rho0u, rho0d, sigma_uu, sigma_ud, sigma_dd


def eval_xc_GGA_pol_sara(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                         custom_eps_x=None, custom_eps_c=None, eps=1e-15):
    if spin == 0:
        rho0_tot, dx, dy, dz = rho[:4]
        rho0_tot = jnp.array(rho0_tot)
        zeta = jnp.zeros_like(rho0_tot)
        sigma_tot = jnp.array(dx**2 + dy**2 + dz**2)
    else:
        rho0u, rho0d, sigma_uu, sigma_ud, sigma_dd = get_rho_sigma_polarized(rho)
        rho0_tot = jnp.array(rho0u + rho0d)
        zeta = (rho0u - rho0d) / rho0_tot
        sigma_tot = jnp.array(sigma_uu + 2*sigma_ud + sigma_dd)

    if spin == 0:
        eps_x = jax.vmap(custom_eps_x)(rho0_tot, zeta, sigma_tot)
    else:
        eps_x = jax.vmap(custom_eps_x)(rho0u, rho0d, sigma_uu, sigma_dd)
    eps_c = jax.vmap(custom_eps_c)(rho0_tot, zeta, sigma_tot)
    exc = (eps_x + eps_c) / rho0_tot

    def eps_xc(rho0u, rho0d, sigma_uu, sigma_ud, sigma_dd):
        rho0_tot_f = jnp.array(rho0u + rho0d)
        zeta_f = (rho0u - rho0d) / rho0_tot_f
        sigma_tot_f = jnp.array(sigma_uu + 2*sigma_ud + sigma_dd)
        eps_x_f = custom_eps_x(rho0u, rho0d, sigma_uu, sigma_dd)
        eps_c_f = custom_eps_c(rho0_tot_f, zeta_f, sigma_tot_f)
        return eps_x_f + eps_c_f

    def eps_xc_unpol(rho0_tot_f, sigma_tot_f):
        zeta_f = jnp.zeros_like(rho0_tot_f)
        return custom_eps_x(rho0_tot_f, zeta_f, sigma_tot_f) + custom_eps_c(rho0_tot_f, zeta_f, sigma_tot_f)

    if spin == 0:
        vrho, vsigma = jax.vmap(jax.grad(eps_xc_unpol, argnums=(0, 1)))(rho0_tot, sigma_tot)
        vrho = jnp.where(jnp.abs(rho0_tot) < eps, 0.0, vrho)
        vsigma = jnp.where(jnp.abs(rho0_tot) < eps, 0.0, vsigma)
    else:
        vrhou, vrhod, vsigma_uu, vsigma_ud, vsigma_dd = jax.vmap(
            jax.grad(eps_xc, argnums=(0, 1, 2, 3, 4))
        )(rho0u, rho0d, sigma_uu, sigma_ud, sigma_dd)
        vrhou = jnp.where(jnp.abs(rho0u) < eps, 0.0, vrhou)
        vrhod = jnp.where(jnp.abs(rho0d) < eps, 0.0, vrhod)
        mask_up = rho0u < eps
        mask_dn = rho0d < eps
        vsigma_uu = jnp.where(mask_up, 0.0, vsigma_uu)
        vsigma_dd = jnp.where(mask_dn, 0.0, vsigma_dd)
        vsigma_ud = jnp.where(mask_up | mask_dn, 0.0, vsigma_ud)
        vrho = jnp.vstack((vrhou, vrhod))
        vsigma = jnp.vstack((vsigma_uu, vsigma_ud, vsigma_dd))

    exc = jnp.where(jnp.abs(rho0_tot) < eps, 0.0, exc)

    vxc = (vrho, vsigma)
    fxc = None
    kxc = None
    vxc = [i.T for i in vxc]
    return exc, vxc, fxc, kxc


def eval_xc_GGA_pol_general(xc_code, rho, spin=0, relativity=0, deriv=1, omega=None, verbose=None,
                            custom_eps_x_p=None, custom_eps_x_up=None, custom_eps_c=None, eps=1e-15):
    if spin == 0:
        custom_eps_x = custom_eps_x_up
    else:
        custom_eps_x = custom_eps_x_p
    exc, vxc, fxc, kxc = eval_xc_GGA_pol_sara(
        xc_code, rho, spin, relativity, deriv, omega, verbose,
        custom_eps_x=custom_eps_x, custom_eps_c=custom_eps_c, eps=eps
    )
    return exc, vxc, fxc, kxc


def get_my_eval_xc_ft(model_fx, model_fc, eps=1e-15):
    custom_eps_x_p = partial(custom_potential_ft_ex_sc_tinputs_pol, model_fx=model_fx)
    custom_eps_x_up = partial(custom_potential_ft_ex_sc_tinputs_unpol, model_fx=model_fx)
    custom_eps_c = partial(custom_potential_ft_ec_sc_tinputs, model_fc=model_fc)
    return partial(
        eval_xc_GGA_pol_general,
        custom_eps_x_p=custom_eps_x_p,
        custom_eps_x_up=custom_eps_x_up,
        custom_eps_c=custom_eps_c,
        eps=eps,
    )


def load_pretrained_model(model_file: str):
    with open(f"{model_file}.json", "r") as f:
        metadata = json.load(f)

    name = metadata["name"]
    Model_Object = {"GGA_Fc_xcdiff": GGA_Fc_xcdiff, "GGA_Fx_xcdiff": GGA_Fx_xcdiff}.get(name)
    if Model_Object is None:
        raise ValueError(f"Model {name} not recognized")

    dummy_model = Model_Object(depth=metadata["depth"], nodes=metadata["nodes"], seed=metadata["seed"])
    model = eqx.tree_deserialise_leaves(f"{model_file}.eqx", like=dummy_model)
    print(f"Loaded {model_file}.eqx")
    return model


# -------------------- END: minimal imports from your notebook --------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="nn_rho_grad_w.dat")
    ap.add_argument("--fx-model", required=True, help="Path (no extension) to Fx eqx/json")
    ap.add_argument("--fc-model", required=True, help="Path (no extension) to Fc eqx/json")
    ap.add_argument("--print-n", type=int, default=10)
    args = ap.parse_args()

    model_fx = load_pretrained_model(args.fx_model)
    model_fc = load_pretrained_model(args.fc_model)
    my_eval = get_my_eval_xc_ft(model_fx=model_fx, model_fc=model_fc)

    grid = np.loadtxt(args.grid, dtype=np.float64, comments="#")
    if grid.shape[1] != 9:
        raise ValueError(f"Expected 9 columns, got {grid.shape[1]}")

    rho_u = grid[:, 0]
    rho_d = grid[:, 1]
    gxu, gyu, gzu = grid[:, 2], grid[:, 3], grid[:, 4]
    gxd, gyd, gzd = grid[:, 5], grid[:, 6], grid[:, 7]
    w = grid[:, 8]

    # Build PySCF-style rho containers for spin=1 GGA:
    # rho = (rho_u_block, rho_d_block), each block is shape (4, N): [rho, dx, dy, dz]
    rho_u_block = jnp.asarray(np.vstack([rho_u, gxu, gyu, gzu]))
    rho_d_block = jnp.asarray(np.vstack([rho_d, gxd, gyd, gzd]))
    rho = (rho_u_block, rho_d_block)

    exc, vxc, _, _ = my_eval("GGA", rho, spin=1, deriv=1)
    # vxc after transpose: vrho shape (N,2), vsigma shape (N,3)
    vrho, vsigma = vxc

    exc = np.asarray(exc)
    vrho = np.asarray(vrho)
    vsigma = np.asarray(vsigma)

    # Map PySCF-style vsigma back to dedg := d(eps_xc)/d|grad(rho_tot)|
    # We have sigma_uu = |gu|^2, sigma_ud = gu·gd, sigma_dd = |gd|^2
    # and vsigma = (dE/dsigma_uu, dE/dsigma_ud, dE/dsigma_dd)
    gu = np.stack([gxu, gyu, gzu], axis=1)
    gd = np.stack([gxd, gyd, gzd], axis=1)
    gtot = gu + gd
    gmod = np.linalg.norm(gtot, axis=1)

    vs_uu = vsigma[:, 0]
    vs_ud = vsigma[:, 1]
    vs_dd = vsigma[:, 2]

    dE_dgtot = (2.0*vs_uu + vs_ud)[:, None] * gu + (2.0*vs_dd + vs_ud)[:, None] * gd
    dedg = np.zeros_like(gmod)
    mask = gmod > 0
    dedg[mask] = (np.sum(dE_dgtot[mask] * gtot[mask], axis=1) / gmod[mask])

    # Integrated energy: sum w * rho_tot * exc
    rho_tot = rho_u + rho_d
    exc_int = float(np.sum(w * rho_tot * exc))

    n = min(args.print_n, exc.shape[0])
    for i in range(n):
        print(
            f"i={i+1:4d} rho_u={rho_u[i]:12.5E} rho_d={rho_d[i]:12.5E} "
            f"grad={gmod[i]:12.5E} exc={exc[i]:12.5E} "
            f"v_u={vrho[i,0]:12.5E} v_d={vrho[i,1]:12.5E} "
            f"vsuu={vsigma[i,0]:12.5E} vsud={vsigma[i,1]:12.5E} vsdd={vsigma[i,2]:12.5E}"
        )

    print(f"Read {exc.shape[0]:10d} grid points")
    print(f"JAX/PySCF integrated E_xc (spin grid) = {exc_int:20.12f}")


if __name__ == "__main__":
    main()
