# NXC-Siesta: neural-network exchange-correlation functionals in SIESTA

NXC-Siesta runs neural-network GGA exchange-correlation functionals, trained in
JAX/Equinox, natively inside [SIESTA](https://siesta-project.org). The networks are
exported to plain-text parameter files and evaluated by a Fortran implementation of the
network in SIESTA's XC module, with analytic first derivatives. No libtorch or Python is
needed at run time.

It is the route used for the periodic SIESTA calculations of the cloning paper
(Navarro-Rodríguez *et al.*, arXiv:2605.10331). For PySCF and Quantum ESPRESSO, and for
meta-GGAs, see [libnxc](https://github.com/marivifsSBU/libnxc).

## Contents

| path | content |
|---|---|
| `siesta/` | `xc_nn.f90` (network forward pass and derivatives), `xc_nn_siesta.f90` (the `NNGGA` functional), and a patch against SIESTA `rel-4.1` |
| `export_eqx_to_fortran.py`, `export_nnxc.sh` | export an Equinox model pair (F_x, F_c) to SIESTA parameter files |
| `models.py` | the Equinox model classes, needed to load the checkpoints |
| `models/` | trained models: spin-aware PBE clones (`polarized/`) and unpolarized clones (`unpolarized/`) |
| `weights/` | exported parameter files, ready for SIESTA |
| `examples/Oatom/` | spin-polarized triplet O atom, NN-GGA vs PBE |
| `validation/` | Fortran tests and Python/JAX references, with the validation report |
| `legacy/` | earlier unpolarized tools, kept for reference |

## Getting a SIESTA build with NNGGA

The NN functional needs a SIESTA 4.1 build that includes the files in `siesta/`. Two options:

1. **Use the fork.** Clone https://gitlab.com/Marivi.Fernandez-Serra/siesta and check out
   branch `nnxc-spin` (spin-polarized and unpolarized, `nSpin = 1, 2`). Then build SIESTA as
   usual.
2. **Patch upstream SIESTA.** Apply `siesta/siesta-rel-4.1-nnxc-spin.patch` to SIESTA
   `rel-4.1` (commit `9d7c63051`) and build as usual:
   ```bash
   cd siesta-rel-4.1 && git apply /path/to/NXC-Siesta/siesta/siesta-rel-4.1-nnxc-spin.patch
   ```
   The patch adds the two Fortran files to `Src/SiestaXC/`, adds an `NNGGA` branch to
   `ggaxc.f`, and extends the SiestaXC makefile.

Non-collinear spin (`nSpin = 4`) is not supported; `NNGGAXC` aborts.

## 1. Export a trained model

```bash
python export_eqx_to_fortran.py \
    --fx-model models/polarized/pt_s05_TL_N20k_25m/GGA_Fx_xcdiff_d3_n16_s42_v_20000 \
    --fc-model models/polarized/pt_s05_TL_N20k_25m/GGA_Fc_xcdiff_d3_n16_s42_v_20000 \
    --out-prefix xc_nn_spin --out-dir nn_params
```
This writes `nn_params/xc_nn_spin_{fx,fc}_{W,b}{1..4}.dat`. `export_nnxc.sh` is a
wrapper that does the same and stages the files. It needs Python with `jax`, `equinox` and
`numpy`.

The Fortran code uses the tanh-approximate GELU (`jax.nn.gelu` with `approximate=True`, the
JAX default), as in training. The erf form changes E_xc by a few mEh.

## 2. Run SIESTA

```text
XC.Functional   GGA
XC.Authors      NNGGA
NNXC.Prefix     nn_params/xc_nn_spin
NNXC.FX_IN      1        # inputs of the F_x network
NNXC.FC_IN      3        # inputs of the F_c network
NNXC.H          16       # hidden-layer width
# NNXC.FX_MODE / NNXC.FC_MODE   TL (default, constrained) or UNCT (unconstrained models)
SpinPolarized   true     # optional; nSpin = 1 and 2 are supported
```
The prefix and architecture must match the exported model. For the unconstrained
`pt_s05_UT_N20k_25m` model, set `NNXC.FX_MODE UNCT` and `NNXC.FC_MODE UNCT`.

The enhancement factors multiply SIESTA's own LDA references: `EXCHNG` (exchange, per spin
channel) and `PW92C` (correlation).

## Models

| directory | model |
|---|---|
| `polarized/pt_s05_TL_N20k_25m` | spin-aware PBE clone with exact constraints (cloning paper) |
| `polarized/pt_s05_UT_N20k_25m` | the same, unconstrained |
| `polarized/pt_s05_TL_N20k_25m_SIESTA` | constrained clone trained on SIESTA (pseudo-valence) densities |
| `unpolarized/pretrained_in_slim05`, `pretrained_in_SIESTA_slim05`, `pretrained_in_x0x1grid` | unpolarized PBE clones |

`weights/polarized/` holds the export of `pt_s05_TL_N20k_25m`.

## Validation

`validation/fortran/` holds the tests, which build against the files in `siesta/`.
`validation/python/` holds the JAX/PySCF references.
`validation/docs/NNXC_validation_report.md` gives the full procedure. The steps are:

1. **Forward pass** (`test_1d_fx_fc.f90` vs `test_1d_jax_fx_fc.py`). F_x and F_c agree
   with the JAX model to machine precision: max |ΔF| = 4e-15 for `pt_s05_TL_N20k_25m`.
2. **First derivatives on a 1D analytic density** (`test_1d_nngga.f90`): about 1e-8.
3. **A real 3D SIESTA grid** (`test_xc_nn.f90` vs `test_xc_nn_jax_pyscf.py`): e_xc, energy
   density, ∂e/∂ρ and ∂e/∂∇ρ.

Do not compare the final grid potentials directly. PySCF treats the divergence term
∇·(∂e/∂∇ρ) analytically, while SIESTA evaluates it numerically on the real-space mesh
(Balbás, Martins and Soler, PRB 64, 165110 (2001)). Differences there are discretization,
not inconsistency.

## Authors and credits

Sara Navarro-Rodríguez (ICN2) and Marivi Fernández-Serra (Stony Brook University). See
[AUTHORS.md](AUTHORS.md).

## Citing

S. Navarro-Rodríguez, A. Wills, K. J. Daas, M. Camarasa-Gómez and M. Fernández-Serra,
*Constraint-aware functional cloning for stable and transferable machine-learned density
functional theory*, arXiv:2605.10331 (2026). See also [CITATION.cff](CITATION.cff).

## Acknowledgments

This work was supported by the U.S. National Science Foundation under award DMR-2427902.
S. N.-R. acknowledges the Joan Oró predoctoral program of the Generalitat de Catalunya
(2024 FI-I 00704).

## License

GPL-3.0 ([LICENSE](LICENSE)), the license of SIESTA, into which the Fortran code is
compiled.
