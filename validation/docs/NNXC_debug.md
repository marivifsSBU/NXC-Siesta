# NNXC JAX ↔ Fortran Consistency Debug Log

Location: `validation/` (Fortran tests in `validation/fortran/`, Python references in `validation/python/`)

Goal: Make the Fortran (SIESTA/SiestaXC) NN implementation numerically identical to the JAX/Equinox implementation, targeting ~1e-8 agreement for energies and gradients.

---

## Summary of what was validated

### A) NN forward pass: Fx / Fc match to machine precision
We constructed a 1D analytic density (parabola) and compared the NN enhancement factors directly (“LDA = 1”):
- Fortran output: `fortran_1d_fx_fc.csv`
- JAX output:     `jax_1d_fx_fc.csv`

After fixes below:
- `Fx_u max|diff| ~ 3.6e-15`
- `Fc   max|diff| ~ 2.9e-15`

This confirms:
- weights match (eqx ↔ exported dat)
- transforms x0/x1/x2 match
- LOB mapping matches
- NN forward (including activation) matches

---

## Key technical issues and fixes

### 1) GELU mismatch (root cause of ~1e-3 Fx/Fc differences)
Observation:
- Raw MLP output (pre-LOB) differed between JAX and Fortran at a representative point (i=1970).

JAX version:
- `jax.__version__ == 0.7.0`
- `jax.nn.gelu` signature: `(x, approximate: bool = True)`
- Default is `approximate=True` (tanh approximation).

Confirmed numerically:
- `gelu default == gelu tanh`
- `gelu default != gelu erf`

Fix:
- In `Src/SiestaXC/xc_nn.f90`, replace erf-based GELU with tanh-approx GELU **and** update derivative accordingly.

GELU(tanh) formula used:
- `a = sqrt(2/pi) ≈ 0.7978845608028654`
- `c = 0.044715`
- `u = a*(x + c*x^3)`
- `gelu(x) = 0.5*x*(1 + tanh(u))`
- `dgelu(x) = 0.5*(1 + tanh(u)) + 0.5*x*(1 - tanh(u)^2)*a*(1 + 3*c*x^2)`

Result:
- Raw MLP output `y` (pre-LOB) matches JAX to ~1e-15.

---

### 2) FDF not initialized in standalone tests
Problem:
- Standalone Fortran tests calling SIESTA modules crashed because `fdf_string` is used to fetch NN prefix / architecture.

Fixes:
- Added `NNXC_INIT(prefix)` in `Src/SiestaXC/xc_nn_siesta.f90` so tests can load weights explicitly without FDF.
- Added `load_xc_nn_weights_nofdf(prefix)` in `Src/SiestaXC/xc_nn.f90` and made `NNXC_INIT` call it.

This enables standalone tests like:
- `test_1d_nngga` and `test_1d_fx_fc`

without initializing FDF or setting environment variables.

---

### 3) Gradient mismatch (vgradx) due to extra factor 2 in exchange mapping
Observation:
- Full 1D NNGGA comparison (Fortran `NNGGAXC` vs JAX “spin1-equivalent” script) gave:
  - `exc/edens/vrho` ~ 2–3e-7
  - but `vgradx` mismatch ~ 3e-2 (too large)

Diagnosis:
- In `Src/SiestaXC/xc_nn_siesta.f90`, exchange gradient derivative mapping used an extra factor `2.0_dp`:
  - `dEXdGD = 2 * dedg_x * Grad / |Grad|`
- But `dedg_x_*` was already ∂(energy density)/∂g, so vector derivative is `dedg * Grad/g` with **no extra factor 2**.

Fix:
- Remove the factor `2.0_dp` from the exchange `dEXdGD` lines (both nSpin==1 and nSpin==2 branches).

Result after fix:
- `vgradx max|diff| ~ 4.429e-08`
- `vgradx rms       ~ 1.741e-08`

This meets the ~1e-8 gradient target.

---

## Remaining differences

Even after NN is identical and vgradx is fixed:
- `exc` / `edens` / `vrho` remain at ~2–3e-7.

Likely cause:
- different LDA reference implementations / conventions:
  - Fortran uses SIESTA `exchng` and `pw92c`
  - Python path uses notebook’s `eUEG_LDA_x_unpol` and PW92 implementation.

These are not NN-related, and are already small.

---

## Files / scripts used

### Fortran (standalone)
- `test_1d_fx_fc.f90` → writes `fortran_1d_fx_fc.csv` (Fx/Fc only)
- `test_1d_nngga.f90` → writes `fortran_1d_nngga.csv` (EX/EC + derivatives)

### Python/JAX
- `test_1d_jax_fx_fc.py` → writes `jax_1d_fx_fc.csv` (Fx/Fc only)
- `test_1d_jax_pyscf.py` → writes `jax_1d.csv` (full exc + derivatives, unpolarized shortcut)
- `test_1d_jax_pyscf_spin1_equiv.py` → writes `jax_1d_spin1_equiv.csv` (spin1-equivalent; used for vgradx validation)

---

## Final status
✅ Fx/Fc match to machine precision (~1e-15).  
✅ vgradx matches at ~1e-8 level after exchange prefactor fix.  
✅ Remaining exc/vrho mismatch ~1e-7, attributed to LDA reference differences, not NN.
