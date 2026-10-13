# Spin-polarized O atom — NN-GGA example

Minimal end-to-end demonstration of the spin-polarized NN-XC path in this SIESTA fork.

## What this example computes

An isolated oxygen atom in its triplet ground state (`TotalSpin = 2`, i.e.
`N_up - N_dn = 2`) inside a 10 Å cubic cell. Two parallel runs are provided so
that the NN-GGA result can be compared directly against a stock GGA reference
on the exact same setup (geometry, basis, mesh, k-grid, SCF parameters):

- `NNPBE/Oatom.fdf` — runs with the neural-network GGA functional
  (`XC.authors NNGGA`), using the spin-polarized weights shipped in
  `NNPBE/nn_params/`.
- `PBE/Oatom.fdf`  — same input, but with the built-in Perdew-Burke-Ernzerhof
  functional (`XC.authors PBE`) as a reference.

Both inputs use TZP basis, a 500 Ry real-space mesh, and Pulay mixing
(`DM.NumberPulay 6`, `DM.MixingWeight 0.05`). The pseudopotential is
`O.psf` (same file in both subdirectories).

## NN-XC settings (in `NNPBE/Oatom.fdf`)

```
XC.functional     GGA
XC.authors        NNGGA
NNXC.Prefix       nn_params/xc_nn_spin
NNXC.FX_IN        1     # number of inputs to the Fx network
NNXC.FC_IN        3     # number of inputs to the Fc network
NNXC.H            16    # neurons per hidden layer
```

`NNXC.Prefix` is interpreted relative to the run directory. With the layout in
this example, the loader reads `nn_params/xc_nn_spin_{fx,fc}_{W,b}{1..4}.dat`
(16 files; 4-layer MLPs, biases `b` and weight matrices `W` per layer).

## How to run

Build SIESTA in this fork as usual (see `Obj/README` and the upstream build
instructions), then from each subdirectory:

```
cd NNPBE
<path-to>/siesta < Oatom.fdf | tee Oatom.out
```

```
cd ../PBE
<path-to>/siesta < Oatom.fdf | tee Oatom.out
```

## How to compare the two runs

After both runs finish, the quantities of interest are:

- Final total energy (`Etot` block in the `.out` file).
- Spin-resolved Mulliken populations (`WriteMullikenPop 1` is on).
- Eigenvalues (`O_atom.EIG`).

A working NN-GGA spin path should reproduce the PBE qualitative picture
(triplet O atom, two unpaired electrons on the 2p shell) while showing the
quantitative shift introduced by the cloned functional. The two
`Oatom.fdf` files are otherwise byte-for-byte identical apart from the
`XC.functional` / `XC.authors` / `NNXC.*` block, so any difference in
`Etot` or in the SCF trajectory is attributable to the functional.

## Files

```
Oatom/
├── README.md                  (this file)
├── NNPBE/
│   ├── Oatom.fdf              (NN-GGA input)
│   ├── O.psf                  (pseudopotential)
│   └── nn_params/
│       ├── xc_nn_spin_fx_{W,b}{1..4}.dat   (8 files, Fx network)
│       └── xc_nn_spin_fc_{W,b}{1..4}.dat   (8 files, Fc network)
└── PBE/
    ├── Oatom.fdf              (PBE reference input)
    └── O.psf                  (same pseudopotential)
```
