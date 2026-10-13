#!/usr/bin/env python3
"""
Export Equinox MLP weights (any input size) to plain-text .dat files for use in Fortran.

Usage example:

    python export_eqx_to_fortran.py \
        --fx-model pretrained_models/GGA_Fx_G_transf_lin_d3_n16_s42_v_20000 \
        --fc-model pretrained_models/GGA_Fc_G_transf_lin_d3_n16_s42_v_20000 \
        --out-prefix xc_nn

This exporter is input-dimension agnostic (e.g. supports 1-, 2-, or 3-input models for unpolarized or spin-aware variants).
The Fortran side must be consistent with the exported W1 shapes.

"""

import argparse
import numpy as np
import jax
import equinox as eqx
import os

from models import load_eqx_model

def extract_mlp_params(mlp):
    """Return lists of (W, b) for each Linear layer in the Equinox MLP.

    Each W has shape (out_features, in_features), b has shape (out_features,).
    """
    weights = []
    biases = []
    # Equinox MLP stores Linear layers in `layers`
    for layer in mlp.layers:
        W = np.asarray(layer.weight)
        b = np.asarray(layer.bias)
        weights.append(W)
        biases.append(b)
    return weights, biases


def save_layer(out_dir, prefix, idx, W, b):
    """Save a single layer's weights and biases to text files.

    Files:
      {prefix}_W{idx}.dat
      {prefix}_b{idx}.dat
    """
    w_file = os.path.join(out_dir, f"{prefix}_W{idx}.dat")
    b_file = os.path.join(out_dir, f"{prefix}_b{idx}.dat")
    np.savetxt(w_file, W)
    np.savetxt(b_file, b)
    print(f"  saved {w_file} (shape {W.shape}), {b_file} (shape {b.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fx-model", required=True,
                        help="Path *without* .eqx/.json to Fx model "
                             "(e.g. pretrained_models/GGA_Fx_G_transf_lin_d3_n16_s42_v_20000)")
    parser.add_argument("--fc-model", required=True,
                        help="Path *without* .eqx/.json to Fc model "
                             "(e.g. pretrained_models/GGA_Fc_G_transf_lin_d3_n16_s42_v_20000)")
    parser.add_argument("--out-prefix", default="xc_nn",
                        help="Prefix for output files (default: xc_nn)")
    parser.add_argument("--expect-layers", type=int, default=4,
                        help="Expected number of Linear layers in each MLP (default: 4). Used for a sanity warning only.")
    parser.add_argument("--out-dir", default=".",
                        help="Directory where exported .dat files will be written (default: current directory)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    jax.config.update("jax_enable_x64", True)

    print("Loading Fx model from", args.fx_model)
    fx_model = load_eqx_model(args.fx_model)
    fx_mlp = fx_model.net  # GGA_Fx_G_transf_lin.net is eqx.nn.MLP

    print("Loading Fc model from", args.fc_model)
    fc_model = load_eqx_model(args.fc_model)
    fc_mlp = fc_model.net  # GGA_Fc_G_transf_lin.net is eqx.nn.MLP

    print("\nExtracting Fx MLP parameters...")
    fx_weights, fx_biases = extract_mlp_params(fx_mlp)
    print(f"  Fx MLP: {len(fx_weights)} layers")

    print("Extracting Fc MLP parameters...")
    fc_weights, fc_biases = extract_mlp_params(fc_mlp)
    print(f"  Fc MLP: {len(fc_weights)} layers")

    if len(fx_weights) != args.expect_layers or len(fc_weights) != args.expect_layers:
        print("\nWARNING: Unexpected layer count detected.")
        print(f"  Expected {args.expect_layers} layers; got Fx={len(fx_weights)}, Fc={len(fc_weights)}")
        print("  Export will proceed, but ensure the Fortran code expects the same architecture.")

    print(f"  Fx W1 shape = {fx_weights[0].shape} (out,in); Fc W1 shape = {fc_weights[0].shape} (out,in)")
    print(f"  Fx W_last shape = {fx_weights[-1].shape}; Fc W_last shape = {fc_weights[-1].shape}")

    # Save Fx layers
    print("\nSaving Fx layers...")
    for i, (W, b) in enumerate(zip(fx_weights, fx_biases), start=1):
        save_layer(args.out_dir, f"{args.out_prefix}_fx", i, W, b)

    # Save Fc layers
    print("\nSaving Fc layers...")
    for i, (W, b) in enumerate(zip(fc_weights, fc_biases), start=1):
        save_layer(args.out_dir, f"{args.out_prefix}_fc", i, W, b)

    print("\nDone. You can now point the Fortran loader at prefix =", args.out_prefix)


if __name__ == "__main__":
    main()
