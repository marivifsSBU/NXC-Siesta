import jax.numpy as jnp
import jax
import equinox as eqx
import json
from typing import Union
import numpy as np

# ----------------- XC networks ---------------------------




# --------------------- FC net------------------------------------------
# Define the neural network module for Fc

class FcNet_simple(eqx.Module):
    """
    Correlation enhancement factor without any constrains.
    """
    name: str = 'Fc_s'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int,):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, grad_rho
                              out_size=1,  # Output is Fx
                              depth=self.depth, width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):
        return self.net(inputs).squeeze()


class FcNet_LOB(eqx.Module):
    """Neural network for the correlation enhancement factor imposing LOB.

    We impose the Lieb-Oxford bound as a constrain with a default value of a=2.
    """
    name: str = 'Fc_LOB'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, grad_rho
                              out_size=1,  # Output is Fx
                              depth=self.depth, width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        return 1 + \
            self.lobf((jnp.tanh(inputs[1])**2)*self.net(inputs).squeeze())


class FcNetG_LOB(eqx.Module):
    name: str = 'FcG_LOB'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        # rho = rho.flatten()
        rho = inputs[0].flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class FcNetG_simple(eqx.Module):
    name: str = 'FcG_s'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):
        # rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        # rho = rho.flatten()
        rho = inputs[0].flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()

        return self.net(netinp).squeeze()


class FcNetSIG_LOB(eqx.Module):
    name: str = 'FcSIG_LOB'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, sigma]
        rho, sigma = inputs
        # rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        rho = rho.flatten()
        # sigma = jnp.maximum(1e-12, inputs[1])  # Prevents division by 0
        sigma = sigma.flatten()

        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)

        s = s.flatten()
        netinp = jnp.stack([rho, s], axis=0).flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()
    

class FcNetSIGt_LOB(eqx.Module):
    name: str = 'FcSIGt_LOB'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, sigma]
        rho, sigma = inputs
        rho = rho.flatten()
        sigma = sigma.flatten()

        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        k_s = jnp.sqrt(4*k_F/jnp.pi)
        t = jnp.sqrt(sigma) / (2 * k_s * rho)

        t = t.flatten()
        netinp = jnp.stack([rho, t], axis=0).flatten()
        tanhterm = jnp.tanh(t)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class GGA_Fc_G_transf_lin(eqx.Module):
    name: str = 'GGA_Fc_G_transf_lin'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, grad_rho]

        eps = 1e-5
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()

        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = grad_rho / (2 * k_F * rho)

        s = s.flatten()
        s_sq = s**2

        x0 = rho**(1/3)
        x0 = jnp.log10(x0+eps)
        x1 = jnp.log10(s+1)*(1-jnp.exp(-s_sq))

        netinp = jnp.stack([x0, x1], axis=0).flatten()
        netterm = self.net(netinp)
        lobterm = self.lobf(x1*netterm)
        return 1+lobterm.squeeze()


# ---------------------------------------------------------------
# Spin-polarized PBE-cloning models (xcdiff naming, in_size=3)

class GGA_Fc_xcdiff(eqx.Module):
    """Spin-capable PBE-cloning correlation enhancement network (xcdiff naming).

    This model expects as input the three *precomputed* features (x0, x1, x2)
    used in the polarized notebooks:
      x0 = log10(rho^(1/3) + eps)
      x1 = log10(zeta_prime + eps)
      x2 = log10(1+s)*(1-exp(-s^2))

    The exporter only needs the architecture to match the saved .eqx leaves.
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(
            in_size=3,      # x0, x1, x2
            out_size=1,
            depth=self.depth,
            width_size=self.nodes,
            activation=jax.nn.gelu,
            key=jax.random.PRNGKey(self.seed),
        )
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # inputs is expected to be (x0, x1, x2)
        x0, x1, x2 = inputs
        x0 = jnp.asarray(x0).flatten()
        x1 = jnp.asarray(x1).flatten()
        x2 = jnp.asarray(x2).flatten()
        netinp = jnp.stack([x0, x1, x2], axis=0).flatten()
        netterm = self.net(netinp)
        # Follow the same LOB pattern used elsewhere: 1 + LOB(x2 * netterm)
        lobterm = self.lobf(x2 * netterm)
        return 1 + lobterm.squeeze()


# ---------------------------------------------------------------
# Unconstrained spin-capable PBE-cloning models (no LOB wrapper)
class GGA_Fc_xcdiff_unct(eqx.Module):
    """Unconstrained spin-capable PBE-cloning correlation network.

    Same MLP architecture as GGA_Fc_xcdiff, but without the Lieb–Oxford wrapper.
    The exporter only needs the architecture to match the saved .eqx leaves.
    """
    name: str = 'GGA_Fc_xcdiff_unct'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=None):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.net = eqx.nn.MLP(
            in_size=3,      # x0, x1, x2
            out_size=1,
            depth=self.depth,
            width_size=self.nodes,
            activation=jax.nn.gelu,
            key=jax.random.PRNGKey(self.seed),
        )

    def __call__(self, inputs):
        # inputs is expected to be (x0, x1, x2)
        x0, x1, x2 = inputs
        x0 = jnp.asarray(x0).flatten()
        x1 = jnp.asarray(x1).flatten()
        x2 = jnp.asarray(x2).flatten()
        netinp = jnp.stack([x0, x1, x2], axis=0).flatten()
        netterm = self.net(netinp)
        return netterm.squeeze()


class GGA_Fx_xcdiff(eqx.Module):
    """Spin-capable PBE-cloning exchange enhancement network (xcdiff naming).

    NOTE: The trained xcdiff-style Fx models in this repository are gradient-feature-only.
    They expect a single input (x2), where x2 is the precomputed gradient feature
    (e.g. x2 = log10(1+s)*(1-exp(-s^2))). Spin dependence of exchange is handled via
    the spin-scaling of the LDA exchange reference in the SIESTA/PySCF wrappers.

    For convenience, this __call__ accepts either:
      - inputs = (x0, x1, x2)  (we will use x2)
      - inputs = x2           (direct)
    """
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(
            in_size=1,      # x2 only (gradient feature)
            out_size=1,
            depth=self.depth,
            width_size=self.nodes,
            activation=jax.nn.gelu,
            key=jax.random.PRNGKey(self.seed),
        )
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Accept either (x0,x1,x2) or x2
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            x2 = inputs[2]
        else:
            x2 = inputs

        x2 = jnp.asarray(x2).flatten()
        netterm = self.net(x2)
        lobterm = self.lobf(x2 * netterm)
        return 1 + lobterm.squeeze()


class GGA_Fx_xcdiff_unct(eqx.Module):
    """Unconstrained spin-capable PBE-cloning exchange network.

    Same MLP architecture as GGA_Fx_xcdiff, but without the Lieb–Oxford wrapper.
    It expects a single input x2 (or a 3-tuple where only x2 is used).
    """
    name: str = 'GGA_Fx_xcdiff_unct'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=None):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.net = eqx.nn.MLP(
            in_size=1,      # x2 only (gradient feature)
            out_size=1,
            depth=self.depth,
            width_size=self.nodes,
            activation=jax.nn.gelu,
            key=jax.random.PRNGKey(self.seed),
        )

    def __call__(self, inputs):
        # Accept either (x0,x1,x2) or x2
        if isinstance(inputs, (tuple, list)) and len(inputs) == 3:
            x2 = inputs[2]
        else:
            x2 = inputs

        x2 = jnp.asarray(x2).flatten()
        netterm = self.net(x2)
        return netterm.squeeze()


class GGA_Fc_G_transf(eqx.Module):
    name: str = 'GGA_Fc_G_transf'
    depth: int
    nodes: int
    seed: int
    lob_lim: float
    net: eqx.nn.MLP
    lobf: eqx.Module
    name: str

    def __init__(self, depth: int, nodes: int, seed: int, lob_lim=2.0):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, grad_rho]

        eps = 1e-5
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()

        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = grad_rho / (2 * k_F * rho)

        s = s.flatten()
        s_sq = s**2

        x0 = rho**(1/3)
        x0 = jnp.log10(x0+eps)
        x1 = jnp.log10(s+1)*(1-jnp.exp(-s_sq))

        netinp = jnp.stack([x0, x1], axis=0).flatten()
        tanhterm = jnp.tanh(x1)**2
        netterm = self.net(netinp)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()

# ------------------------------ FX net -----------------------------------

# Define the neural network module for Fx


class FxNet_simple(eqx.Module):
    """
    Model for the exchange enhancement factor
    """
    name: str = 'Fx_s'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        self.net = eqx.nn.MLP(in_size=2,  # Input is rho, grad_rho
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):
        return self.net(inputs).squeeze()


class FxNet_LOB(eqx.Module):
    name: str = 'Fx_LOB'
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
        self.lob_lim = lob_lim
        #  to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY grad_rho
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        """
        Assume the inputs is [rho, grad] and select the appropriate input
        Takes forever if inputs[1] tanh input has extended shape, i.e. (1,1)
        as opposed to scalar shape (1,)
        S: inputs[1] in this LOB, would not be s?
        """
        return 1+self.lobf((jnp.tanh(inputs[1])**2) *
                           self.net(inputs[1, jnp.newaxis]).squeeze())


class FxNetG_LOB(eqx.Module):
    name: str = 'FxG_LOB'
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
        self.lob_lim = lob_lim
        # to constrain this, we require only gradient inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        # rho = rho.flatten()
        # print('WITHOUT RHO MAXIMUM')
        rho = inputs[0].flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(s)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class FxNetG_simple(eqx.Module):
    name: str = 'FxG_s'
    depth: int
    nodes: int
    seed: int
    net: eqx.nn.MLP

    def __init__(self, depth: int, nodes: int, seed: int):
        self.depth = depth
        self.nodes = nodes
        self.seed = seed
        # to constrain this, we require only s inputs
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))

    def __call__(self, inputs):

        rho = jnp.maximum(1e-12, inputs[0])  # Prevents division by 0
        rho = rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = inputs[1].flatten() / (2 * k_F * rho)
        s = s.flatten()
        return self.net(s).squeeze()


class FxNetSIG_LOB(eqx.Module):
    name: str = 'FxSIG_LOB'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, sigma]
        rho, sigma = inputs
        # rho = jnp.maximum(1e-12, inputs[0]) #Prevents division by 0
        rho = rho.flatten()
        # sigma = jnp.maximum(1e-12, inputs[1]) #Prevents division by 0
        sigma = sigma.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)
        s = jnp.sqrt(sigma) / (2 * k_F * rho)
        s = s.flatten()
        tanhterm = jnp.tanh(s)**2
        netterm = self.net(s)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()


class GGA_Fx_G_transf_lin(eqx.Module):
    name: str = 'GGA_Fx_G_transf_lin'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, grad_rho]
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)

        s = grad_rho / (2 * k_F * rho)
        s = s.flatten()
        s_sq = s**2

        x1 = jnp.log10(s+1)*(1-jnp.exp(-s_sq))
        netterm = self.net(x1)
        lobterm = self.lobf(x1*netterm)
        return 1+lobterm.squeeze()


class GGA_Fx_G_transf(eqx.Module):
    name: str = 'GGA_Fx_G_transf'
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
        self.lob_lim = lob_lim
        self.net = eqx.nn.MLP(in_size=1,  # Input is ONLY s
                              out_size=1,  # Output is Fx
                              depth=self.depth,
                              width_size=self.nodes,
                              activation=jax.nn.gelu,
                              key=jax.random.PRNGKey(self.seed))
        self.lobf = LOB(limit=lob_lim)

    def __call__(self, inputs):
        # Here, assume the inputs is [rho, grad_rho]
        rho, grad_rho = inputs
        rho = rho.flatten()
        grad_rho = grad_rho.flatten()
        k_F = (3 * jnp.pi**2 * rho)**(1/3)

        s = grad_rho / (2 * k_F * rho)
        s = s.flatten()
        s_sq = s**2

        x1 = jnp.log10(s+1)*(1-jnp.exp(-s_sq))
        tanhterm = jnp.tanh(x1)**2
        netterm = self.net(x1)
        lobterm = self.lobf(tanhterm*netterm)
        return 1+lobterm.squeeze()

# LOB
# Define the equinox module for the Lieb-Oxford bound enforcement


class LOB(eqx.Module):
    limit: float

    def __init__(self, limit: float):
        self.limit = limit

    def __call__(self, x):
        # This formulation is so that the constraints can be enforced
        # while avoiding potential gradient issues with the sigmoid
        return self.limit * jax.nn.sigmoid(x-jnp.log(self.limit - 1))-1


# Saving models
def save_eqx_model(model, path: str = '', fixing: Union[str, None] = None,
                   tail_info: Union[str, None] = None, loss=None):
    if fixing is None:
        fixing = ''
    else:
        fixing = f'_{fixing}'
    if tail_info is None:
        tail_info = ''
    else:
        tail_info = f'_{tail_info}'
    save_name = f'{model.name}_d{model.depth}_n{model.nodes}_s{model.seed}\
{fixing}{tail_info}'

    needen_info = {'depth': model.depth, 'nodes': model.nodes,
                   'seed': model.seed, 'name': model.name}
    eqx.tree_serialise_leaves(f'{path}/{save_name}.eqx', model)
    with open(f"{path}/{save_name}.json", "w") as f:
        json.dump(needen_info, f)
    print(f'Saved {path}/{save_name}.eqx')
    
    if loss is not None:
        with open(f"{path}/{save_name}_loss.txt", "w") as f:
            np.savetxt(f, loss)
            print(f'Saved the loss values in {path}/{save_name}_loss.txt')


def load_eqx_model(path: str):
    jax.config.update("jax_enable_x64", True)  # Ensure 64-bit is enabled first

    with open(f"{path}.json", "r") as f:
        metadata = json.load(f)

    # Model selection
    name = metadata['name']
    # Be robust to accidental whitespace/newlines in saved metadata
    if isinstance(name, str):
        name = name.strip()
    # Handle common xcdiff naming variants
    if name in ("GGA_Fx_xcdiff", "GGA_Fx_xcdiff "):
        name = "GGA_Fx_xcdiff"
    if name in ("GGA_Fc_xcdiff", "GGA_Fc_xcdiff "):
        name = "GGA_Fc_xcdiff"
    if name in ("GGA_Fx_xcdiff_unct", "GGA_Fx_xcdiff_unct "):
        name = "GGA_Fx_xcdiff_unct"
    if name in ("GGA_Fc_xcdiff_unct", "GGA_Fc_xcdiff_unct "):
        name = "GGA_Fc_xcdiff_unct"
    Model_Object = {
        'Fc_s': FcNet_simple,
        'Fc_LOB': FcNet_LOB,
        'FcG_LOB': FcNetG_LOB,
        'FcG_s': FcNetG_simple,
        'Fx_s': FxNet_simple,
        'Fx_LOB': FxNet_LOB,
        'FxG_LOB': FxNetG_LOB,
        'FxG_s': FxNetG_simple,
        'FxSIG_LOB': FxNetSIG_LOB,
        'FcSIG_LOB': FcNetSIG_LOB,
        'FcSIGt_LOB': FcNetSIGt_LOB,
        'GGA_Fx_G_transf': GGA_Fx_G_transf,
        'GGA_Fc_G_transf': GGA_Fc_G_transf,
        'GGA_Fx_G_transf_lin': GGA_Fx_G_transf_lin,
        'GGA_Fc_G_transf_lin': GGA_Fc_G_transf_lin,
        'GGA_Fx_xcdiff': GGA_Fx_xcdiff,
        'GGA_Fc_xcdiff': GGA_Fc_xcdiff,
        'GGA_Fx_xcdiff_unct': GGA_Fx_xcdiff_unct,
        'GGA_Fc_xcdiff_unct': GGA_Fc_xcdiff_unct,
    }.get(name)
    if Model_Object is None:
        raise ValueError(f"Model {name} not recognized. Please check the \
model name.")

    dummy_model = Model_Object(depth=metadata["depth"],
                               nodes=metadata["nodes"],
                               seed=metadata["seed"])

    # Load the saved model into the dummy structure
    model = eqx.tree_deserialise_leaves(f"{path}.eqx", like=dummy_model)
    print(f'Loaded {path}.eqx')
    return model


class RXCModel(eqx.Module):
    xnet: eqx.Module
    cnet: eqx.Module

    def __init__(self, xnet, cnet):
        self.xnet = xnet
        self.cnet = cnet

    def __call__(self, inputs):
        #this generate epsilon, not exc -- divide end result by rho when needed
        rho = inputs[0]
        sigma = inputs[1]
        # print('RXCModel call - inputs {}'.format(inputs))
        return rho*(lda_x(rho)*jax.vmap(self.xnet)(inputs[..., jnp.newaxis]) + pw92c_unpolarized(rho)*jax.vmap(self.cnet)(inputs[..., jnp.newaxis])).flatten()[0]