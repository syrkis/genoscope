# %% main.py
#   genoscope main file
# by: Aske Brunken

# Imports
import jax
from jax import lax
import jax.numpy as jnp
import requests as req
import matplotlib.pyplot as plt

import esch

CODONS_LINK = "http://csg.sph.umich.edu/zhanxw/software/anno/codon.txt"
codons = req.get(CODONS_LINK).text.lower()
b2a = {row.split("\t")[0]: row.split("\t")[1] for row in codons.strip().split("\n")[2:]}

# %% TODO:
# 1. find base pair dataset.
# 2. "compile" compile basepais to codons
with open("data.txt", "r") as f:
    x = "".join(
        [
            "".join([char for char in row if char in ["a", "c", "t", "g"]])
            for row in f.read().strip().split("ORIGIN")[1].split("\n")
        ]
    )

# %%
n2i = {n: i for i, n in enumerate(list(set(x)))}
i2n = {i: n for i, n in enumerate(list(set(x)))}
encode = lambda x: jnp.array([n2i[n] for n in x])
decode = lambda x: "".join([i2n[i] for i in x.tolist()])


# %%
def palin_fn(seq, n=3):  # return all pallindromes if length n in seq
    aux = jnp.arange(n // 2) + 1
    kernel = jnp.concat((aux, jnp.zeros(n % 2), -aux[::-1]))[::-1]
    esch.tile(kernel[None, :])
    return (jnp.convolve(seq, kernel, mode="same") == 0).astype(int)


palin_fn(encode(x), 10).sum() / encode(x).size
