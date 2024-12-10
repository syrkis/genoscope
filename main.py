# %% main.py
#   genoscope main file
# by: Aske Brunken

# Imports
from collections import _KT
import jax
from jax import lax
import jax.numpy as jnp
import requests as req
import matplotlib.pyplot as plt


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
n2i = {n: i for i, n in enumerate(list(set(nucleotides)))}
i2n = {i: n for i, n in enumerate(list(set(nucleotides)))}
encode = lambda x: jnp.array([n2i[n] for n in x])
decode = lambda x: "".join([i2n[i] for i in x.tolist()])


# %%
def palin_fn(seq, n=3):  # return all pallindromes if length n in seq
    aux = jnp.arange(n // 2) + 1
    kernel = jnp.concat((aux, jnp.zeros(n % 2), -aux[::-1]))[::-1]
    return seq, (jnp.convolve(seq, kernel, mode="same") == 0).astype(int), kernel


palin_fn(jnp.array([1, 2, 3, 2, 1, 0, 1, 2]), 5)
