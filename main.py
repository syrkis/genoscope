# %% main.py
#   genoscope main file
# by: Aske Brunken

# Imports
import jax
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
def palin_fn(seq, n):  # return all pallindromes if length n in seq
    return seq, seq[::-1]


palin_fn(encode(x)[:20], n=3)
