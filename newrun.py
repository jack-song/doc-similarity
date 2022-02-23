# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

# built-in libs
import os
from pathlib import Path

# obsidiantools requirements
import numpy as np
import pandas as pd
import networkx as nx


VAULT_DIR = Path(
    "/Users/jacksong/Library/Mobile Documents/iCloud~md~obsidian/Documents/Incredex"
)

import obsidiantools.api as otools  # api shorthand

vault = otools.Vault(VAULT_DIR).connect().gather()

print(f"Connected?: {vault.is_connected}")
print(f"Gathered?:  {vault.is_gathered}")

import docsim


docsim_obj = docsim.DocSim(verbose=True)


def get_full_text(name):
    return name + ". " + vault.get_text(name)


documents = {name: get_full_text(name) for name in vault.file_index.keys()}

print(docsim_obj.top_pairs(documents))
print()
