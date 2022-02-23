# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import docsim
import os

docsim_obj = docsim.DocSim(verbose=True)

# Dir containing .md files
vault_path = (
    "/Users/jacksong/Library/Mobile Documents/iCloud~md~obsidian/Documents/Incredex"
)
periodics_path = vault_path + "/Periodics"
md_file_paths = []  # empty list to add .md files to

for file_name in os.listdir(vault_path):
    if file_name.endswith(".md"):
        md_file_paths.append(vault_path + "/" + file_name)
# for file_name in os.listdir(periodics_path):
#     if file_name.endswith(".md"):
#         md_file_paths.append(periodics_path + "/" + file_name)

fillers = [
    "korean / g",
    "NLP ML /",
    "## Retro",
    "## Plan",
    "## Ideas",
    "## Focus",
    "Order of oper",
    "Think about: G",
    "sketching / pa",
    "art / languages",
    "## Today",
    "## Yesterday",
    "## Idea Eng",
    "## Weekly Pos",
    "Apps are just not that important",
]


def no_filler(line):
    for filler in fillers:
        if filler in line:
            return False
    return True


documents = []
for path in md_file_paths:
    with open(path) as in_file:
        # documents = documents + in_file.readlines()
        # in_file.read()
        lines = in_file.readlines()
        contentLines = [line for line in lines if no_filler(line)]
        documents.append(" ".join(contentLines))

print(f"docs imported: {len(documents)}")

filtered_documents = [
    document
    for document in documents
    if len(document) > 60
    and len(document) < 1000
    and len(docsim_obj.preprocess(document)) > 7
    and len(docsim_obj.preprocess(document)) < 150
]

print(f"docs pushed in: {len(filtered_documents)}")

results = docsim_obj.top_pairs(filtered_documents)
for idx1, idx2, score in results:
    print(
        f"{score:0.3f} \n A: {filtered_documents[idx1]} \n B: {filtered_documents[idx2]} \n"
    )
