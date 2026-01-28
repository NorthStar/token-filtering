import numpy as np
import tiktoken
import matplotlib.pyplot as plt
import json
import pandas as pd
from transformers import AutoTokenizer

enc = tiktoken.get_encoding("cl100k_base")
# enc = AutoTokenizer.from_pretrained('google/gemma-2-9b')

tokens = np.memmap('../../../../../workspace/med/data/probe/token/tokens.bin', dtype=np.uint32, mode='r')
labels = np.memmap('../../../../../workspace/med/data/probe/token/labels.bin', dtype=bool, mode='r')

print(tokens.shape)
print(labels.shape)
print(labels.sum())

max_value = labels.max()
cmap = plt.get_cmap('Greens')

def get_color(value):
    if value == 1:
        return f'rgba(255, 97, 97, 0.75)'
    else:
        return f'rgba(255, 255, 255, 0.75)'

html_content = f"""
<html>
<head>
<meta charset="utf-8">
<title>annotated</title>
</head>
<body style="font-family:Jet Brains Mono, monospace; font-size:14px; white-space:pre-wrap;">
"""

for i in range(0, 10000):
    
    html_content += f'<span style="background-color: {get_color(labels[i])}">{enc.decode([tokens[i]]).replace("<", "&lt;").replace(">", "&gt;")}</span>'

html_content += "</body>\n</html>"

with open('annotated.html', 'w') as f:
    f.write(html_content)