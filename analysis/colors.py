"""
Centralized color palette configuration for all analysis plots.

Color Schemes:
- MASK_COLORS: Discrete colors for mask types (Ohchi palette + extras)
- MODEL_SIZE_CMAP: GnBu discrete for model sizes
- THRESHOLD_CMAP: SunsetDark discrete for threshold sweep
- NOISE_CMAP: Green-Gold for noise sweep frontiers

Usage:
    from colors import MASK_COLORS, THEME_COLORS, get_model_size_colors, get_threshold_colors, get_noise_colors
"""

from pypalettes import load_cmap

# =============================================================================
# Mask Type Colors (Ohchi palette + custom colors)
# =============================================================================

# Order: remove, mask, document, nomask, unlearn, token-with-doc-labels, token-with-sent-labels
# Colors: remove=1st, mask=2nd, document=3rd, nomask=4th, unlearn=5th
MASK_ORDER = ['remove', 'mask', 'document', 'nomask', 'unlearn', 'token-with-doc-labels', 'token-with-sent-labels']

# Load Ohchi palette and extract colors
_ohchi_cmap = load_cmap("Ohchi")
_ohchi_colors = ['#%02x%02x%02x' % tuple(int(c*255) for c in _ohchi_cmap(i/5)[:3]) for i in range(5)]

# Combine Ohchi colors with custom additions
# token-with-doc-labels: #D2E4C4 (for Document in label sweep)
# token-with-sent-labels: #B5CBB7 (for Token in label sweep)
_all_mask_colors = _ohchi_colors + ["#B5CBB7", "#CC978E"]

# Create the mask color mapping
MASK_COLORS = {mask: color for mask, color in zip(MASK_ORDER, _all_mask_colors)}

# Add 'token' as an alias for 'mask' (some files use 'token' as the display value)
MASK_COLORS['token'] = MASK_COLORS['mask']
# Add 'document_document' as an alias for 'document' (used in label-sweep-scaling.py)
MASK_COLORS['document_document'] = MASK_COLORS['document']

# Labels for mask types (for legends)
MASK_LABELS = {
    'nomask': 'Baseline',
    'document': 'Document',
    'mask': 'Token (Masking)',
    'remove': 'Token (Removal)',
    'unlearn': 'Unlearning',
    'token-with-doc-labels': 'Token (Doc Labels)',
    'token-with-sent-labels': 'Token (Sent Labels)',
}

def get_mask_colors(mask_types):
    """
    Get colors for a list of mask types in consistent order.
    
    Args:
        mask_types: list of mask type strings
    
    Returns:
        dict mapping mask type to hex color
    """
    return {m: MASK_COLORS.get(m, '#888888') for m in mask_types}

def get_mask_color_list(mask_types):
    """
    Get list of colors for mask types in the order provided.
    
    Args:
        mask_types: list of mask type strings
    
    Returns:
        list of hex colors
    """
    return [MASK_COLORS.get(m, '#888888') for m in mask_types]

# =============================================================================
# Model Size Colors (GnBu discrete)
# =============================================================================

# Standard model sizes (in millions of params) - superset for consistent mapping
MODEL_SIZE_ORDER = ['13M', '28M', '61M', '113M', '224M', '521M', '1030M']

# Load GnBu and create discrete colors (last model = last color)
_gnbu_cmap = load_cmap("GnBu")
_n_sizes = len(MODEL_SIZE_ORDER)
MODEL_SIZE_COLORS = {
    size: '#%02x%02x%02x' % tuple(int(c*255) for c in _gnbu_cmap(i/(_n_sizes-1))[:3])
    for i, size in enumerate(MODEL_SIZE_ORDER)
}

def get_model_size_colors(sizes):
    """
    Get colors for model sizes using GnBu palette.

    Args:
        sizes: list of size strings (e.g., ['61M', '113M', '224M'])

    Returns:
        dict mapping size to hex color
    """
    return {s: MODEL_SIZE_COLORS.get(s, '#888888') for s in sizes}

def get_model_size_color_list(sizes):
    """
    Get list of colors for model sizes in the order provided.
    
    Args:
        sizes: list of size strings
    
    Returns:
        list of hex colors
    """
    return [MODEL_SIZE_COLORS.get(s, '#888888') for s in sizes]

# =============================================================================
# Threshold Sweep Colors (SunsetDark discrete)
# =============================================================================

def get_threshold_colors(n_categories):
    """
    Get colors for threshold sweep using SunsetDark palette (discrete).

    Args:
        n_categories: number of threshold levels

    Returns:
        list of hex colors
    """
    sunset_cmap = load_cmap("SunsetDark")
    return [
        '#%02x%02x%02x' % tuple(int(c*255) for c in sunset_cmap(1 - i/(n_categories-1) if n_categories > 1 else 0)[:3])
        for i in range(n_categories)
    ]

# =============================================================================
# Noise Sweep Colors (Green-Gold for frontiers)
# =============================================================================

def get_noise_colors(n_categories):
    """
    Get colors for noise sweep frontiers using Green-Gold palette.
    
    Args:
        n_categories: number of noise levels
    
    Returns:
        list of hex colors
    """
    green_gold_cmap = load_cmap("Green-Gold")
    return [
        '#%02x%02x%02x' % tuple(int(c*255) for c in green_gold_cmap(i/(n_categories-1) if n_categories > 1 else 0)[:3])
        for i in range(n_categories)
    ]

# =============================================================================
# Probe/Classifier Colors (GnBu discrete, aligned with model sizes)
# =============================================================================

# Probe labels in order - should align with model size indices where applicable
PROBE_ORDER = [
    'No Masking',      # baseline (like nomask)
    'ModernBERT',      # external
    'RoBERTa',         # external
    'edu-61M',         # 61M
    'biLM-61M',        # 61M
    'biLM-113M',       # 113M
    'biLM-224M',       # 224M
]

# Map probes to consistent colors using GnBu (last probe = last color)
_n_probes = len(PROBE_ORDER)
PROBE_COLORS = {
    probe: '#%02x%02x%02x' % tuple(int(c*255) for c in _gnbu_cmap(i/(_n_probes-1))[:3])
    for i, probe in enumerate(PROBE_ORDER)
}
# Add 'No Filtering' as an alias using the nomask color for consistency
PROBE_COLORS['No Filtering'] = MASK_COLORS['nomask']
PROBE_COLORS['No Masking'] = MASK_COLORS['nomask']

def get_probe_colors(probes):
    """
    Get colors for probes/classifiers using GnBu palette.

    Args:
        probes: list of probe label strings

    Returns:
        dict mapping probe to hex color
    """
    return {p: PROBE_COLORS.get(p, '#888888') for p in probes}

def get_probe_color_list(probes):
    """
    Get list of colors for probes in the order provided.
    
    Args:
        probes: list of probe label strings
    
    Returns:
        list of hex colors
    """
    return [PROBE_COLORS.get(p, '#888888') for p in probes]

# =============================================================================
# Delayed Filter Colors (monet discrete)
# =============================================================================

def get_delayed_colors(n_categories):
    """
    Get colors for delayed filter sweep using monet palette.

    Args:
        n_categories: number of delay levels

    Returns:
        list of hex colors
    """
    monet_cmap = load_cmap("monet")
    return [
        '#%02x%02x%02x' % tuple(int(c*255) for c in monet_cmap(i/(n_categories-1) if n_categories > 1 else 0)[:3])
        for i in range(n_categories)
    ]

# =============================================================================
# Theme Configuration (light mode)
# =============================================================================

# Standard theme colors (light mode)
THEME_COLORS = {
    'bg_color': '#ffffff',
    'text_color': '#000000',
    'line_color': '#000000',
    'grid_color': '#dddddd',
}
