from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np

def get_method_colors(METHOD_NAME_2_GROUP_ID):
    # 1. Pick base colors:
    cmap = plt.cm.get_cmap('Set1')  # for groups 0,1,2
    group_base_colors = {}
    for gid in set(METHOD_NAME_2_GROUP_ID.values()):
        if gid == 6:
            group_base_colors[gid] = (0.6, 0.6, 0.6, 1.0)
            # group_base_colors[gid] = (1.0, 0.3, 0.3, 1.0)
        else:
            # qualitative hues for groups 0,1,2
            group_base_colors[gid] = cmap(gid % cmap.N)

    # 2. Helper to mix with white for tints
    def tint_color(color, mix):
        r, g, b, a = color
        return (
            r + (1 - r) * mix,
            g + (1 - g) * mix,
            b + (1 - b) * mix,
            a
        )

    # 3. Group methods together
    group_to_methods = defaultdict(list)
    for method, gid in METHOD_NAME_2_GROUP_ID.items():
        group_to_methods[gid].append(method)

    # 4. Build the method_to_color dict with varying tints
    method_to_color = {}
    for gid, methods in group_to_methods.items():
        base = group_base_colors[gid]
        if gid == 0:
            mixes = np.linspace(0, 0.3, len(methods))  # tint less
        else:
            mixes = np.linspace(0, 0.6, len(methods))  # from base to lighter
            
        for method, mix in zip(methods, mixes):
            method_to_color[method] = tint_color(base, mix)
            
    return method_to_color
