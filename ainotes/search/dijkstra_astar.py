# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] slideshow={"slide_type": "slide"}
# # Dijkstra's and A* algorithms
#
# ```{warning}
# This chapter is under construction.
# ```

# %% [markdown] slideshow={"slide_type": "slide"}
# ## Environment setup

# %%
import platform
from IPython.display import YouTubeVideo

print(f"Python version: {platform.python_version()}")

# %% [markdown] slideshow={"slide_type": "slide"}
# ## A* on a real map
#
# Uses the streets of Chicago and Rome. Intersections of streets are represented as nodes and streets as edges.
#
# Another cool resource on the same topic: <https://github.com/honzaap/Pathfinding>

# %%
YouTubeVideo("CgW0HPHqFE8")

# %%
