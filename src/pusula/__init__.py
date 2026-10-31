"""pusula: a directional systematic equity & options trading platform.

Layering (enforced by ``tests/test_architecture.py``; a module may import only from its own
layer or a lower one)::

    core -> stats -> data -> alpha -> risk -> execution -> portfolio -> ml
         -> engine -> options -> research -> live -> web -> cli
"""

__version__ = "0.1.0"
