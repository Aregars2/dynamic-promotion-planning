"""Compatibility namespace for artifacts created before package renaming.

Active code must import :mod:`dynamic_promotion_planning`.  This namespace is
retained only because existing pickle artifacts encode the historical module path.
"""

from dynamic_promotion_planning import *  # noqa: F401,F403
