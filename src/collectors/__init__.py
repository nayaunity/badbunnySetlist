"""Data collectors for setlist information."""
from .bad_bunny import collect_bad_bunny_setlists
from .superbowl import collect_superbowl_halftime_setlists
from .musicbrainz import collect_bad_bunny_catalog

__all__ = [
    "collect_bad_bunny_setlists",
    "collect_superbowl_halftime_setlists",
    "collect_bad_bunny_catalog"
]
