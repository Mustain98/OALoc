"""Second fixture module, so traversal and ranking have more than one file."""
from src.cache import Cache


def lookup(key):
    return Cache().get(key)


def prefetch(keys):
    return [lookup(k) for k in keys]
