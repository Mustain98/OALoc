"""Tiny fixture module used by the Farhan/3 stubs and the offline tests."""


class Cache:
    def __init__(self):
        self._store = {}

    def get(self, k):
        # The seeded bug: raises KeyError instead of returning None.
        return self._store[k]

    def put(self, k, v):
        self._store[k] = v


def helper():
    c = Cache()
    return c.get("x")
