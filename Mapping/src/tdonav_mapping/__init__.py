"""Compatibility access to the live mapping implementation."""

from pbsp.mapping.types import CameraObservation

__all__ = ["CameraObservation", "OnlineMapper"]


def __getattr__(name):
    if name == "OnlineMapper":
        from pbsp.mapping.mapper import OnlineMapper

        return OnlineMapper
    raise AttributeError(name)
