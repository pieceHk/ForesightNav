"""Resolve the separately prepared SuperGlue inference dependency."""

from runtime.external_sources import superglue_directory

__path__ = [str(superglue_directory())]
