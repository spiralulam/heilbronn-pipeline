"""Search pipeline for the Heilbronn triangle problem in the unit square ('s'), the unit right triangle ('t') and in
convex position ('c').  See README.md for the overview and run.py for the driver."""
from .geometry import VARIANTS, load, save, value
from .polish import slp
from .refine import refine
from .store import Store

__all__ = ['VARIANTS', 'load', 'save', 'value', 'slp', 'refine', 'Store']
