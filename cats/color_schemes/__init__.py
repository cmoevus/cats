"""Color schemes for Cats."""
from __future__ import absolute_import, division, print_function
import importlib
import cats.options

default = importlib.import_module('cats.color_schemes.' + cats.options.color_scheme).colors
