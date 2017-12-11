# -*- coding: utf8 -*-
"""Classes and functions to deal with particles and tracking."""
from __future__ import absolute_import, division, print_function
import pandas as pd


class Particles(object):
    """Container for particles.

    Note on implementation
    ----------------------
    Subclassing vs composition?
    Simple composition:
        class dfc(object):
             def __init__(self, df):
                self.df = df

             def __getattr__(self, attr):
                return getattr(self.df, attr)
    Subclassing: cleaner but more complex (= more bugs)

    """
