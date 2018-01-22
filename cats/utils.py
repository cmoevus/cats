"""Make objects saveable/loadable using pickle."""
from __future__ import absolute_import, division, print_function
import pickle


class pickle_save(object):
    """Save/Load an instance using pickle."""

    def save(self, f):
        """Save the instance to the given file."""
        with open(f, 'wb') as d:
            pickle.dump(self, d)

    @staticmethod
    def load(f):
        """Load an instance from the given file."""
        with open(f, 'rb') as d:
            i = pickle.load(d)
        return i
