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


class InheritFromElements(object):
    """Inherit attributes from the elements within self.

    This is meant for container objects like `DNA`, `DNAs`, `Particles`, etc.

    Parameters:
    -----------
    elements_list: container object
        The container containing the subelements. If None, will assume it is `self`.

    """

    def __init__(self, elements_list=None):
        """Prepare the object."""
        self._elements_attribute = elements_list if elements_list is not None else self
        raise NotImplementedError


class InheritFromGroup(object):
    """Inherit attributes from a parent class, usually a group that contains the object.

    Parameters:
    -----------
    group: object
        The object from which to inherit attributes

    """

    def __init__(self, group):
        """Prepare the instance."""
        self._group = group

    def __getattr__(self, attr):
        """Look for attributes in the group object.

        This screens out methods on purpose to avoid big confusion.

        """
        ret = getattr(self._group, attr)
        if ret is not None and not isinstance(ret, type(self.__init__)):
            return getattr(self._group, attr)
        else:
            return self.__getattribute__(attr)
