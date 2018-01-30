# -*- coding: utf8 -*-
"""Classes and functions to deal with particles and tracking.

for particle in particles:
    type(particle)
>> cats.Particle

"""
from __future__ import absolute_import, division, print_function
import pandas as pd
import inspect

from cats import extensions


@extensions.append
class Particles(pd.DataFrame, object):
    """Container for particles.

    One can access individual particles by iterating over the object, which returns Particle objects, or by using the `Particles.groupby('particle')` method, which returns the DataFrame grouped by particle id.

    Arguments
    _________
    source: cats.Images object
        The source of the data. None if undefined.

    Notes on implementation
    ------------------------
    Particles objects subclass pandas DataFrame objects and share all of their properties. However, when indexing with numerical indexes, the object returns Particle objects. For example:
        p = Particles(detection_df)
        type(p[0])
        >>> cats.particles.Particle
    As such, one should avoid giving numerical names to columns.

    In addition, iterating over the object returns `Particle` objects.

    `Particle` objects will be given the attributes listed in `Particles._element_attributes`. One can set a specific attribute by adding it to the `_element_attributes` set and creating an attribute with the same name in the `Particles` object that contains a dictionary of pairs (particle_number, value)

    Unlike `pandas.DataFrame`, `Particles` object do not allow setting new columns


    """

    _internal_names = pd.DataFrame._internal_names + ['_iter']
    _internal_names_set = set(_internal_names)

    @property
    def _constructor(self):
        """Return a Particles object when manipulating."""
        return Particles

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        self._element_attributes = set()
        self._metadata = ['_element_attributes']

        # # Get metadata from incoming DataFrame, THIS DOES NOT COPY, BUT LINK
        # if len(args) > 0 and issubclass(args[0].__class__, pd.DataFrame):
        #     for m in args[0]._metadata:
        #         setattr(self, m, getattr(args[0], m))
        #         if m not in self._metadata:
        #             self._metadata.append(m)

        # Deal with element attributes
        expected_attributes = set(['source', 'tracking_parameters', 'filtering_parameters'])
        user_attributes = kwargs.pop('_element_attributes', [])
        _element_attributes = expected_attributes.union(user_attributes)
        for attr in _element_attributes:
            if attr in kwargs:
                self.set_element_attribute(attr, kwargs.pop(attr))

        # Initialize DataFrame
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        """Return the particle 'i' if i is an int else item i from the dataframe."""
        if type(i) is int:
            try:
                return self._package_element(self[self['particle'] == i])
            except KeyError:
                raise IndexError("Particle not found.") from None
        return super().__getitem__(i)

    def __setattr__(self, attr, value):
        """Bypass `pandas` annoying warnings when setting attributes."""
        object.__setattr__(self, attr, value)

    def __getattr__(self, attr):
        """Use methods and attributes from the underlying Particle objects."""
        # Let pandas do the work for column names
        try:
            return super().__getattr__(attr)
        # Get attribute from the underlying objects
        except AttributeError:
            if hasattr(Particle, attr):
                if type(getattr(Particle, attr)) == property:
                    return [getattr(particle, attr) for particle in self]
                else:
                    def all_attr(*args, **kwargs):
                        return [getattr(particle, attr)(*args, **kwargs) for particle in self]
                    return all_attr
            elif any([hasattr(v, attr) for v in self]):
                return [getattr(particle, attr, None) for particle in self]
        # Handle properties
        return super().__getattribute__(attr)

    def __iter__(self):
        """Iterate over particles rather than columns."""
        #
        # I feel like this is going to lead to big, hard to debug problems...
        #
        caller = inspect.currentframe().f_back.f_code.co_name  # This is to avoid breaking the __repr__ function. There has to be a cleaner way to do this, but I don't want to rewrite __repr__ :)
        if caller == '_to_str_columns':  # This is the function __repr__ calls to get the column names
            return super().__iter__()
        else:
            try:
                self._iter = self.groupby('particle').__iter__()
            except AttributeError:
                raise AttributeError("Particles not found. Make sure you track the features and link them together first.") from None
            return self

    def __next__(self):
        """Continue iteration."""
        return self._package_element(self._iter.__next__()[1])

    def _set_element_attribute(self, attr, value, element=None):
        """Set the given attribute value to the elements in `element`."""
        if element is None:
            element = self
        values = {i: value for i in element['particle'].unique()}
        self._element_attributes.add(attr)
        if attr not in self._metadata:
            self._metadata.append(attr)
        if hasattr(self, attr):
            getattr(self, attr).update(values)
        else:
            setattr(self, attr, values)

    def _update_element_attribute(self, attr, value, element=None):
        """Update the given dictionary `attr` with dictionary `value` for all particles in element."""
        if element is None:
            element = self
        try:
            attr = getattr(self, attr)
            for i in element['particle'].unique():
                attr[i].update(value)
        except AttributeError:
            self._set_element_attribute(attr, value, element)

    def _get_element_attribute(self, attr, element=None):
        """Return the given attribute value for all the particles in the DataFrame."""
        if element is None:
            element = self
        values = {i: getattr(self, attr)[i] for i in element['particle'].unique()}
        if len(values) == 1:
            return tuple(values.values())[0]

    def _package_element(self, element):
        """Return an independent Particle object."""
        e = Particle(element)
        e._group = self
        return e

    def copy(self):
        """Copy the DataFrame as well as its metadata."""
        copy = super().copy()
        for m in self._metadata:
            if hasattr(self, m):
                attr = getattr(self, m)
                if hasattr(attr, 'copy'):
                    attr = attr.copy()
                setattr(copy, m, attr)
        copy._metadata = self._metadata.copy()
        return copy

    def renumber_particles(self, start=0, copy=True):
        """Reorganise the `particle` column so that particle numbers follow each other by increments of 1 starting at `start`.

        Parameters:
        ----------
        start: int
            The value to start at
        copy: bool
            Whether to copy the object before rearranging or rearrange in place.

        """
        df = self if copy is False else self.copy()
        numbers = df['particle'].unique()
        new_numbers = range(start, start + len(numbers))
        for n, nn in zip(numbers, new_numbers):
            df.loc[df['particle'] == n, 'particle'] = nn
            for attr in self._element_attributes:
                attr = getattr(df, attr)
                attr[nn] = attr.pop(n)
        return df

    @property
    def number(self):
        """The number of particles."""
        return len(self.groupby('particle'))


@extensions.append
class Particle(pd.DataFrame, object):
    """Container for one particle.

    Notes on implementation
    -----------------------

    """

    @property
    def _constructor(self):
        """Return a Particle object when manipulating."""
        return Particle

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        super().__init__(*args, **kwargs)

    def __getattr__(self, attr):
        """Look for attributes in group."""
        if "_group" in self.__dict__ and attr in self._group._element_attributes:
            return self._group._get_element_attribute(attr, self)
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        """Write attributes from the group in the group."""
        if hasattr(self, "_group") and attr in self._group._element_attributes:
            self._group._set_element_attribute(attr, value, self)
        else:
            object.__setattr__(self, attr, value)
