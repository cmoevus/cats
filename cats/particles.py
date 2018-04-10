# -*- coding: utf8 -*-
"""Classes and functions to deal with particles and tracking.

for particle in particles:
    type(particle)
>> cats.Particle

"""
from __future__ import absolute_import, division, print_function
import pandas as pd
import inspect
import copy

from cats import extensions


@extensions.append
class Particles(pd.DataFrame, object):
    """Container for particles.

    One can access individual particles by iterating over the object, which returns Particle objects, or by using the `Particles.groupby('particle')` method, which returns the DataFrame grouped by particle id.

    Notes on implementation
    ------------------------
    Particles objects subclass pandas DataFrame objects and share all of their properties. However, when indexing with numerical indexes, the object returns Particle objects. For example:
        p = Particles(detection_df)
        type(p[0])
        >>> cats.particles.Particle
    As such, one should avoid giving numerical names to columns.

    In addition, iterating over the object returns `Particle` objects.

    `Particle` objects will be given the attributes listed in `Particles._element_attributes`. One can set a specific attribute by adding it to the `_element_attributes` dict as a dictionary of pairs (particle_number, value)

    Unlike `pandas.DataFrame`, `Particles` object do not allow setting new columns


    """

    _internal_names = pd.DataFrame._internal_names + ['_iter']
    _internal_names_set = set(_internal_names)
    _metadata = ['_element_attributes']

    @property
    def _constructor(self):
        """Return a Particles object when manipulating."""
        return Particles

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        self._element_attributes = dict()

        # Remove element attributes from list so that they don't bother the DataFrame initiation
        expected_attributes = set(['source', 'tracking_parameters', 'filtering_parameters'])
        user_attributes = kwargs.pop('_element_attributes', [])
        element_attributes = dict()
        for attr in expected_attributes.union(user_attributes):
            if attr in kwargs:
                element_attributes[attr] = kwargs.pop(attr)

        # Initialize DataFrame
        super().__init__(*args, **kwargs)

        # Deal with element attributes
        for attr, value in element_attributes.items():
            self._set_element_attribute(attr, value)

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
        # Return a column
        if attr in self.columns:
            return self[attr]

        # Get attribute from the underlying objects
        if 'particle' in self.columns:
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
        elif len(self) == 0 or 'particle' not in self.columns:
            return [].__iter__()
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
        if 'particle' in self.columns:
            if element is None:
                element = self
            values = {i: value for i in element['particle'].unique()}
            self._set_elements_attribute(attr, values)
        #
        # Note that elements attributes are not recorded if no particle exists, and the user doesn't know. This is terrible behavior and needs to be updated asap.
        #

    def _set_elements_attribute(self, attr, values):
        """Set the given attribute to the given values for the given Particle objects.

        Parameters:
        -----------
        attr: string
            The name of the attribute
        values: dict
            The mapping particle_number: value for the attribute.

        """
        if attr not in self._element_attributes:
            self._element_attributes[attr] = values
        else:
            self._element_attributes[attr].update(values)

    def _update_element_attribute(self, attr, value, element=None):
        """Update the given element attribute `attr` (which has to be a dictionary) with dictionary `value` for all particles in element."""
        if element is None:
            element = self
        try:
            attrib = self._element_attributes[attr]
            for i in element['particle'].unique():
                attrib[i].update(value)
        except KeyError:
            self._set_element_attribute(attr, value, element)

    def _get_element_attribute(self, attr, element=None):
        """Return the given attribute value for all the particles in the DataFrame."""
        if element is None:
            element = self
        values = {i: self._element_attributes[attr][i] for i in element['particle'].unique()}
        if len(values) == 1:
            return tuple(values.values())[0]

    def _package_element(self, element):
        """Return an independent Particle object."""
        e = Particle(element)
        e._group = self
        return e

    def copy(self, deep=True):
        """Copy the DataFrame as well as its metadata.

        Parameters:
        -----------
        copy: bool
            Whether to perform a deep copy.

        """
        if deep:
            copy_func = copy.deepcopy
        else:
            copy_func = copy.copy
        new_self = super().copy()
        for m in self._metadata:
            setattr(new_self, m, copy_func(getattr(self, m)))
        new_self._metadata = self._metadata.copy()
        return new_self

    def renumber_particles(self, start=0, copy=True):
        """Reorganise the `particle` column so that particle numbers follow each other by increments of 1 starting at `start`.

        Parameters:
        ----------
        start: int
            The value to start at
        copy: bool
            Whether to copy the object before rearranging or rearrange in place.

        Note:
        -----
        This method also cleans up unused element attributes

        """
        df = self if copy is False else self.copy()
        if 'particle' not in df.columns:
            return df
        numbers = df['particle'].unique()
        new_numbers = range(start, start + len(numbers))
        new_element_attrs = {attr: dict() for attr in df._element_attributes}
        for n, nn in zip(numbers, new_numbers):
            df.loc[df['particle'] == n, 'particle'] = nn
            for attr, values in df._element_attributes.items():
                new_element_attrs[attr][nn] = values[n]
        df._element_attributes = new_element_attrs
        return df

    @property
    def number(self):
        """The number of particles."""
        if 'particle' in self.columns:
            return len(self.groupby('particle'))
        else:
            return 0

    def append(self, particles):
        """Append another Particle(s) object to this one and returns it as a new object."""
        if not isinstance(particles, Particle) and not isinstance(particles, Particles):
            raise ValueError('Can only append Particles or Particle objects.')

        if particles.number == 0:
            return self.copy()

        if 'particle' not in particles.columns:
            return super().append(particles, ignore_index=True)

        if 'particle' not in self.columns:
            if len(self) == 0:
                return particles.copy().renumber_particles()
            else:
                raise ValueError('Particles are not defined in the appending Particles object.')

        m = self['particle'].max()
        start = int(m + 1) if m > 0 else 0
        particles = particles.renumber_particles(start)
        new_self = super().append(particles, ignore_index=True)
        for attr in self._element_attributes:
            new_self._set_elements_attribute(attr, self._element_attributes[attr].copy())
        for attr in particles._element_attributes:
            new_self._set_elements_attribute(attr, particles._element_attributes[attr].copy())
        return new_self


@extensions.append
class Particle(pd.DataFrame, object):
    """Container for one particle.

    Notes on implementation
    -----------------------

    """

    _metadata = ['_group']

    @property
    def _constructor(self):
        """Return a Particle object when manipulating."""
        return Particle

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        super().__init__(*args, **kwargs)
        self._group = None

    def __getattr__(self, attr):
        """Look for attributes in group."""
        if self._group is not None and attr in self._group._element_attributes:
            return self._group._get_element_attribute(attr, self)
        return super().__getattr__(attr)

    def __setattr__(self, attr, value):
        """Write attributes from the group in the group."""
        if attr != '_group' and self._group is not None and attr in self._group._element_attributes:
            self._group._set_element_attribute(attr, value, self)
        else:
            object.__setattr__(self, attr, value)

    def copy(self, deep=False):
        """Copy the DataFrame as well as its metadata.

        Parameters:
        -----------
        copy: bool
            Whether to perform a deep copy. Note that this will most likely lead to an error as pandas dataframes do not support deep copies.

        """
        if not deep:
            copy_func = copy.copy
        else:
            copy_func = copy.deepcopy
        new_self = super().copy()
        for m in self._metadata:
            setattr(new_self, m, copy_func(getattr(self, m)))
        new_self._metadata = self._metadata.copy()
        return new_self
