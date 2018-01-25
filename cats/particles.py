# -*- coding: utf8 -*-
"""Classes and functions to deal with particles and tracking.

for particle in particles:
    type(particle)
>> cats.Particle

"""
from __future__ import absolute_import, division, print_function
import pandas as pd
import inspect

from . import extensions
import cats.dna


@extensions.append
class Particles(pd.DataFrame):
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

    """

    _internal_names = pd.DataFrame._internal_names + ['_iter']
    _internal_names_set = set(_internal_names)
    _metadata = ['source']

    @property
    def _constructor(self):
        """Return a Particles object when manipulating."""
        return Particles

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        self.source = kwargs.pop('source', None)
        super().__init__(*args, **kwargs)

    def __getitem__(self, i):
        """Return the particle 'i' if i is an int else item i from the dataframe."""
        if type(i) is int:
            try:
                return Particle(self[self['particle'] == self['particle'].unique()[i]], source=self.source)
            except KeyError:
                raise AttributeError("Particles not found. Make sure you track the features and link them together first.") from None
        return super().__getitem__(i)

    def __getattr__(self, attr):
        """Use methods and attributes from the underlying Particle bjects."""
        try:
            return super().__getattr__(attr)
        except AttributeError:
            # Get attribute from the underlying objects
            if hasattr(Particle, attr):
                if type(getattr(Particle, attr)) == property:
                    return [getattr(particle, attr) for particle in self]
                else:
                    def all_attr(*args, **kwargs):
                        return [getattr(particle, attr)(*args, **kwargs) for particle in self]
                    return all_attr
            elif any([hasattr(v, attr) for v in self]):
                return [getattr(particle, attr, None) for particle in self]
        return super().__getattribute__(attr)

    def __iter__(self):
        """Iterate over particles rather than columns."""
        #
        # I feel like this is going to lead to big, hard to debug problems...
        #
        caller = inspect.currentframe().f_back.f_code.co_name  # This is to avoid breaking the __repr__ function. There has to be a cleaner way to do this, but I don't want to rewrite __repr__ :)
        # print(caller)
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
        return Particle(self._iter.__next__()[1], source=self.source)

    @property
    def number(self):
        """The number of particles."""
        return len(self.groupby('particle'))


@extensions.append
class Particle(pd.DataFrame):
    """Container for a particle.

    Notes on implementation
    -----------------------

    """

    _metadata = ['source']

    @property
    def _constructor(self):
        """Return a Particle object when manipulating."""
        return Particle

    def __init__(self, *args, **kwargs):
        """Instanciate the object."""
        self.source = kwargs.pop('source', None)
        super().__init__(*args, **kwargs)

    @property
    def dwell_time(self):
        """The number of frames this particle was observed."""
        return self['frame'].max() - self['frame'].min()
