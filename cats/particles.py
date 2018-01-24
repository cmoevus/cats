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


def from_kymogram_handtracks(tracks, dnas, channel=0, extrapolate=False):
    """Transform handtracks from kymograms into a Particles object.

    Kymograms are missing one dimension, the position of the DNA molecule in y. To get this information back, a dict/list of functions (1/kymogram) that give the y position for a given x must be given.
    This function expects the coordinates to be relative to a ROI, and will transform them into the same coordinates as the DNA molecule.

    Parameters:
    -----------
    tracks: dict
        the handtracks as outputed by `cats.kymograms.import_handtracking`
    dnas: dict/list
        the DNA objects with key/index equal to the tracks' key
    channel: int
        the channel from which the kymogram was extracted in the DNA molecule
    extrapolate: bool
        whether to fill in the particle position in between the given points, using lines.

    Returns:
    --------
    particles: dict of cats.particles.Particles objects
        the processed particles as kymo_id: Particles

    """
    particles = dict()
    for key, kymo_tracks in tracks.items():
        if type(dnas[key]) != cats.dna.DNA:
            raise ValueError("Use DNA object for reference.")

        dna_particles = list()
        p1, p2 = dnas[key].points[channel]

        # Function for y
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
        b = p1[1] - m * p1[0]

        # Direction of x
        d = (p2[0] - p1[0]) / abs(p2[0] - p1[0])

        # Offset from the ROI
        roi_limits = dnas[key].roi[channel].slices[1]
        rel_y = min(roi_limits.start, roi_limits.stop)

        for i, track in enumerate(kymo_tracks):
            if extrapolate:
                ext_track = list()
                for j in range(0, len(track) - 1):
                    (x0, y0), (x1, y1) = track[j], track[j + 1]
                    x_m = (y0 - y1) / (x0 - x1)
                    x_b = y0 - x_m * x0
                    ext_track.extend([(frame, frame * x_m + x_b) for frame in range(x0, x1)])
                ext_track.append((x1, x1 * x_m + x_b))  # Add the last frame that was skipped by the last range.
                track = ext_track
            dna_particles.extend([(x, m * (p1[0] + x * d) + b - rel_y, frame, i) for frame, x in track])  # There's clearly a better way but I'm tired
        particles[key] = Particles(dna_particles, columns=('x', 'y', 'frame', 'particle'), source=dnas[key].roi[channel])
    return particles
