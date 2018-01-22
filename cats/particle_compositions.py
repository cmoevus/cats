"""Legacy compositions of DataFrames into Particle(s) objects."""


class Particles(object):
    """Container for particles.

    One can access individual particles by iterating over the object, which returns Particle objects, or by using the Particles.grouped attribute, which is the DataFrame grouped by particle ID.

    Arguments:
    ----------
    dataframe: pandas.DataFrame
        The dataframe containing the particle/feature information, in a format similar/equal to trackpy's
    source: cats.Images
        The source of the data

    Note on implementation
    ----------------------
    This is a simple composition of a Pandas DataFrame. Using a composition has many advantages for the programmer, and ultimately for the user (less bugs), but be aware that you may have to go and play with the dataframe.
    Implemented as follow:
    class Particles(object):
         def __init__(self, df):
            self.dataframe = df
            self.particles = self.dataframe.groupby('particles')

         def __getattr__(self, attr):
            try:
                 return getattr(self.dataframe, attr)
            except AttributeError:
                raise AttributeError('Particles object has no attribute {0}'.format(attr))

    One of the issues is that if you subset the Particle object, you will get a DataFrame back.
    The Particles object will first look into its own attributes (be carefule not to set attributes that would override important DataFrame methods and attributes) and then within the DataFrame's attributes.
    """

    def __init__(self, dataframe=None, source=None):
        """Instanciate the class."""
        self.dataframe = dataframe
        self.source = source

    @property
    def grouped(self):
        """Return the particles as a DataFrameGroupBy object."""
        if not hasattr(self, '_grouped'):
            try:
                self._grouped = self.dataframe.groupby('particle')
            except KeyError:
                raise RuntimeError('You must first link features into particles to be able to access this attribute.') from None
        return self._grouped

    @property
    def dataframe(self):
        """Return the DataFrame containing the features and particles information."""
        return self._dataframe

    @dataframe.setter
    def dataframe(self, df):
        self._dataframe = df
        if hasattr(self, 'grouped'):
            del self._grouped

    def __getattr__(self, attr):
        """Get the method or attribute from the dataframe."""
        if attr in ['grouped', '_grouped']:
            raise AttributeError('Particles have no attribute {0}'.format(attr))
        try:
            return getattr(self.dataframe, attr)
        except AttributeError:
            raise AttributeError('Particles have no attribute {0}'.format(attr))

    def __getitem__(self, i):
        """Return the particle 'i' if i is an int else item i from the dataframe."""
        if type(i) is int:
            return Particle(self.dataframe[self.dataframe['particle'] == i], source=self.source)
        return self.dataframe[i]

    def __iter__(self):
        """Make Particles object iterable."""
        try:
            self._iter = self.grouped.__iter__()
        except AttributeError:
            raise AttributeError("Particles not found. Make sure you track the features and link them together first.") from None
        return self

    def __next__(self):
        """Continue iteration."""
        return Particle(self._iter.__next__()[1], self.source)


class Particle(object):
    """Container for a particle.

    Arguments:
    ----------
    dataframe: pandas.DataFrame
        The dataframe containing the particle/feature information, in a format similar/equal to trackpy's
    source: cats.Images
        The source of the data,

    Note on implementation
    ----------------------
    This is a simple composition of a Pandas DataFrame. Using a composition has many advantages for the programmer, and ultimately for the user (less bugs), but be aware that you may have to go and play with the dataframe.
    Implemented as follow:
    class Particle(object):
         def __init__(self, df):
            self.dataframe = df
            self.particles = self.dataframe.groupby('particles')

         def __getattr__(self, attr):
            try:
                 return getattr(self.dataframe, attr)
            except AttributeError:
                raise AttributeError('Particles object has no attribute {0}'.format(attr))

    One of the issues is that if you subset the Particle object, you will get a DataFrame back.
    The Particle object will first look into its own attributes (be careful not to set attributes that would override important DataFrame methods and attributes) and then within the DataFrame's attributes.
    """

    def __init__(self, dataframe, source=None):
        """Instanciate a particle."""
        self.dataframe = dataframe
        self.source = source

    def __getattr__(self, attr):
        """Get the method or attribute from the dataframe."""
        try:
            return getattr(self.dataframe, attr)
        except AttributeError:
            raise AttributeError('Particle has no attribute {0}'.format(attr))

    def __getitem__(self, item):
        """Return items from the dataframe."""
        return self.dataframe[item]
