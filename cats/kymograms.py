# -*- coding: utf8 -*-
"""Draw and assemble kymograms."""
from __future__ import absolute_import, division, print_function

import numpy as np
import skimage
import skimage.draw
import pims.display
from . import extensions


@extensions.append
class Kymogram(np.ndarray):
    """Create and handle a kymogram.

    A kymogram is a displacement plot built with raw data. It is a projection of a given axis along time.

    `Kymogram` objects are nothing more than numpy arrays with special methods. They can be built using any input acceptable to the `numpy.array` function, or using the `from_coordinates` static method, as follow:
    ```python
    >>> images = cats.Images('/path/to/images/*.tif')
    >>> kymogram = Kymogram.from_coordinates((12, 12), (120, 58), images)
    >>> type(kymogram)
    cats.kymograms.Kymogram
    >>> other_kymogram = Kymogram(np.ones((10, 120)))
    >>> type(other_kymogram)
    cats.kymograms.Kymogram
    ```

    """

    @staticmethod
    def from_coordinates(beginning, end, images):
        """Build a kymogram from the given coordinates.

        Parameters:
        -----------
        beginning: 2-tuple of int
            the coordinates of the initial pixel of the kymogram as (x, y)
        end: 2-tuple of int
            the coordinates of the last pixel of the kymogram as (x, y). This last pixel will be included into the kymogram.
        images: cats.images.Images
            the images in which these coordinates are to be taken.

        """
        slice_ = skimage.draw.line(beginning[1], beginning[0], end[1], end[0])
        a = [image[slice_] for image in images]
        return np.array(a, dtype=images.dtype).T.view(Kymogram)

    def from_slice(s, images):
        """Build a kymogram from the given slice.

        Parameters:
        -----------
        s: lists of 2D coordinates
            an advanced index returning the pixels of the kymogram. For example, the output of `skimage.draw.line`.
        images: cats.images.Images
            the images in which this slice is to be taken.

        """
        a = [image[s] for image in images]
        return np.array(a, dtype=images.dtype).T.view(Kymogram)

    def __new__(cls, *args, **kwargs):
        """Create a new instance of Kymogram."""
        if len(args) == 0 and len(kwargs) == 0:
            args = [[]]
        return np.array(*args, **kwargs).view(cls)

    def __array_finalize__(self, obj):
        """Save the attributes."""
        self.display_width = 512

    def as_rgb(self, rgb=(1, 1, 1)):
        """Return the kymogram as an RGB array.

        Parameters:
        -----------
        rgb: `matplotlib.colors.ColorConverter`-compatible color
            The RGB color of the signal

        """
        return pims.display.to_rgb(self, rgb)

    def save(self, f):
        """Save the kymogram to the given file.

        Parameters:
        ----------
        f: str
            The file to save the kymogram in.

        """
        skimage.io.imsave(f, self)

    def save_as_rgb(self, f, rgb=(1, 1, 1)):
        """Save the kymogram to the given file as an 8-bit RGB image, instead of greyscale.

        Parameters:
        ----------
        f: str
            The file to save the kymogram in.
        rgb: `matplotlib.colors.ColorConverter`-compatible color
            The RGB color to give to the signal

        """
        skimage.io.imsave(f, self.as_rgb(rgb))

    def _repr_png_(self):
        """Show the kymogram in Jupyter."""
        if len(self.shape) < 3:
            i = self.as_rgb()
        else:
            i = self
        return pims.display._as_png(i, width=self.display_width)
