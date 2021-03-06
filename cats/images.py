# -*- coding: utf8 -*-
"""Utility functions to help with image handling."""
from __future__ import absolute_import, division, print_function

import numpy as np
import colorsys
import warnings
import pims
import pims.display
import slicerator
import os
import skimage
import skimage.io
import skimage.exposure
import skimage.transform
import imageio

from cats import extensions


@extensions.append
class Images(object):
    """Handle multi-dimensinal images/datasets.

    Import is based on PIMS and, as a consequence, all formats supported are
    those supported by PIMS.

    Channels are not supported for now.

    The Images objects are indexable for frames and dimensions, in the order
    (frame, y, x). If only one frame is requested, it will return a
    pims.frame.Frame object. If several frames are requested, it will return an
    Images object sliced accordingly.

    Images objects can be pickled and unpickled. If they are unpickled in the
    absence of the source of the images, the user will, of course, not be able
    to access the images themselves, but will still be able to use the object.
    If one brings the source back, one can use the 'reload' method to reload
    the object with the images.

    Parameters:
    -----------
    images: str, pims or cats.Images object
        The images to read. Can be a path, readable by `pims.open()`, or an object with images already loaded.
    frames: slice, list, numpy array...
        The frames to use, following numpy's indexing format (including advanced indexing)
    pixels: slice, list, numpy array...
        The pixels to use in the images, following numpy's indexing format (including advanced indexing)

    Example:
    --------
    >>> i = Images('/path/to/images/*.tif')
    >>> type(i[0])
    pims.frames.Frames
    >>> type(i[:10])
    cats.images.Images

    Examples of acceptable slicing syntax:
    Images[0]  # Get the first frame
    Images[0, 5:10, 10:60]  # Get the first frame, between pixels 5 and 10 in y
                              and pixels 10 and 60 in x
    Images[..., :256]  # Get all images, truncated in x until pixel 256
    Images[::10]  # To get one out of ten images
    Images[[1, 2, 3], [0, 10, 20], [15, 15, 25]]  # Get pixels with (x, y)
                                                    positions (15, 0), (15, 10),
                                                    (25, 20) from the second,
                                                    third and fourth images.

    """

    def __init__(self, images, frames=slice(None), pixels=(slice(None), slice(None))):
        """Instanciate the images."""
        if type(images) is str:
            if os.path.isfile(images) and os.path.splitext(images)[1] in ['.tif', '.tiff']:
                self.base = imageio.imread(images)
                self.base.path = images
            else:
                self.base = pims.open(images)
        else:
            self.base = images

        # For now, this is how we'll deal with slicing:
        # Frames are stored as an array of the frame numbers. They can be directly sliced using advanced indexing.
        # x and y are just going to be applied to the loaded image every time an image is loaded.
        # In the end, all I'm doing is letting numpy deal with the slicing :)
        self.frame_index = np.arange(len(self.base))[frames]
        self.pixel_slice = pixels

        # Store for future pickling and unpickling in a base-less context.
        # These are commonly used and calculated from the base, so they are worth saving.
        self.path
        self.shape
        self.dtype
        self.itemsize

    def __getitem__(self, items):
        """Return the slice of or whole frame(s) as requested.

        Expects input as: frame, y, x

        """
        #
        # THIS IS SLOPPY!
        # This only works when frames and pixels are separated, which may be confusing to the user...
        #

        # Deal with a single frame slice
        if type(items) is not tuple:
            items = items,

        # Deal with an ellipsis
        try:
            pos = items.index(...)  # Figure out the position of the ellipsis
            n_ellipsized = 3 - len(items) + 1  # Figure out how many dimmensions are contained in the ellipsis
            items = items[:pos] + (slice(None), ) * n_ellipsized + items[pos + 1:]
        except ValueError:
            pass

        # How many frames to return?
        frames = self.frame_index[items[0]]
        if np.isscalar(frames):
            frames = [frames]

        # No frames
        elif len(frames) == 0:
            return np.array([], dtype=self.dtype).view(pims.frame.Frame)

        # Only return one frame
        if len(frames) == 1:
            frame_no = frames[0]
            f = self.base[frame_no][self.pixel_slice][items[1:]].copy()
            if type(f) is pims.frame.Frame:
                f.frame_no = frame_no
            return f

        # Return several frames as an Images object
        else:
            return Images(self, frames=items[0], pixels=items[1:])

    def __getattr__(self, attr):
        """Return the requested attribute.

        When dealing with the absence of images, i.e. the absence of the "base" attribute, this makes sure that an explicit error is returned.
        It does so too with image related attributes.

        """
        if attr == 'base':
            try:
                self.reload()  # Check if the source magically reappeared first
                return self.base
            except (FileNotFoundError, pims.api.UnknownFormatError) as e:
                errormsg = 'Cannot access the images'
                if hasattr(self, '_path'):
                    errormsg += " at " + self._path
                raise FileNotFoundError(errormsg) from None
        self.__getattribute__(attr)

    def __iter__(self):
        """Start iteration."""
        self._iter_counter = 0
        return self

    def __next__(self):
        """Continue iteration."""
        if self._iter_counter < len(self):
            self._iter_counter += 1
            return self.__getitem__(self._iter_counter - 1)
        else:
            raise StopIteration

    @property
    def path(self):
        """The path to the base images."""
        if not hasattr(self, '_path'):
            t = type(self.base)
            if t is slicerator.Slicerator:
                self._path = self.base._ancestor.pathname  # Slicerator could do that job for us...
            elif t is Images or t is imageio.core.util.Image:
                self._path = self.base.path
            else:
                self._path = self.base.pathname  # Pims object
        return self._path

    @property
    def shape(self):
        """The shape of the object as (frame, y, x)."""
        if not hasattr(self, '_shape'):
            f = len(self.frame_index)
            self._shape = (f, ) + self[0].shape
        return self._shape

    @property
    def ndim(self):
        """The number of dimensions."""
        return len(self.shape)

    @property
    def dtype(self):
        """The data type of the images."""
        return self.base[0].dtype

    @property
    def itemsize(self):
        """The size of one pixel, in bytes."""
        return self.base[0].itemsize

    @property
    def size(self):
        """The total number of pixels in the object."""
        return np.multiply(*self.shape)

    def __len__(self):
        """Return the number of frames."""
        return self.shape[0]

    def __getstate__(self):
        """Pickle the whole object, except the reader."""
        # Do not pickle source images.
        blacklist = []
        if 'base' in self.__dict__ and not isinstance(self.base, Images):
            blacklist.append('base')
        return dict(((k, v) for k, v in self.__dict__.items() if k not in blacklist))

    def __setstate__(self, state):
        """Load the whole object, but tolerate the absence of the images."""
        for k, v in state.items():
            setattr(self, k, v)
        try:
            self.base  # Try to reload the pims images if need be
        except FileNotFoundError:
            pass

    def reload(self):
        """Reload the images in case the source was missing."""
        if 'base' in self.__dict__.keys() and type(self.base) is Images:
            self.base.reload()
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                if os.path.isfile(self.path) and os.path.splitext(self.path)[1] in ['.tif', '.tiff']:
                    self.base = imageio.imread(self.path)
                    self.base.path = self.path
                else:
                    self.base = pims.open(self.path)

    def save(self, folder, prefix='', start_value=0, extension='tif'):
        """Save the images as a sequence into the given folder.

        Parameters:
        -----------
        folder: str
            the folder in which the save the data. If the folder does not exist, it will be created.
        prefix: format()able type
            the prefix to give to the files.
        start_value: int
            The number of the first frame
        extension: str
            the file extension. It has to be an extension writable by skimage.io.imsave

        """
        if not os.path.isdir(folder):
            os.makedirs(folder)
        nb = np.ceil(np.log10(len(self))) if len(self) > 0 else 0
        for i, image in enumerate(self):
            skimage.io.imsave(os.path.join(folder, "{}{:0{}d}.{}".format(prefix, i + start_value, int(nb), extension)), image)

    def asarray(self):
        """Return the object as a numpy array (loaded in memory)."""
        return np.array([i for i in self])

    # def __copy__(self):
    #     """Can not copy images."""
    #     return self
    #
    # def __deepcopy__(self, memo):
    #     """Can not copy images."""
    #     return self


@extensions.append
class Image(np.ndarray):
    """Represent one image and associated functions."""

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

    def rescale_intensity(self, scale):
        """Rescale the intensity of the image to the given scale."""
        return skimage.exposure.rescale_intensity(self, in_range=scale).view(Image)

    def rescale(self, scale):
        """Rescale the image to the given scale, no interpolation."""
        return skimage.transform.rescale(self, scale, order=0, preserve_range=True).astype(self.dtype).view(Image)


def stack_images(*images, spacing=2, spacing_color=255, axis=0):
    """Stack images into one image.

    You need to provide 2D RGBs as input (height * width * 3). Sizes must match in the axis not being stacked.

    Parameters:
    ----------
    *images: np.array
        the images to stack
    spacing: int
        the number of pixels between each image
    spacing_color: int or 3-tuple
        the color, as number for grayscales or 3-tuple for RGB, of the pixels in between images
    axis: int
        the axis onto which to stack the images (0 for vertical, 1 for horizontal)

    Returns:
    -------
    image: np.array
        The stacked images

    """
    dims = np.array([i.shape[:2] for i in images])
    if axis == 0:
        width, height = dims[:, 1].max(), sum(dims[:, 0]) + (len(dims) - 1) * spacing
    else:
        height, width = dims[:, 0].max(), sum(dims[:, 1]) + (len(dims) - 1) * spacing
    stack = np.zeros((height, width, 3), dtype=np.uint8)
    stack.fill(spacing_color)
    position = 0
    for i in images:
        if axis == 0:
            stack[position: position + i.shape[0], :, :] = i
        else:
            stack[:, position: position + i.shape[1], :] = i
        position += i.shape[axis] + spacing
    return stack


def get_image_depth(image):
    """Return the pixel depth of the image (given as a ndarray from, for example, bioformats or scipy.misc.imread), in bits."""
    convert = {
        np.uint8: 8,
        np.uint16: 16,
        np.uint32: 32,
        np.uint64: 64,
        np.int8: 8,
        np.int16: 16,
        np.int32: 32,
        np.int64: 64,
    }
    try:
        return convert[image.dtype.type]
    except KeyError:
        raise ValueError('Unrecognized image type.')


def grayscale_to_rgb(grayscale):
    """Transform a grayscale image to a 8bits RGB.

    Parameters:
    ------------
        grayscale: numpy array
            the image to transform into 8bit RGB

    """
    max_i = 2**get_image_depth(grayscale) - 1
    rgb = np.zeros([3] + list(grayscale.shape), dtype=np.uint8)
    rgb[..., :] = grayscale / max_i * 255
    rgb = rgb.transpose(1, 2, 0)
    return rgb


def color_grayscale_image(image, rgb):
    """Color a grayscale image into an RGB image.

    Parameters:
    -----------
    image: np.array
        The grayscale image to colorify
    rgb: 3-tuple
        A tuple of RGB values from 0 to 255 representing the color that the signal in the image is to take.

    Returns:
    --------
    rgb_image: the color, 8-bit RGB image.

    """
    light_factor = image.flatten() / 2**get_image_depth(image)
    hls = colorsys.rgb_to_hls(*[c / 255 for c in rgb])

    # Write the image as HLS
    flat_image = np.zeros([np.prod(image.shape), 3])
    flat_image[:, 0] = hls[0]
    flat_image[:, 1] = light_factor
    flat_image[:, 2] = hls[2]

    # Transform back to RGB
    return (np.apply_along_axis(lambda x: colorsys.hls_to_rgb(*x), 1, flat_image) * 255).astype(np.uint8).reshape(list(image.shape) + [3])


def blend_rgb_images(*images):
    """Blend RGB images together.

    Parameters:
    ----------
    images: np.ndarray
        All the RGB images to blend.

    """
    image = np.sum(np.array(images).astype(int), axis=0)
    return image.clip(max=255).astype(np.uint8)


def generate_registration_function(reference, relative):
    """Generate a registration function that transforms coordinates (x, y) from a reference source into (x, y) to the relative source.

    Parameters:
    -----------
    reference: 2-tuple
        the coordinates (x0, y0) of the reference point
    relative: 2-tuple
        the coordinates (x1, y1) of the relative point

    Returns:
    --------
    register: func
        a function that transforms reference points into relative points

    """
    dx = relative[0] - reference[0]
    dy = relative[1] - reference[1]

    def register(x, y):
        """Transform coordinates (x,y) of the reference into (x,y) of the relative dataset."""
        return x + dx, y + dy

    return register


def align_channels(c1, c2, coords_c1, coords_c2):
    """Align two channels using the given registration function.

    Parameters:
    -----------
    c1, c2: cats.Images
        The channels to align
    coords_c1, coords_c2: 2-tuple of int/float
        The coordinates of a same point in the two channels

    Returns:
    --------
    slices: list of cats.Images
        the two channels cropped so that they are aligned

    """
    slices = list()
    for ref, rel, coords_ref, coords_rel in ((c1, c2, coords_c1, coords_c2), (c2, c1, coords_c2, coords_c1)):
        orig = (0, 0)
        end = (ref.shape[2], ref.shape[1])
        reg = generate_registration_function(coords_rel, coords_ref)
        rel_orig = reg(*orig)
        rel_end = reg(*end)
        slice_x = slice(max(orig[0], rel_orig[0]), min(end[0], rel_end[0], rel.shape[2]))
        slice_y = slice(max(orig[1], rel_orig[1]), min(end[1], rel_end[1], rel.shape[1]))
        slices.append(ref[:, slice_y, slice_x])
    return slices
