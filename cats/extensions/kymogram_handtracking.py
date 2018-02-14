"""Track particles and filter with handpicked tracks."""
from __future__ import absolute_import, division, print_function

import math
import scipy as sp
import numpy as np
import pandas as pd


def import_handtracking(track_file):
    """Transform a file of tracks manually picked from kymograms into a list.

    Parameters:
    -----------
    track_file: str
        path to the file containing the tracks

    File format:
    ------------
    # kymogram number 0
    frame 1 track 1  (kymogram's x)  x 1 track 1 (kymogram's y)
    frame 2 track 1 (kymogram's x)   x 2 track 1 (kymogram's y)
    ...

    frame 1 track 2  (kymogram's x)  x 1 track 2 (kymogram's y)
    frame 2 track 2 (kymogram's x)   x 2 track 2 (kymogram's y)
    frame 3 track 2 (kymogram's x)   x 3 track 2 (kymogram's y)
    ...

    ...

    # kymogram number 1
    ...

    Hence, it is:
    - Define kymogram number by starting the line with a `#`.
    - Add features location underneath in the format `frame` -tab- `x`
    - Separate tracks within the same kymogram with an empty line
    - You can have as many kymograms, as many tracks per kymogram and as many features per track that you'd like.

    Notes:
    ------
    - The kymogram id has to be an int. The number should match with the DNA molecule number of the `dna.csv_to_coordinates` function, if you built the kymogram this way, to ease your work with `particles.from_kymogram_handtracks`.

    Returns:
    --------
    tracks: dict
        A dictionary of tracks as follow:
        {kymo_id_0: [(frame0, x0), (frame1, x1), ...], [(frame0, x0), ...], ...],
        kymo_id_x: [(frame0, x0), (frame1, x1), ...], [(frame0, x0), ...], ...],
        ...}

    """
    with open(track_file, 'r') as f:
        raw_tracks = f.read()

    tracks = dict()
    for kymogram in raw_tracks.split('#'):
        kymo_tracks = kymogram.split('\n\n')
        split = kymo_tracks[0].partition('\n')
        kymo_id, kymo_tracks[0] = split[0].strip(), split[2]
        if kymo_id != '':
            kymo_id = int(kymo_id)
            tracks[kymo_id] = list()
            for track in kymo_tracks:
                t = list()
                for feature in track.split('\n'):
                    positions = feature.split('\t')
                    if len(positions) > 1:
                        t.append((int(math.floor(float(positions[0]))), float(positions[1])))
                if len(track) > 2:
                    tracks[kymo_id].append(t)

    return tracks


def handtracks_to_particles(tracks, dnas, channel=None, extrapolate=False):
    """Transform handtracks from kymograms into a Particles object.

    Kymograms are missing one dimension, the position of the DNA molecule in y. To get this information back, a dict/list of functions (1/kymogram) that give the y position for a given x must be given.
    This function returns coordinates that are relative to the DNA molecule, like in the DNA's `roi`.

    Parameters:
    -----------
    tracks: dict
        the handtracks as outputed by `cats.kymograms.import_handtracking`
    dnas: cats.dna.DNAs
        the DNA objects with key/index equal to the tracks' key
    channel: int
        the channel from which the kymogram was extracted in the DNA molecule
    extrapolate: bool
        whether to fill in the particle position in between the given points, using lines. If True, it will add a column `segment` to the dataframe, which represents which segment of the trace the feature belongs to (a segment is defined by two consecutive handpicked points).

    Returns:
    --------
    particles: dict of cats.particles.Particles objects
        the processed particles as kymo_id: Particles

    """
    import cats.particles  # This cannot be imported at loading time, as this would break the loading of extensions for particles
    particles = dict()
    for key, kymo_tracks in tracks.items():
        dna_particles = list()
        dna = dnas[key][channel]
        p1, p2 = dna.beginning, dna.end
        # Function for y
        m = (p1[1] - p2[1]) / (p1[0] - p2[0])
        p_c = dna.roi.shape[-1] / 2, dna.roi.shape[-2] / 2  # Some central reference point on the DNA molecule to get b
        b = p_c[1] - m * p_c[0]

        for i, track in enumerate(kymo_tracks):
            if extrapolate:
                ext_track = list()
                for j in range(0, len(track) - 1):
                    (x0, y0), (x1, y1) = track[j], track[j + 1]
                    x_m = (y0 - y1) / (x0 - x1)
                    x_b = y0 - x_m * x0
                    ext_track.extend([(frame, frame * x_m + x_b, j) for frame in range(x0, x1)])
                ext_track.append((x1, x1 * x_m + x_b, j))  # Add the last frame that was skipped by the last range.
                track = ext_track
            else:
                track = [(t[0], t[1], j) for j, t in enumerate(track[:-1])] + [(track[-1][0], track[-1][1], len(track) - 2)]
            dna_particles.extend([(x, m * x + b, frame, i, segment) for frame, x, segment in track])  # There's clearly a better way but I'm tired
        particles[key] = cats.particles.Particles(dna_particles, columns=('x', 'y', 'frame', 'particle', 'segment'), source=dnas[key].roi[channel], tracking_parameters={'handtracking': {'extrapolate': extrapolate}})
    return particles


def match_handtracks(particles, hand_particles, dist=2, allow_duplication=False):
    """Match handpicked particles with features that are tracked by software.

    Parameters:
    -----------
    particles: pandas.DataFrame or cats.particles.Particles or cats.particles.Particle
    hand_particles: pandas.DataFrame or cats.particles.Particles or cats.particles.Particle as obtained from `handtracks_to_particles`
        the manually selected particles (with all frames extrapolated)
    dist: int
        the maximal distance features can be from the points in the tracks or infered from the tracks.
    allow_duplication: bool
        Whether the same feature can be used for two different particles

    Track format:
    -------------
    A track is a list of features, in the format (x, y, f), with x, the position in x, y, the position in y and f, the frame at which the feature is found. Hence, the `tracks` parameter is a list of lists of tuples.
    The features defined by the tracks only have to be those between the segments of linear movement of the particle. The features in between those defined will be inferred (linearly).

    Notes
    ------
    Make sure that the coordinates in the DataFrame are from the same reference as in the tracks. For example, tracking will often return values from an ROI.

    """
    import cats.particles  # This cannot be imported at loading time, as this would break the loading of extensions for particles

    distances = sp.spatial.distance.cdist(hand_particles[['x', 'y']], particles[['x', 'y']])
    less_than_dist = distances <= dist
    same_frame = hand_particles['frame'].values[:, np.newaxis] == particles['frame'].values
    adjusted_distances = np.where(np.logical_and(less_than_dist, same_frame), distances, np.inf)
    if len(adjusted_distances) == 0:
        return cats.particles.Particles()
    matches = np.array([np.arange(0, len(hand_particles)), np.argmin(adjusted_distances, axis=1), np.min(adjusted_distances, axis=1)])
    valid_particles = matches[2] != np.inf
    filtered = particles.iloc[matches[1][valid_particles]].copy()

    # Write down segment and particle numbers
    filtered['particle'] = hand_particles.iloc[matches[0][valid_particles]]['particle'].values
    if 'segment' in hand_particles.columns:
        filtered['segment'] = hand_particles.iloc[matches[0][valid_particles]]['segment'].values

    # Remove duplicates
    if not allow_duplication:
        filtered = filtered[np.logical_not(filtered.duplicated(['x', 'y', 'frame']))].copy()

    # Update element attributes
    if hasattr(particles, '_element_attributes') and 'particle' not in particles.columns:
        filtered._element_attributes = {}
        for attr, value in particles._element_attributes.items():
            filtered._set_element_attribute(attr, value)
    filtered._update_element_attribute('tracking_parameters', {'handtracking': {'matched': True, 'allow_duplication': allow_duplication}})

    return filtered.renumber_particles()


def import_kymogram_handtracks(self, track_files, match_particles=True, replace_particles=True, extrapolate=True, dist=2, allow_duplication=False, ignore_unmatched=True):
    """Track particles in each DNA in the group, and filter the tracks using the provided hand tracks. For DNA without handtracks provided, filter as possible.

    Parameters:
    -----------
    track_files: dict
        The paths to the handtrack files for each channel, in the shape {channel_name: path}
    match_particles: bool
        If True, will try to match the points in the hand tracks with tracked features from the `particles` attribute.
    replace_particles: bool
        If True, will replace the `particles` attribute with the handtracks. If False, will write the handtracks in the `hand_particles` attribute.
    extrapolate: bool
        Whether to extrapolate the position on features in between two point on the track. Generally a good idea when `match_particles` is True. If True, it will add a column `segment` to the dataframe, which represents which segment of the trace the feature belongs to (a segment is defined by two consecutive handpicked points).
    dist: int
        If `match_handtracks` is True, the maximal distance features can be from the points in the tracks or infered from the tracks.
    allow_duplication: bool
        If `match_handtracks` is True, whether the same feature can be used for two different particles
    ignore_unmatched: bool
    If `match_handtracks` is True, whether to ignore the matching process if a DNA object does not have pre-tracked particles.

    """
    for channel, f in track_files.items():
        hand_particles = handtracks_to_particles(import_handtracking(f), self, channel, extrapolate)
        for i, hand in hand_particles.items():
            if match_particles:
                try:
                    particles = self[i][channel].particles
                    hand = match_handtracks(particles, hand, dist, allow_duplication)
                except AttributeError:
                    if not ignore_unmatched:
                        raise ValueError('If `match_particles` is True, particles must have been tracked by software and stored in each DNA object\'s `particles` attribute prior to using this function.')
                    else:
                        pass

            # Write the dataframe in the DNAChannel objects.
            if replace_particles:
                self[i][channel].particles = hand
            else:
                self[i][channel].hand_particles = hand


_extension = {
    'DNAs': {
        'import_handtracks': import_kymogram_handtracks
    }
}
