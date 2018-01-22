"""Track particles and filter with handpicked tracks."""
from __future__ import absolute_import, division, print_function

import cats.kymograms
import cats.particles
import cats.detect


def track_with_handtracks(self, track_files):
    """Track particles in each DNA in the group, and filter the tracks using the provided hand tracks. For DNA without handtracks provided, filter as possible.

    Parameters:
    -----------
    track_files: list of str
        The paths to the handtrack files for each channel

    """
    hand_particles = [cats.particles.from_kymogram_handtracks(cats.kymograms.import_handtracking(f), self, i, True) for i, f in enumerate(track_files)]
    for i, dna in enumerate(self):
        # Associate hand tracks with DNA molecules
        dna.particles = []
        dna.hand_particles = []
        # Track
        for j in range(len(dna.datasets)):
            hand = hand_particles[j][i] if j < len(hand_particles) and i in hand_particles[j].keys() else None
            features = cats.detect.features(dna.roi[j])
            if hand is not None:
                particles = cats.detect.particles_using_handtracks(features, hand)
            else:
                particles = cats.detect.filter_particles(cats.detect.particles(cats.detect.filter_features(features)))
            dna.hand_particles.append(hand)
            dna.particles.append(particles)


_extension = {
    'DNAs': {
        'track_with_handtracks': track_with_handtracks
    }
}
