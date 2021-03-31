"""Test TVR-DART"""
import os
import imageio
import numpy as np
import astra
import tvrdart


def main():
    """Estimate reconstruction and gray values."""
    image = imageio.imread(os.path.join('phantoms',
                                        'phantom.png'))
    image = image.mean(axis=-1)

    n_pixels = image.shape[0]
    max_angle = 70
    angles = np.linspace(-max_angle/180*np.pi, max_angle/180*np.pi, 15,
                         endpoint=True) + 0.2 / 180 * np.pi

    proj_geom = astra.create_proj_geom('parallel', 4, int(2 * n_pixels / 4),
                                       angles)
    vol_geom = astra.create_vol_geom(n_pixels, n_pixels)
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)

    tomo_projector = astra.OpTomo(proj_id)
    projections = tomo_projector.FP(image)

    # back projector
    proj_geom = astra.create_proj_geom('parallel', 1,
                                       2 * int(n_pixels / 4), angles)
    vol_geom = astra.create_vol_geom(int(n_pixels / 4),
                                     int(n_pixels / 4))
    proj_id = astra.create_projector('cuda', proj_geom, vol_geom)
    tomo_projector = astra.OpTomo(proj_id)

    noise = np.random.normal(0, 0.1 * projections.mean(),
                             size=projections.shape)
    projections = projections + noise

    gray_values = np.array([0, 90, 234]) - 128
    thresholds = np.diff(gray_values) / 2 + gray_values[:-1]

    gray_values = [None, None, None]

    # initialize algorithm
    tvrdart_alg = tvrdart.Tvrdart(projections, tomo_projector,
                                  astra.geom_size(vol_geom), gray_values,
                                  thresholds, sigma=10,
                                  iter_lim=100, nonnegative=True)

    # run algorithm
    reconstruction = tvrdart_alg.run_all(outer_iterlim=500)


if __name__ == '__main__':
    main()
