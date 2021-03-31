"""Test TVR-DART"""
import os
import numpy as np
import astra
import tvrdart


def main():
    """TVR-DART on a 3d cone beam dataset."""
    image = np.fromfile(os.path.join('phantoms',
                                     'phantom.raw')).reshape((200, 200, 200))
    image = image / image.max()
    print("Image shape", image.shape)

    n_pixels = image.shape[0]
    angles = np.linspace(0, np.pi, 12, endpoint=False)

    origin_det = 0
    source_origin = 2000
    proj_geom = astra.create_proj_geom('cone', 1, 1, int(1.5 * n_pixels), int(1.5 * n_pixels),
                                       angles, source_origin, origin_det)
    vol_geom = astra.create_vol_geom(n_pixels, n_pixels, n_pixels)
    proj_id = astra.create_projector('cuda3d', proj_geom, vol_geom)

    tomo_projector = astra.OpTomo(proj_id)
    projections = tomo_projector.FP(image)

    gray_values = np.unique(image.ravel())
    thresholds = np.diff(gray_values) / 2 + gray_values[:-1]
    print('Gray values are', gray_values)

    tvrdart_alg = tvrdart.Tvrdart(projections,
                                  tomo_projector,
                                  image.shape,
                                  gray_values,
                                  thresholds,
                                  sigma=100,
                                  iter_lim=200,
                                  nonnegative=True)

    reconstruction = tvrdart_alg.run()


if __name__ == '__main__':
    main()
