"""Test TVR-DART"""
import os
import numpy as np
import astra
import tvrdart


def main():
    """TVR-dart on a 3d phantom."""
    if not os.path.isfile(os.path.join('phantoms', 'phantom.raw')):
        print("Run s001_create_3d_phantom.py to generate a 3d phantom!")

    image = np.fromfile(os.path.join('phantoms',
                                     'phantom.raw')).reshape((200, 200, 200))
    image = image / image.max()

    n_pixels = image.shape[0]
    angles = np.linspace(0.1, np.pi + 0.5, 5, endpoint=False)

    proj_geom = astra.create_proj_geom('parallel3d', 1, 1,
                                       n_pixels,
                                       n_pixels,
                                       angles)
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
                                  sigma=10,
                                  iter_lim=100,
                                  nonnegative=True)

    tvrdart_alg.run()


if __name__ == '__main__':
    main()
