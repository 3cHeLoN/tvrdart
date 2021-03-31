"""
Create a 3d phantom.
"""
import os
import numpy as np
from tvrdart.util import cylinder, sphere


def main(grid_size=500):
    """Create a simple 3d phantom."""
    volume = np.zeros((grid_size, grid_size, grid_size))

    v_1 = sphere(int(0.2 * grid_size))
    cube_size = int(0.4 * grid_size)
    v_2 = np.ones((cube_size, cube_size, cube_size))
    v_3 = cylinder(int(0.1 * grid_size), int(0.7 * grid_size))
    v_3 = np.transpose(v_3, [1, 2, 0])

    z_offset = int(0.25 * grid_size)
    y_offset = x_offset = int(0.1 * grid_size)
    volume[z_offset:v_1.shape[0] + z_offset,
           y_offset:v_1.shape[1] + y_offset,
           x_offset:v_1.shape[2] + x_offset] += v_1
    volume[z_offset:v_2.shape[0] + z_offset,
           int(0.5 * grid_size):v_2.shape[1] + int(0.5*grid_size),
           x_offset:v_2.shape[2] + x_offset] += v_2
    volume[z_offset:v_3.shape[0] + z_offset,
           int(0.3 * grid_size):v_3.shape[1] + int(0.3 * grid_size),
           int(0.6 * grid_size):v_3.shape[0] + int(0.6 * grid_size)] += v_3

    volume.tofile(os.path.join('phantoms',
                               'phantom.raw'))


if __name__ == '__main__':
    main(grid_size=200)
