"""Utility functions for tomography."""
import numpy as np
import matplotlib.pyplot as plt


def sphere(radius):
    """Generate a sphere."""
    z_points, y_points, x_points = np.meshgrid(np.arange(-radius + 0.5,
                                                         radius - 0.5, 1),
                                               np.arange(-radius + 0.5,
                                                         radius - 0.5, 1),
                                               np.arange(-radius + 0.5,
                                                         radius - 0.5, 1),
                                               indexing='ij')
    return (x_points ** 2 + y_points ** 2 + z_points ** 2 <=
            radius ** 2).astype(float)


def cylinder(radius, length):
    """Generate a cylinder."""
    disk_image = disk(radius)
    return np.transpose(np.tile(disk_image, (length, 1, 1)), [2, 1, 0])


def disk(radius):
    """Generates image of a disk."""
    y_points, x_points = np.meshgrid(np.arange(-radius + 0.5, radius - 0.5, 1),
                                     np.arange(-radius + 0.5, radius - 0.5, 1),
                                     indexing='ij')
    return (x_points ** 2 + y_points ** 2 <= radius ** 2).astype(float)


class RealtimeImager:

    """Image picture animation in realtime."""

    def __init__(self, image_0, vmin=None, vmax=None, cmap='gray'):
        """Initialize object."""
        self.figure = plt.figure()
        self.axis = self.figure.add_subplot(111)
        self.figure.canvas.draw()
        self.vmin = vmin
        self.vmax = vmax
        if vmin is None:
            vmin = np.quantile(image_0, 0.05)
        if vmax is None:
            vmax = np.quantile(image_0, 0.95)
        self.axes_image = plt.imshow(image_0, vmin=vmin, vmax=vmax, cmap=cmap)
        plt.show(block=False)
        plt.draw()

    def update(self, image):
        """Show the next image frame."""
        if self.vmin is None:
            vmin = np.quantile(image, 0.05)
        else:
            vmin = self.vmin
        if self.vmax is None:
            vmax = np.quantile(image, 0.95)
        else:
            vmax = self.vmax

        #  self.axes_image.set_array(image / image.max())
        self.axes_image.set_clim(vmin, vmax)
        self.axes_image.set_data(image)
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()

    def close(self):
        plt.close(self.figure)
