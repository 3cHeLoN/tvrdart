import sys
import numpy as np
import scipy.sparse.linalg


class OpTV3D(scipy.sparse.linalg.LinearOperator):
    """Object that implements the total variation operator."""

    def __init__(self, row_count, col_count, slice_count):
        """Initialize TV operator."""
        self.dtype = np.float32

        self.row_count = row_count
        self.col_count = col_count
        self.slice_count = slice_count

        matrix_row_count = row_count * col_count * slice_count
        self.shape = (3 * matrix_row_count, matrix_row_count)

        super().__init__(self.dtype, self.shape)

        self.input_size = (slice_count, row_count, col_count)
        self.transpose_optv3d = OpTVTranspose(self)

    def _transpose(self):
        return self.transpose_optv3d

    def _matvec(self, input_vec):
        """Forward product."""
        volume = input_vec.reshape(self.input_size)

        grad_x = -volume
        grad_x[:, :, :-1] += volume[:, :, 1:]
        grad_y = -volume
        grad_y[:, :-1, :] += volume[:, 1:, :]
        grad_z = -volume
        grad_z[:-1, :, :] += volume[1:, :, :]
        return np.concatenate((grad_x.ravel(), grad_y.ravel(), grad_z.ravel()))

    def rmatvec(self, input_vec):
        """Backward product."""
        grad_x = input_vec[:self.shape[1]].reshape(self.input_size).copy()
        grad_y = input_vec[self.shape[1]:2 * self.shape[1]].reshape(self.input_size).copy()
        grad_z = input_vec[2 * self.shape[1]:].reshape(self.input_size).copy()

        vol_x = -grad_x
        vol_x[:, :, 1:] += grad_x[:, :, :-1]
        vol_y = -grad_y
        vol_y[:, 1:, :] += grad_y[:, :-1, :]
        vol_z = -grad_z
        vol_z[1:, :, :] += grad_z[:-1, :, :]

        return (vol_x + vol_y + vol_z).ravel()


class OpTV2D(scipy.sparse.linalg.LinearOperator):
    """Object that implements the total variation operator."""

    def __init__(self, row_count, col_count):
        """Initialize TV operator."""
        self.dtype = np.float32

        self.row_count = row_count
        self.col_count = col_count

        matrix_row_count = row_count * col_count
        self.shape = (2 * matrix_row_count, matrix_row_count)

        super().__init__(self.dtype, self.shape)

        self.input_size = (row_count, col_count)
        self.transpose_optv2d = OpTVTranspose(self)

    def _transpose(self):
        return self.transpose_optv2d

    def _matvec(self, input_vec):
        """Forward product."""
        volume = input_vec.reshape(self.input_size)

        # compute the diff in each direction
        grad_x = -volume
        grad_x[:, :-1] += volume[:, 1:]
        grad_y = -volume
        grad_y[:-1, :] += volume[1:, :]

        # combine the results
        return np.concatenate((grad_x.ravel(), grad_y.ravel()))

    def rmatvec(self, input_vec):
        """Backward product."""
        grad_x = input_vec[:self.shape[1]].reshape(self.input_size)
        grad_y = input_vec[self.shape[1]:].reshape(self.input_size)

        vol_x = -grad_x
        vol_x[:, 1:] += grad_x[:, :-1]
        vol_y = -grad_y
        vol_y[1:, :] += grad_y[:-1, :]

        return (vol_x + vol_y).ravel()


class OpTVTranspose(scipy.sparse.linalg.LinearOperator):
    """Object that provides the transpose operator ".T" of an OpTV object."""

    def __init__(self, parent):
        self.parent = parent
        self.dtype = np.float32
        self.shape = (parent.shape[1], parent.shape[0])

        super().__init__(self.dtype, self.shape)

    def _tranpose(self):
        return self.parent

    def _matvec(self, input_vec):
        return self.parent.rmatvec(input_vec)

    def rmatvec(self, input_vec):
        return self.parent._matvec(input_vec)
