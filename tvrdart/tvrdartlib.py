"""TVR-DART implementation."""

import numpy as np
import scipy
import matplotlib.pyplot as plt
import sklearn.mixture
import sklearn.preprocessing
from . import reconstruct


class Computable:
    """Computable object."""

    def __init__(self):
        self._value = None

    def _compute(self):
        """Dummy function."""
        return self

    def __call__(self):
        """Return value."""
        if self._value is None:
            self._value = self._compute()
        return self._value

    def reset(self):
        """Remove computed value."""
        self._value = None


class Data:

    def __init__(self, image, gray_values, thresholds, transition_constant=5):
        self.image = image
        self.gray_values = gray_values
        self.thresholds = thresholds
        self.transition_constant = transition_constant


class Segmentation(Computable):
    """Segmentate image."""

    def __init__(self, data, logistics):
        """Initialize."""
        super().__init__()
        self.data = data
        self.logistics = logistics

    def _compute(self):
        gray_value_diffs = np.diff(self.data.gray_values)
        segmentation = np.zeros_like(self.data.image)
        gray_value_0 = self.data.gray_values[0]

        for gray_value_diff, logistic in zip(gray_value_diffs,
                                             self.logistics()):
            segmentation += (gray_value_diff) * logistic

        return segmentation + gray_value_0


class Logistics(Computable):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def _compute(self):
        gray_value_diffs = np.diff(self.data.gray_values)
        steps = self.data.transition_constant / gray_value_diffs

        logistics = []
        for step, gray_value_diff, threshold in zip(steps,
                                                    gray_value_diffs,
                                                    self.data.thresholds):
            logistics.append(self._logistic(self.data.image - threshold, step))
        return logistics

    @staticmethod
    def _logistic(x_in, step):
        """Compute the logistic function.

        :x_in: Input array
        :steps: Steps in logistic function (1d numpy array)
        :returns: The logisitic function of x_in

        """
        return 1 / (1 + np.exp(-2 * step * x_in))


class Residual(Computable):

    def __init__(self, projector, segmentation, projections):
        super().__init__()
        self.projector = projector
        self.segmentation = segmentation
        self.projections = projections

    def _compute(self):
        return self.projector.FP(self.segmentation()) - self.projections


class ImageGradient(Computable):

    def __init__(self, image, direction=1):
        super().__init__()
        self.image = image
        self.direction = direction

    def _compute(self):
        n_dims = self.image().ndim

        output = []

        for i_dim in range(n_dims):
            gradient = -self.direction * self.image().copy()
            index_min = 1
            index_max = self.image().shape[i_dim]
            if self.direction == -1:
                index_min -= 1
                index_max -= 1

            gradient += self.direction * shift_array(
                self.image(), -1, i_dim)

            # set first or last row/col/slice to zero
            if self.direction == 1:
                slice_axis(gradient, i_dim, index_max, index_max)[:] = 0
            elif self.direction == -1:
                slice_axis(gradient, i_dim, index_min, index_min)[:] = 0

            output.append(gradient)

        return tuple(output)


class CostFun(Computable):

    def __init__(self, cost_fit, cost_reg, sigma):
        super().__init__()
        self.cost_fit = cost_fit
        self.cost_reg = cost_reg
        self.sigma = sigma

    def _compute(self):
        return self.cost_fit() + self.sigma * self.cost_reg()


class CostFit(Computable):

    def __init__(self, residual):
        super().__init__()
        self.residual = residual

    def _compute(self):
        return np.linalg.norm(self.residual().ravel())**2


class CostReg(Computable):

    def __init__(self, gradient_norm):
        super().__init__()
        self.gradient_norm = gradient_norm

    def _compute(self):
        huber_parameter = 1e-4
        l2_mask = self.gradient_norm() < huber_parameter
        l1_mask = self.gradient_norm() >= huber_parameter

        norm = np.zeros_like(self.gradient_norm())
        norm += 1 / (2 * huber_parameter) * self.gradient_norm()**2 * l2_mask
        norm += (self.gradient_norm() - huber_parameter / 2) * l1_mask
        return norm.sum()


class Jacobian(Computable):

    def __init__(self, jacobian_fit, jacobian_reg, sigma):
        super().__init__()
        self.jacobian_fit = jacobian_fit
        self.jacobian_reg = jacobian_reg
        self.sigma = sigma

    def _compute(self):
        return self.jacobian_fit() + self.sigma * self.jacobian_reg()


class JacobianFit(Computable):

    def __init__(self, projector, residual, segmentation_gradient):
        super().__init__()
        self.projector = projector
        self.residual = residual
        self.segmentation_gradient = segmentation_gradient

    def _compute(self):
        return (2 * self.projector.BP(self.residual())
                * self.segmentation_gradient())


class JacobianReg(Computable):

    def __init__(self, segmentation_gradient, gradient_reg):
        super().__init__()
        self.segmentation_gradient = segmentation_gradient
        self.gradient_reg = gradient_reg

    def _compute(self):
        return self.gradient_reg() * self.segmentation_gradient()


class HessianFit(Computable):

    def __init__(self, projector, residual,
                 segmentation_gradient, segmentation_hessian):
        super().__init__()
        self.projector = projector
        self.residual = residual
        self.segmentation_gradient = segmentation_gradient
        self.segmentation_hessian = segmentation_hessian

    def _compute(self):
        return 2 * (self.segmentation_gradient()
                    * (self.projectorBP(self.projector.FP(
                        self.segmentation_gradient()))
                       + (self.projector.BP(self.residual())
                          * self.segmentation_hessian())))


class GradientSums(Computable):

    def __init__(self, image_gradient):
        super().__init__()
        self.image_gradient = image_gradient

    def _compute(self):
        ans = sum(self.image_gradient())
        ans[ans == 0] = 1
        return ans


class GradientSquaredSums(Computable):

    def __init__(self, image_gradient):
        super().__init__()
        self.image_gradient = image_gradient

    def _compute(self):
        ans = sum([grad ** 2 for grad in self.image_gradient()])
        ans[ans == 0] = 1
        return ans


class GradientNorm(Computable):

    def __init__(self, gradient_sum_squares):
        super().__init__()
        self.gradient_sum_squares = gradient_sum_squares

    def _compute(self):
        return np.sqrt(self.gradient_sum_squares())


class GradientReg(Computable):

    def __init__(self, image_gradient, gradient_sums, gradient_norm):
        super().__init__()
        self.image_gradient = image_gradient
        self.gradient_sums = gradient_sums
        self.gradient_norm = gradient_norm

    def _compute(self):
        """Compute gradient of regularization term."""
        huber_parameter = 1e-4
        # determine huber masks
        l2_mask = self.gradient_norm() <= huber_parameter
        l1_mask = self.gradient_norm() > huber_parameter

        gradient_reg = np.zeros_like(self.gradient_norm())
        gradient_reg += -1 / huber_parameter * self.gradient_sums() * l2_mask

        gradient_reg -= self.gradient_sums() / self.gradient_norm() * l1_mask
        for dim, grad in enumerate(self.image_gradient()):
            gradient_reg += 1 / huber_parameter * shift_array(grad * l2_mask,
                                                              1, axis=dim)
            gradient_reg += shift_array(grad / self.gradient_norm() * l1_mask,
                                        1, axis=dim)
        return gradient_reg


class HessianReg(Computable):

    def __init__(self, gradient_reg, segmentation_hessian):
        super().__init__()
        self.gradient_reg = gradient_reg
        self.segmentation_hessian = segmentation_hessian

    def _compute(self):
        """Hessian of regularization term."""
        return self.gradient_reg() * self.segmentation_hessian()


class JacobianDerivative(Computable):

    def __init__(self, segmentation, gradient_sums, gradient_norm):
        super().__init__()
        self.segmentation = segmentation
        self.gradient_sums = gradient_sums
        self.gradient_norm = gradient_norm

    def _compute(self):
        gradient = np.zeros_like(self.segmentation())
        huber_parameter = 1e-4

        gradient += 3 / huber_parameter * l2_mask

        for dim, grad in enumerate(self.image_gradient()):
            gradient += 1 / huber_parameter * shift_array(l2_mask, -1, axis=dim)


class SegmentationGradient(Computable):

    def __init__(self, data, logistics):
        super().__init__()
        self.data = data
        self.logistics = logistics

    def _compute(self):
        """Gradient of soft segmentation."""
        gradient = np.zeros_like(self.data.image)

        for logistic in self.logistics():
            gradient += logistic * (1 - logistic)

        return 2 * self.data.transition_constant * gradient


class SegmentationHesssian(Computable):

    def __init__(self, data, logistics):
        super().__init__()
        self.data = data
        self.logistics = logistics

    def _compute(self):
        gradient = np.zeros_like(self.data.image)
        steps = self.data.transition_constant / np.diff(self.data.gray_values)

        for step, logistic in zip(steps, self.logistics()):
            gradient += (step * logistic
                         * (1 - logistic) * (1 - 2 * logistic))

        return 4 * self.data.transition_constant * gradient


class ComputableArray(Computable):

    def __init__(self, array):
        self._value = array


def slice_axis(array, axis, start, end):
    """Slice array among arbitrary axis."""
    slc = [slice(None)] * len(array.shape)
    slc[axis] = slice(start, end)
    return np.squeeze(array[tuple(slc)])


def add_slice_axis(array_out, array_in, axis, start, end):
    """Add data to array slice.

    A slice from array_out is summated to array_in and
    assigned to the same slice of array_out.
    Assumed is that the shape of the slice corresponds
    to the shape of the array_in.

    :array_out: Array to write into.
    :array_in: Array to copy from.
    :axis: Axis along to slice.
    :start: Slice start index.
    :end: Slice end index.
    """
    slc = [slice(None)] * len(array_out.shape)
    slc[axis] = slice(start, end)
    new_shape = array_out[tuple(slc)].shape
    array_out[tuple(slc)] += array_in.reshape(new_shape)


class GradientGrayValue(Computable):

    def __init__(self, data, logistics):
        super().__init__()
        self.data = data
        self.logistics = logistics

    def _compute(self):
        gray_value_diffs = np.diff(self.data.gray_values)
        n_gray_values = self.data.gray_values.size - 1
        gradient = np.zeros(np.hstack(
            (self.data.image.shape, n_gray_values)))

        # "unpack" data
        k_p = self.data.transition_constant / gray_value_diffs
        K = self.data.transition_constant
        logistics = self.logistics()
        thresholds = self.data.thresholds
        image = self.data.image

        for idx in range(n_gray_values):
            if idx < n_gray_values - 1:
                add_slice_axis(gradient,
                               -(logistics[idx + 1]
                                 * (1 - 2 * k_p[idx + 1]
                                    * (image - thresholds[idx + 1])
                                    * (1 - logistics[idx + 1])
                                 + K * (1 - logistics[idx + 1]))),
                               axis=-1,
                               start=idx,
                               end=idx + 1)

                add_slice_axis(gradient,
                           logistics[idx]
                           * (1 - 2 * k_p[idx]
                              * (image - thresholds[idx])
                              * (1 - logistics[idx])
                              - K * (1-logistics[idx])),
                           axis=-1,
                           start=idx,
                           end=idx + 1)

            return gradient


class JacobianGrayValue(Computable):
    """Compute Jacobian w.r.t. grayvalue."""

    def __init__(self, jacobian_grayvalue_fit, jacobian_grayvalue_reg, sigma):
        super().__init__()
        self.jacobian_grayvalue_fit = jacobian_grayvalue_fit
        self.jacobian_grayvalue_reg = jacobian_grayvalue_reg
        self.sigma = sigma

    def _compute(self):
        return (self.jacobian_grayvalue_fit() + self.sigma
                * self.jacobian_grayvalue_reg())


class JacobianGrayValueFit(Computable):
    """Compute fit jacobian w.r.t. grayvalue"""

    def __init__(self, projector, residual, gradient_grayvalue):
        super().__init__()
        self.projector = projector
        self.residual = residual
        self.gradient_grayvalue = gradient_grayvalue

    def _compute(self):
        # project each gradient separately
        n_gray_values = self.gradient_grayvalue().shape[-1]
        jacobian = np.zeros((n_gray_values,))
        for idx in range(n_gray_values):
            jacobian[idx] = np.sum(
                self.projector.FP(
                    slice_axis(self.gradient_grayvalue(), -1, idx, idx + 1))
                * self.residual())
        return 2 * jacobian


class JacobianGrayValueReg(Computable):
    """Compute regularized jacobian w.r.t. grayvalues."""

    def __init__(self, gradient_reg, gradient_grayvalue):
        super().__init__()
        self.gradient_reg = gradient_reg
        self.gradient_grayvalue = gradient_grayvalue

    def _compute(self):
        n_gray_values = self.gradient_grayvalue().shape[-1]
        jacobian = np.zeros((n_gray_values,))
        for idx in range(n_gray_values):
            jacobian[idx] = np.sum(
                self.gradient_reg()
                * slice_axis(self.gradient_grayvalue(), -1,
                             idx, idx + 1))
        return jacobian


class GradientK(Computable):

    def __init__(self, data, logistics):
        super().__init__()
        self.data = data
        self.logistics = logistics

    def _compute(self):
        gray_value_diffs = np.diff(self.data.gray_values)
        image = self.data.image
        gradient = np.zeros_like(image)

        for gv_diff, logistic in zip(gray_value_diffs, self.logistics()):
            gradient += 2 * image * logistic * (1 - logistic)
        return gradient


class JacobianKFit(Computable):

    def __init__(self, projector, residual, gradient_k):
        super().__init__()
        self.projector = projector
        self.residual = residual
        self.gradient_k = gradient_k

    def _compute(self):
        return 2 * np.sum(
            self.projector.FP(self.gradient_k()) * self.residual())


class JacobianKReg(Computable):

    def __init__(self, gradient_reg, gradient_k):
        super().__init__()
        self.gradient_reg = gradient_reg
        self.gradient_k = gradient_k

    def _compute(self):
        return np.sum(
            self.gradient_reg() * self.gradient_k())


class JacobianK(Computable):

    def __init__(self, jacobian_k_fit, jacobian_k_reg, sigma):
        super().__init__()
        self.jacobian_k_fit = jacobian_k_fit
        self.jacobian_k_reg = jacobian_k_reg
        self.sigma = sigma

    def _compute(self):
        return (self.jacobian_k_fit()
                + self.sigma * self.jacobian_k_reg())


def shift_array(array, shift, axis=0):
    """Shift array along an axis."""

    '''
    pad_size = list(array.shape)
    pad_size[axis] = 1

    if shift >= 0:
        return np.concatenate((np.zeros(pad_size),
                               slice_axis(array, axis, max(0, -shift),
                                          min(array.shape[axis] - shift,
                                              array.shape[axis]))), axis=axis)
    return np.concatenate((slice_axis(array, axis, max(0, -shift),
                                      min(array.shape[axis] - shift,
                                          array.shape[axis])),
                           np.zeros(pad_size)), axis=axis)
    '''
    # OLD code
    shifted_array = np.zeros_like(array)
    slice_axis(shifted_array, axis, max(shift, 0),
               min(array.shape[axis] + shift, array.shape[axis]))[:] = \
        slice_axis(array, axis, max(0, -shift), min(array.shape[axis] - shift,
                                                    array.shape[axis]))[:]
    return shifted_array


class Tvrdart:

    def __init__(self, righthand_side, operator, input_shape,
                 gray_values, thresholds, sigma=1, iter_lim=100,
                 a_scale=None, tv_scale=None, lipschitz_factor=None,
                 initial_solution=None, nonnegative=False, show=True):
        """Setup self.

        :righthand_side: Of the linear system, numpy array.
        :operator: Matrix or linear operator.
        :sigma: Regularization parameter.
        """
        self.righthand_side = righthand_side
        self.operator = operator
        self.sigma = sigma
        self.iter_lim = iter_lim
        self.input_shape = input_shape
        self.nonnegative = nonnegative
        self.gray_values = np.array(gray_values).astype('float')
        if thresholds is not None:
            self.thresholds = np.array(thresholds).astype('float')
        else:
            self.thresholds = self.get_thresholds(self.gray_values)

        self.show = show

        # reconstructed image
        self.image = None
        self.a_scale = a_scale
        self.tv_scale = tv_scale
        self.lipschitz_factor = lipschitz_factor
        self.initial_solution = initial_solution

        # keep copy of current evaluated functions
        self.iteration = None
        self.costfun_l2 = None
        self.jacobfun_l2 = None

    def _costfun(self, param):
        """Cost function."""

        # If K is varied
        # K = param[-1]
        # image = param[:-1].reshape(self.input_shape)

        image = param.reshape(self.input_shape)

        # construct data iterate
        data = Data(image, self.gray_values,
                    self.thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        residual = Residual(self.operator,
                            segmentation, self.righthand_side)
        image_gradient = ImageGradient(segmentation)
        gradient_sum_squares = GradientSquaredSums(image_gradient)
        gradient_norm = GradientNorm(gradient_sum_squares)
        cost_fit = CostFit(residual)
        cost_reg = CostReg(gradient_norm)
        fun_val = CostFun(cost_fit, cost_reg, self.sigma)
        return fun_val()

    def _jacobfun(self, param):
        """Compute Jacobian."""

        # Assume K is fixed, otherwise:
        #  K = param[-1]
        #  image = param[:-1].reshape(self.input_shape)

        image = param.reshape(self.input_shape)
        data = Data(image, self.gray_values,
                    self.thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        residual = Residual(self.operator,
                            segmentation,
                            self.righthand_side)
        image_gradient = ImageGradient(segmentation)
        gradient_sums = GradientSums(image_gradient)
        gradient_sum_squares = GradientSquaredSums(
            image_gradient)
        gradient_norm = GradientNorm(
            gradient_sum_squares)
        segmentation_gradient = SegmentationGradient(
            data, logistics)
        gradient_reg = GradientReg(image_gradient,
                                   gradient_sums,
                                   gradient_norm)
        jacobian_fit = JacobianFit(
            self.operator,
            residual,
            segmentation_gradient)
        jacobian_reg = JacobianReg(
            segmentation_gradient,
            gradient_reg)
        jacobian = Jacobian(jacobian_fit,
                            jacobian_reg,
                            self.sigma)

        return jacobian().ravel()

    @staticmethod
    def get_thresholds(gray_values):
        """Return corresponding thresholds."""
        return gray_values[:-1] + np.diff(gray_values) / 2

    def _costfun_param(self, param):
        """Cost function for parameters only."""
        # unpack
        gray_values = np.hstack((self.gray_values[0], param))
        thresholds = self.get_thresholds(gray_values)
        data = Data(self.image, gray_values, thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        residual = Residual(self.operator,
                            segmentation,
                            self.righthand_side)
        image_gradient = ImageGradient(segmentation)
        gradient_sum_squares = GradientSquaredSums(image_gradient)
        gradient_norm = GradientNorm(gradient_sum_squares)
        cost_fit = CostFit(residual)
        cost_reg = CostReg(gradient_norm)
        fun_val = CostFun(cost_fit, cost_reg, self.sigma)

        self.costfun_l2 = fun_val()
        return self.costfun_l2

    def _jacobfun_param(self, param):
        """Jacobian for paremeters only."""
        gray_values = np.hstack((self.gray_values[0], param))
        thresholds = self.get_thresholds(gray_values)
        data = Data(self.image, gray_values, thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        residual = Residual(self.operator,
                            segmentation,
                            self.righthand_side)
        image_gradient = ImageGradient(segmentation)
        gradient_sums = GradientSums(image_gradient)
        gradient_sum_squares = GradientSquaredSums(image_gradient)
        gradient_norm = GradientNorm(gradient_sum_squares)

        # TODO: Check implementation
        gradient_grayvalue = GradientGrayValue(data, logistics)
        gradient_reg = GradientReg(image_gradient,
                                   gradient_sums,
                                   gradient_norm)
        jacobian_grayvalue_fit = JacobianGrayValueFit(
            self.operator,
            residual,
            gradient_grayvalue)
        jacobian_grayvalue_reg = JacobianGrayValueReg(
            gradient_reg,
            gradient_grayvalue)
        jacobian = JacobianGrayValue(jacobian_grayvalue_fit,
                                     jacobian_grayvalue_reg,
                                     self.sigma)
        jacobian = jacobian().ravel()
        self.jacobfun_l2 = np.linalg.norm(jacobian)
        return jacobian

    def run(self):
        """Run the algorithm."""
        if self.initial_solution is None:
            initial_solution = reconstruct.tvmin(
                self.righthand_side, self.operator,
                iter_lim=100, sigma=10,
                a_scale=self.a_scale,
                tv_scale=self.tv_scale,
                lipschitz_factor=self.lipschitz_factor,
                nonnegative=self.nonnegative, show=True)
        else:
            initial_solution = self.initial_solution

        # scaling
        self.gray_values /= initial_solution.max()
        self.righthand_side /= initial_solution.max()
        initial_solution /= initial_solution.max()
        self.thresholds = self.get_thresholds(self.gray_values)

        if initial_solution.ndim == 3:
            plt.imshow(initial_solution[int(initial_solution.shape[0] / 2),
                                        :, :], cmap='gray')
            plt.title('initial solution')
        else:
            plt.imshow(initial_solution, cmap='gray')
            plt.title('initial solution')
        plt.pause(0.001)

        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.set_title("Reconstruction")
        ax2.set_title("Segmenation")

        if initial_solution.ndim == 3:
            window1 = ax1.imshow(
                initial_solution[int(initial_solution.shape[0] / 2),
                                 :, :], cmap='gray')
            window2 = ax2.imshow(
                initial_solution[int(initial_solution.shape[0] / 2),
                                 :, :], cmap='gray')
        else:
            window1 = ax1.imshow(initial_solution, cmap='gray')
            window2 = ax2.imshow(initial_solution, cmap='gray')
        fig.canvas.draw()

        def callback(current_iterate):
            """Show current iterate."""
            n_pixels = np.prod(self.input_shape)
            image = current_iterate[:n_pixels]
            K = current_iterate[-1]
            data = Data(image,
                        self.gray_values, self.thresholds,
                        transition_constant=K)
            logistics = Logistics(data)
            segmentation = Segmentation(data, logistics)

            # construct data iterate
            if len(self.input_shape) == 2:
                window1.set_data(image.reshape(self.input_shape))
                window1.set_clim(np.quantile(image, 0.05),
                                 np.quantile(image, 0.95))
                window2.set_data(segmentation().reshape(self.input_shape))
                window1.set_clim(np.quantile(image, 0.05),
                                 np.quantile(image, 0.95))
            else:
                window1.set_data(current_iterate.reshape(self.input_shape)[
                    int(self.input_shape[0] / 2), :, :])
                window1.set_clim(np.quantile(current_iterate, 0.05),
                                 np.quantile(current_iterate, 0.95))
                window2.set_data(segmentation().reshape(self.input_shape)[
                    int(self.input_shape[0] / 2), :, :])
                window1.set_clim(np.quantile(current_iterate, 0.05),
                                 np.quantile(current_iterate, 0.95))

            fig.canvas.draw()
            plt.pause(0.000001)

        self.iteration = 0
        if self.show:
            print("Iteration\t||costfun||\t||jacob||")
            print("========================================================================")
        output = scipy.optimize.minimize(
            self._costfun,
            initial_solution.ravel(),
            method='L-BFGS-B',
            jac=self._jacobfun,
            options={'maxiter': self.iter_lim, 'disp': True},
            callback=callback)

        reconstruction = output.x.reshape(self.input_shape)
        data = Data(reconstruction, self.gray_values, self.thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        return segmentation()

    def run_all(self, outer_iterlim=50):
        """Run full optimization algorithm."""
        print("Initial Chambolle-Pock reconstruction...")

        if self.initial_solution is None:
            self.image = reconstruct.tvmin(
                self.righthand_side, self.operator,
                iter_lim=100, sigma=10,
                a_scale=self.a_scale,
                tv_scale=self.tv_scale,
                lipschitz_factor=self.lipschitz_factor,
                nonnegative=self.nonnegative, show=True)
        else:
            self.image = self.initial_solution

        # rescale parameters
        self.righthand_side /= self.image.max()
        self.image /= self.image.max()

        if np.isnan(self.gray_values[0]):
            print("Estimating gray values using Gaussian mixture.")

            # estimate gray values
            sorted_pixels = np.sort(self.image.ravel())

            # fit gaussians
            mixture = sklearn.mixture.GaussianMixture(
                n_components=len(self.gray_values),
                covariance_type='diag')
            scaler = sklearn.preprocessing.RobustScaler()
            x_train = scaler.fit_transform(sorted_pixels[::4].reshape((-1, 1)))
            mixture.fit(x_train)
            means = scaler.inverse_transform(mixture.means_).ravel()
            covars = scaler.inverse_transform(mixture.covariances_).ravel()

            # set gray values
            gv_sort_idx = np.argsort(means)
            self.gray_values = means[gv_sort_idx]
            self.gray_values[0] -= np.sqrt(covars[gv_sort_idx][0])

            print("Gray value estimate:", self.gray_values)

        self.gray_values /= self.image.max()
        self.thresholds = self.get_thresholds(self.gray_values)

        if self.image.ndim == 3:
            plt.imshow(self.image[int(self.image.shape[0] / 3),
                                  :, :], cmap='gray')
        else:
            plt.imshow(self.image, cmap='gray')
        plt.title("Initial reconstruction")
        plt.pause(0.001)

        fig = plt.figure()
        ax1 = plt.subplot(1, 2, 1)
        ax2 = plt.subplot(1, 2, 2)
        ax1.set_title("Reconstruction")
        ax2.set_title("Segmentation")

        if self.image.ndim == 3:
            window1 = ax1.imshow(
                self.image[int(self.image.shape[0] / 2),
                           :, :], cmap='gray')
            window2 = ax2.imshow(
                self.image[int(self.image.shape[0] / 2),
                           :, :], cmap='gray')
        else:
            window1 = ax1.imshow(self.image, cmap='gray')
            window2 = ax2.imshow(self.image, cmap='gray')
        fig.canvas.draw()

        def callback(current_iterate):
            """Show current iterate."""
            data = Data(current_iterate.reshape(self.input_shape),
                        self.gray_values, self.thresholds)
            logistics = Logistics(data)
            segmentation = Segmentation(data, logistics)
            if len(self.input_shape) == 2:
                window1.set_data(current_iterate.reshape(self.input_shape))
                window1.set_clim(np.quantile(current_iterate, 0.05),
                                 np.quantile(current_iterate, 0.95))
                segm = segmentation().reshape(self.input_shape)
                window2.set_data(segm)
                window2.set_clim(np.quantile(segm, 0.05),
                                 np.quantile(segm, 0.95))
            else:
                window1.set_data(current_iterate.reshape(self.input_shape)[
                    int(self.input_shape[0] / 2), :, :])
                window2.set_data(segmentation().reshape(self.input_shape)[
                    int(self.input_shape[0] / 2), :, :])

            if self.iteration and self.costfun and self.jacobfun:
                print(f"{self.iteration}\t{self.costfun_l2:e}\t{self.jacobfun_l2:e}") 
            fig.canvas.draw()
            plt.pause(0.000001)

        def callback_param(current_iterate):
            """Show current iterate."""
            print("Gray values:", current_iterate)

        for outer_idx in range(outer_iterlim):
            print("Outer iteration:", outer_idx)

            self.iteration = 0
            if self.show:
                print("Iteration\t||costfun||\t||jacob||")
                print("========================================================================")

            # get a reconstruction
            output = scipy.optimize.minimize(
                self._costfun,
                self.image.ravel(),
                method='L-BFGS-B',
                jac=self._jacobfun,
                options={'maxiter': self.iter_lim, 'disp': True},
                callback=callback)
            # set output
            self.image = output['x'].reshape(self.input_shape).copy()

            # optimize gray values
            output = scipy.optimize.minimize(
                self._costfun_param,
                self.gray_values[1:],
                method='L-BFGS-B',
                jac=self._jacobfun_param,
                options={'maxiter': self.iter_lim, 'disp': True},
                callback=callback_param)

            # set output [keep lower gray value fixed]
            self.gray_values[1:] = output['x'].copy()
            print("new grayvalues:", self.gray_values)
            print("new thresholds:", self.thresholds)

            # update thresholds
            self.thresholds = self.get_thresholds(self.gray_values)

        data = Data(self.image, self.gray_values, self.thresholds)
        logistics = Logistics(data)
        segmentation = Segmentation(data, logistics)
        return segmentation()
