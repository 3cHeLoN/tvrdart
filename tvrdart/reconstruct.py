"""
Reconstruction algorithm wrapper.
"""
from . import operators, solvers


def tvmin(projections, projector,
          iter_lim=50, sigma=1,
          a_scale=None, tv_scale=None,
          lipschitz_factor=None, show=False,
          nonnegative=False):
    """Utility function for calling tvmin."""
    # 2D
    if len(projector.vshape) == 2:
        tv_operator = operators.OpTV2D(projector.vshape[1],
                                       projector.vshape[0])
    # 3D
    elif len(projector.vshape) == 3:
        tv_operator = operators.OpTV3D(projector.vshape[1],
                                       projector.vshape[0],
                                       projector.vshape[2])

    cp_alg = solvers.ChambollePock(projector,
                                   tv_operator,
                                   projections.ravel(),
                                   tv_weight=sigma,
                                   nonnegative=nonnegative,
                                   max_iter=iter_lim,
                                   a_scale=a_scale,
                                   tv_scale=tv_scale,
                                   lipschitz_factor=lipschitz_factor,
                                   show=show)
    volume = cp_alg.run()
    return volume.reshape(projector.vshape)
