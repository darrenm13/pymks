import numpy as np
from .cahn_hilliard_simulation import CahnHilliardSimulation
from .microstructure_generator import MicrostructureGenerator
from pymks import DiscreteIndicatorBasis, MKSRegressionModel
from .load_images import load_ti64

__all__ = ['make_delta_microstructures', 'make_elastic_FE_strain_delta',
           'make_elastic_FE_strain_random', 'make_cahn_hilliard',
           'make_microstructure', 'make_checkerboard_microstructure',
           'make_elastic_stress_random', 'load_ti64']
           
def load_ti64()
           """future input arguments will be the images from a Georgia Tech database, but for now they are
           inputs from flickr
           
           Z = np.zeros((63, 376, 500))

    pic_array = (
     'https://farm4.staticflickr.com/3913/15225473936_b0e7b83734_z_d.jpg',
     'https://farm6.staticflickr.com/5588/15245381241_700ea05db1_z_d.jpg',
     'https://farm4.staticflickr.com/3877/15245381211_c2dce16a2f_z_d.jpg',
     'https://farm4.staticflickr.com/3921/15061717289_53c8e486d4_z_d.jpg',
     'https://farm4.staticflickr.com/3890/15061918228_227a6d9b74_z_d.jpg',
     'https://farm4.staticflickr.com/3889/15245380771_2eca66bea7_z_d.jpg',
     'https://farm6.staticflickr.com/5565/15245380551_20e4e141c7_z_d.jpg',
     'https://farm6.staticflickr.com/5570/15061909607_8af8afe624_z_d.jpg',
     'https://farm6.staticflickr.com/5584/15061796840_c7a73c26d0_z_d.jpg',
     'https://farm4.staticflickr.com/3921/15061909357_0a4536ed1e_z_d.jpg',
     'https://farm6.staticflickr.com/5593/15061909187_ece1b3ea06_z_d.jpg',
     'https://farm4.staticflickr.com/3897/15061796360_d771a63e4a_z_d.jpg',
     'https://farm4.staticflickr.com/3914/15061715939_1f571a913c_z_d.jpg',
     'https://farm4.staticflickr.com/3908/15225472466_58d6ca44b4_z_d.jpg',
     'https://farm6.staticflickr.com/5591/15248096422_4a63a3fa55_z_d.jpg',
     'https://farm4.staticflickr.com/3921/15245379711_c73333892a_z_d.jpg',
     'https://farm6.staticflickr.com/5560/15061796010_c6feb5714e_z_d.jpg',
     'https://farm4.staticflickr.com/3913/15245379481_fa3286bfc6_z_d.jpg',
     'https://farm4.staticflickr.com/3898/15061795770_1fd51d9383_z_d.jpg',
     'https://farm4.staticflickr.com/3898/15061795770_1fd51d9383_z_d.jpg',
     'https://farm6.staticflickr.com/5557/15225471866_7a9b298846_z_d.jpg',
     'https://farm4.staticflickr.com/3838/15248470565_56b8881cb2_z_d.jpg',
     'https://farm4.staticflickr.com/3850/15245379231_919d661b53_z_d.jpg',
     'https://farm4.staticflickr.com/3904/15248470295_971c077b40_z_d.jpg',
     'https://farm6.staticflickr.com/5594/15245378771_1cd67c16e6_z_d.jpg',
     'https://farm4.staticflickr.com/3901/15061714479_63d4e52693_z_d.jpg',
     'https://farm6.staticflickr.com/5594/15248469785_471dd12a5f_z_d.jpg',
     'https://farm4.staticflickr.com/3837/15061915848_46584e8279_z_d.jpg',
     'https://farm4.staticflickr.com/3911/15061915828_b3e9aa67ed_z_d.jpg',
     'https://farm4.staticflickr.com/3849/15061794910_289dde4d8c_z_d.jpg',
     'https://farm4.staticflickr.com/3857/15248469535_d08cef2e36_z_d.jpg',
     'https://farm6.staticflickr.com/5584/15061714239_e35509b83f_z_d.jpg',
     'https://farm4.staticflickr.com/3835/15061907407_f0cd26759f_z_d.jpg',
     'https://farm4.staticflickr.com/3872/15245378111_3a172eaab5_z_d.jpg',
     'https://farm4.staticflickr.com/3899/15248094622_ccd40138ba_z_d.jpg',
     'https://farm4.staticflickr.com/3869/15061794330_2c9cca24a9_z_d.jpg',
     'https://farm6.staticflickr.com/5583/15061794300_ed1680901f_z_d.jpg',
     'https://farm6.staticflickr.com/5583/15248469175_2039c3b0ef_z_d.jpg',
     'https://farm4.staticflickr.com/3858/15248469135_51d898e620_z_d.jpg',
     'https://farm4.staticflickr.com/3863/15245377731_b3bf1ca851_z_d.jpg',
     'https://farm4.staticflickr.com/3835/15225470036_b3967a6f5f_z_d.jpg',
     'https://farm4.staticflickr.com/3901/15061906677_b1c39a5254_z_d.jpg',
     'https://farm6.staticflickr.com/5570/15061906557_3d36711e17_z_d.jpg',
     'https://farm6.staticflickr.com/5582/15061906367_f4c4a586b6_z_d.jpg',
     'https://farm4.staticflickr.com/3868/15061793360_1183a45351_z_d.jpg',
     'https://farm4.staticflickr.com/3908/15225469326_f7202c5cb9_z_d.jpg',
     'https://farm4.staticflickr.com/3925/15225469176_10f65dc211_z_d.jpg',
     'https://farm4.staticflickr.com/3916/15061712399_1a2d451ce6_z_d.jpg',
     'https://farm4.staticflickr.com/3915/15061905357_4907b07388_z_d.jpg',
     'https://farm4.staticflickr.com/3876/15061792370_522a81fa15_z_d.jpg',
     'https://farm4.staticflickr.com/3904/15245376151_c7043cb8ba_z_d.jpg',
     'https://farm6.staticflickr.com/5586/15061905067_4f545e4f92_z_d.jpg',
     'https://farm6.staticflickr.com/5595/15061792190_c425e4d2cc_z_d.jpg',
     'https://farm4.staticflickr.com/3904/15061711949_8083bba099_z_d.jpg',
     'https://farm4.staticflickr.com/3908/15248092302_3a94178821_z_d.jpg',
     'https://farm6.staticflickr.com/5579/15061912968_eab32bc312_z_d.jpg',
     'https://farm4.staticflickr.com/3897/15248466815_958b48c000_z_d.jpg',
     'https://farm4.staticflickr.com/3897/15248466765_2354142b9a_z_d.jpg',
     'https://farm4.staticflickr.com/3911/15245375521_f1e14e1ddc_z_d.jpg',
     'https://farm6.staticflickr.com/5565/15248466485_78e3a05904_z_d.jpg')
    for x in range(0, 59):

        X = io.imread(pic_array[x])[None]
        X = np.round(X / float(np.max(X))).astype(int)
        Z[x] = X


def make_elastic_FE_strain_delta(elastic_modulus=(100, 150),
                                 poissons_ratio=(0.3, 0.3),
                                 size=(21, 21), macro_strain=0.01):
    """Generate delta microstructures and responses

    Simple interface to generate delta microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The length of `elastic_modulus` and
    `poissons_ratio` indicates the number of phases in the
    microstructure. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    Args:
        elastic_modulus (list, optional): elastic moduli for the phases
        poissons_ratio (list, optional): Poisson's ratios for the phases
        size (tuple, optional): size of the microstructure
        macro_strain (float, optional): Scalar for macroscopic strain applied
        strain_index (int, optional): interger value to return a particular
            strain field. 0 returns exx, 1 returns eyy, etc. To return all
            strain fields set strain_index equal to slice(None).

    Returns:
        tuple containing delta microstructures and their strain fields

    Example

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elastic_FE_strain_delta(elastic_modulus=elastic_modulus,
    ...                                     poissons_ratio=poissons_ratio,
    ...                                     size=(5, 5))

    `X` is the delta microstructures, and `y` is the
    strain response fields.

    """
    from .elastic_FE_simulation import ElasticFESimulation

    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = make_delta_microstructures(len(elastic_modulus), size=size)
    FEsim.run(X)
    return X, FEsim.response


def make_delta_microstructures(n_phases=2, size=(21, 21)):
    """Constructs delta microstructures

    Constructs delta microstructures for an arbitrary number of phases
    given the size of the domain.

    Args:
        n_phases (int, optional): number of phases
        size (tuple, optional): dimension of microstructure

    Returns:
        delta microstructures for the system of shape
        (n_samples, n_x, ...)

    Example

    >>> X = np.array([[[[0, 0, 0],
    ...                 [0, 0, 0],
    ...                 [0, 0, 0]],
    ...                [[0, 0, 0],
    ...                 [0, 1, 0],
    ...                 [0, 0, 0]],
    ...                [[0, 0, 0],
    ...                 [0, 0, 0],
    ...                 [0, 0, 0]]],
    ...               [[[1, 1, 1],
    ...                 [1, 1, 1],
    ...                 [1, 1, 1]],
    ...                [[1, 1, 1],
    ...                 [1, 0, 1],
    ...                 [1, 1, 1]],
    ...                [[1, 1, 1],
    ...                 [1, 1, 1],
    ...                 [1, 1, 1]]]])

    >>> assert(np.allclose(X, make_delta_microstructures(2, size=(3, 3, 3))))

    """
    shape = (n_phases, n_phases) + size
    center = tuple((np.array(size) - 1) / 2)
    X = np.zeros(shape=shape, dtype=int)
    X[:] = np.arange(n_phases)[(slice(None), None) + (None,) * len(size)]
    X[(slice(None), slice(None)) + center] = np.arange(n_phases)
    mask = ~np.identity(n_phases, dtype=bool)
    return X[mask]


def make_elastic_FE_strain_random(n_samples=1, elastic_modulus=(100, 150),
                                  poissons_ratio=(0.3, 0.3), size=(21, 21),
                                  macro_strain=0.01):
    """Generate random microstructures and responses

    Simple interface to generate random microstructures and their
    strain response fields that can be used for the fit method in the
    `MKSRegressionModel`. The following example is or a two phase
    microstructure with dimensions of `(5, 5)`.

    Args:
        elastic_modulus (list, optional): elastic moduli for the phases
        poissons_ratio (list, optional): Poisson's ratios for the phases
        size (tuple, optional): size of the microstructure
        macro_strain (float, optional): Scalar for macroscopic strain applied
        strain_index (int, optional): interger value to return a particular
            strain field. 0 returns exx, 1 returns eyy, etc. To return all
            strain fields set strain_index equal to slice(None).

    Returns:
         tuple containing delta microstructures and their strain fields

    Example

    >>> elastic_modulus = (1., 2.)
    >>> poissons_ratio = (0.3, 0.3)
    >>> X, y = make_elastic_FE_strain_random(n_samples=1,
    ...                                      elastic_modulus=elastic_modulus,
    ...                                      poissons_ratio=poissons_ratio,
    ...                                      size=(5, 5))

    `X` is the delta microstructures, and `y` is the
    strain response fields.

    """
    from .elastic_FE_simulation import ElasticFESimulation

    FEsim = ElasticFESimulation(elastic_modulus=elastic_modulus,
                                poissons_ratio=poissons_ratio,
                                macro_strain=macro_strain)

    X = np.random.randint(len(elastic_modulus), size=((n_samples, ) + size))
    FEsim.run(X)
    return X, FEsim.response


def make_cahn_hilliard(n_samples=1, size=(21, 21), dx=0.25, width=1.,
                       dt=0.001, n_steps=1):
    """Generate microstructures and responses for Cahn-Hilliard.
    Simple interface to generate random concentration fields and their
    evolution after one time step that can be used for the fit method in the
    `MKSRegressionModel`.  The following example is or a two phase
    microstructure with dimensions of `(6, 6)`.

    Args:
        n_samples (int, optional): number of microstructure samples
        size (tuple, optional): size of the microstructure
        dx (float, optional): grid spacing
        dt (float, optional): timpe step size
        width (float, optional): interface width between phases.
        n_steps (int, optional): number of time steps used

    Returns:
        Array representing the microstructures at n_steps ahead of 'X'

    Example

    >>> X, y = make_cahn_hilliard(n_samples=1, size=(6, 6))

    `X` is the initial concentration fields, and `y` is the
    strain response fields (the concentration after one time step).

    """
    CHsim = CahnHilliardSimulation(dx=dx, dt=dt, gamma=width ** 2)

    X0 = 2 * np.random.random((n_samples,) + size) - 1
    X = X0.copy()
    for ii in range(n_steps):
        CHsim.run(X)
        X = CHsim.response
    return X0, X


def make_microstructure(n_samples=10, size=(101, 101), n_phases=2,
                        grain_size=(33, 14), seed=10):
    """
    Constructs microstructures for an arbitrary number of phases
    given the size of the domain, and relative grain size.

    Args:
        n_samples (int, optional): number of samples
        size (tuple, optional): dimension of microstructure
        n_phases (int, optional): number of phases
        grain_size (tuple, optional): effective dimensions of grains
        seed (int, optional): seed for random number microstructureGenerator

    Returns:
        microstructures for the system of shape (n_samples, n_x, ...)

    Example

    >>> n_samples, n_phases = 1, 2
    >>> size, grain_size = (3, 3), (1, 1)
    >>> Xtest = np.array([[[0, 1, 0],
    ...                [0, 0, 0],
    ...                [0, 1, 1]]])
    >>> X = make_microstructure(n_samples=n_samples, size=size,
    ...                         n_phases=n_phases, grain_size=grain_size,
    ...                         seed=0)

    >>> assert(np.allclose(X, Xtest))

    """
    MS = MicrostructureGenerator(n_samples=n_samples, size=size,
                                 n_phases=n_phases, grain_size=grain_size,
                                 seed=seed)
    return MS.generate()


def make_checkerboard_microstructure(square_size, n_squares):
    """
    Constructs a checkerboard_microstructure with the `square_size` by
    `square_size` size squares and on a `n_squares` by `n_squares`

    Args:
        square_size (int): length of the side of one square in the
            checkerboard.
        n_squares (int): number of squares along on size of the checkerboard.

    Returns:
        checkerboard microstructure with shape of (1, square_size * n_squares,
        square_size * n_squares)

    Example

    >>> square_size, n_squares = 2, 2
    >>> Xtest = np.array([[[0, 0, 1, 1],
    ...                    [0, 0, 1, 1],
    ...                    [1, 1, 0, 0],
    ...                    [1, 1, 0, 0]]])
    >>> X = make_checkerboard_microstructure(square_size, n_squares)
    >>> assert(np.allclose(X, Xtest))

    """

    L = n_squares * square_size
    X = np.ones((2 * square_size, 2 * square_size), dtype=int)
    X[:square_size, :square_size] = 0
    X[square_size:, square_size:] = 0
    return np.tile(X, ((n_squares + 1) / 2, (n_squares + 1) / 2))[None, :L, :L]


def make_elastic_stress_random(n_samples=[10, 10], elastic_modulus=(100, 150),
                               poissons_ratio=(0.3, 0.3), size=(21, 21),
                               macro_strain=0.01, grain_size=[(3, 3), (9, 9)],
                               seed=10):
    """
    Generates microstructures and their macroscopic stress values for an
    applied macroscopic strain.

    Args:
        n_samples (int, optional): number of samples
        elastic_modulus (tuple, optional): list of elastic moduli for the
            different phases.
        poissons_ratio (tuple, optional): list of poisson's ratio values for
            the phases.
        size (tuple, optional): size of the microstructures
        macro_strain (tuple, optional): macroscopic strain applied to the
            sample.
        grain_size (tuple, optional): effective dimensions of grains
        seed (int, optional): seed for random number generator

    Returns:
        array of microstructures with dimensions (n_samples, n_x, ...) and
        effective stress values

    Example

    >>> X, y = make_elastic_stress_random(n_samples=1, elastic_modulus=(1, 1),
    ...                                   poissons_ratio=(1, 1),
    ...                                   grain_size=(3, 3), macro_strain=1.0)
    >>> assert np.allclose(y, np.ones(y.shape))
    >>> X, y = make_elastic_stress_random(n_samples=1, grain_size=(1, 1),
    ...                                   elastic_modulus=(100, 200),
    ...                                   size=(2, 2), poissons_ratio=(1, 3),
    ...                                   macro_strain=1., seed=3)
    >>> X_result = np.array([[[1, 1],
    ...                       [0, 1]]])
    >>> assert np.allclose(X, X_result)
    >>> assert float(np.round(y, decimals=5)[0]) == 228.74696
    >>> X, y = make_elastic_stress_random(n_samples=1, grain_size=(1, 1, 1),
    ...                                   elastic_modulus=(100, 200),
    ...                                   poissons_ratio=(1, 3),  seed=3,
    ...                                   macro_strain=1., size=(2, 2, 2))
    >>> X_result = np.array([[[1, 1],
    ...                       [0, 0]],
    ...                      [[1, 1],
    ...                       [0, 0]]])
    >>> assert np.allclose(X, X_result)
    >>> assert np.round(y[0]).astype(int) == 150

    """
    if not isinstance(grain_size[0], (list, tuple, np.ndarray)):
        grain_size = (grain_size,)
    if not isinstance(n_samples, (list, tuple, np.ndarray)):
        n_samples = (n_samples,)
    if not isinstance(size, (list, tuple, np.ndarray)) or len(size) > 3:
        raise RuntimeError('size must have length of 2 or 3')
    [RuntimeError('dimensions of size and grain_size are not the same.')
     for grains in grain_size if len(size) != len(grains)]
    if len(elastic_modulus) != len(poissons_ratio):
        raise RuntimeError('length of elastic_modulus and poissons_ratio are \
                           not the same.')
    X_cal, y_cal = make_elastic_FE_strain_delta(elastic_modulus,
                                                poissons_ratio, size,
                                                macro_strain)
    n_states = len(elastic_modulus)
    basis = DiscreteIndicatorBasis(n_states)
    model = MKSRegressionModel(basis=basis)
    model.fit(X_cal, y_cal)
    X = np.concatenate([make_microstructure(n_samples=sample, size=size,
                                            n_phases=n_states,
                                            grain_size=gs, seed=seed) for gs,
                        sample in zip(grain_size, n_samples)])
    X_ = basis.discretize(X)
    index = tuple([None for i in range(len(size) + 1)]) + (slice(None),)
    modulus = np.sum(X_ * np.array(elastic_modulus)[index], axis=-1)
    y_stress = model.predict(X) * modulus
    return X, np.average(y_stress.reshape(np.sum(n_samples), y_stress[0].size),
                         axis=1)
