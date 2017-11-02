import numpy as np

def test_bin_binary():
    """
    Test that corals in the population only contain values in {0, 1}
    """
    from ..reef_initialization import bin_binary

    M, N, r0, L = 2, 2, 0.6, 8
    REEF, REEFpob = bin_binary(M, N, r0, L)
    assert set(REEFpob.ravel()) == {0, 1}

def test_disc_equalRange():
    """
    Test that corals in population contain values specified in the grid
    """
    from ..reef_initialization import disc_equalRange

    M, N, r0, L = 2, 2, 0.6, 8
    grid = {'x': [2, 10]}      # Discrete values between 2 and 10

    REEF, REEFpob = disc_equalRange(M, N, r0, L, param_grid=grid)
    p = sum(REEFpob[np.where(REEFpob!=0)]<grid['x'][0]) + sum(REEFpob[np.where(REEFpob!=0)]>grid['x'][1])
    assert p == 0
