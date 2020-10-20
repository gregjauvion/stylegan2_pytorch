
import numpy as np


def linear_interpolation(v0, v1, nb_values):

    t_array = np.arange(0, 1 + 1e-10, 1./nb_values)
    return np.array([(1 - t) * v0 + t * v1 for t in t_array])


def slerp_interpolation(v0, v1, nb_values):
    """ Spherical linear interpolation
    Implementation found here https://github.com/soumith/dcgan.torch/issues/14
    """

    t_array = np.arange(0, 1 + 1e-10, 1./nb_values)
    omega = np.arccos(np.clip(np.dot(v0 / np.linalg.norm(v0), v1 / np.linalg.norm(v1)), -1, 1))
    so = np.sin(omega)

    if so == 0:
        return (1 - t_array) * v0 + t_array * v1

    all_f0, all_f1 = np.sin((1 - t_array) * omega) / so, np.sin(t_array * omega) / so
    return np.array([f0 * v0 + f1 * v1 for f0, f1 in zip(all_f0, all_f1)])


if __name__=='__main__':

    import matplotlib.pyplot as plt

    v0, v1 = np.array([1, 1, 0]), np.array([0, 1, 2])
    linear = linear_interpolation(v0, v1, 20)
    slerp = slerp_interpolation(v0, v1, 20)

    plt.plot(np.sqrt(np.sum(linear**2, axis=1)), label='Norm (linear interpolation)')
    plt.plot(np.sqrt(np.sum(slerp ** 2, axis=1)), label='Norm (slerp interpolation)')
    plt.legend() ; plt.show()


    v0, v1 = np.array([1, 0]), np.array([0, 2])
    linear = linear_interpolation(v0, v1, 20)
    slerp = slerp_interpolation(v0, v1, 20)

    plt.scatter(v0, v1)
    plt.plot([i[0] for i in linear], [i[1] for i in linear])
    plt.plot([i[0] for i in slerp], [i[1] for i in slerp])
    plt.grid() ; plt.show()
