import matplotlib.pyplot as plt
import numpy as np


def calculate_ellipsis(P1, i1, P2, i2, switcher=True):
    p = 0.95
    s = - 2 * np.log(1 - p)

    if switcher:
        D0, V0 = np.linalg.eig(P1[:, :, i1] * s)
        D1, V1 = np.linalg.eig(P2[:, :, i2] * s)
    else:
        D0, V0 = np.linalg.eig(P1[:, :, i1] * s)
        D1, V1 = np.linalg.eig(P2 * s)

    vel = np.linspace(-np.pi, np.pi)

    a = (V0 * np.sqrt(D0)).dot(np.array([[np.cos(vel)], [np.sin(vel)]]).reshape((2, len(vel))))
    b = (V1 * np.sqrt(D1)).dot(np.array([[np.cos(vel)], [np.sin(vel)]]).reshape((2, len(vel))))

    return a, b

def plot_prediction_chart(x, x_real, a, b, minx, maxx, miny, maxy):
    plt.figure(0)
    plt.plot(a[0, :] + x[0, 0], a[1, :] + x[1, 0], 'k-')
    plt.plot(b[0, :] + x[0, 1], b[1, :] + x[1, 1], 'r--')

    s1 = plt.scatter(x[0, 0], x[1, 0], c='k')
    s2 = plt.scatter(x[0, 1], x[1, 1], c='r')
    s3 = plt.scatter(x_real[0], x_real[1], c='b', marker='x')

    plt.legend([s1, s2, s3], ['Initial estimate', 'Predicted state', 'Real'], loc='best')
    plt.title('Prediction')
    plt.xlabel('position (x)')
    plt.ylabel('velocity (v)')
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])
    plt.show()

def plot_correction_chart(x, x_predicted, x_real, vel, a, b, minx, maxx, miny, maxy):
    plt.figure(1)
    plt.plot(a[0, :] + x[0, 1], a[1, :] + x[1, 1], 'k-')
    plt.plot(b[0, :] + x_predicted[0], b[1, :] + x_predicted[1], 'r--')

    plt.scatter(x_real[0], x_real[1], c='b', marker='x')
    plt.scatter(x[0, 1], x[1, 1], c='k')
    plt.scatter(x_predicted[0], x_predicted[1], c='r')

    omega = np.ones((len(vel), 1)) * x[1, 1]
    plt.plot(np.linspace(minx, maxx), omega, 'g--')

    plt.legend(['Updated estimate', 'Predicted estimate', '$\Omega$*', 'Real'])
    plt.title('Correction')
    plt.xlabel('position (x)')
    plt.ylabel('velocity (v)')
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])
    plt.show()

def plot_iteration_chart(k, x, a, b, minx, maxx, miny, maxy):
    plt.figure(k)
    plt.plot(a[0, :] + x[0, k - 1], a[1, :] + x[1, k - 1], 'k-')
    plt.plot(b[0, :] + x[0, k], b[1, :] + x[1, k], 'r--')

    plt.scatter(x[0, k - 1], x[1, k - 1], c='k')
    plt.scatter(x[0, k], x[1, k], c='r')

    plt.legend(['Initial estimate', 'Computed estimate'])
    plt.title('Iteration ' + str(k))
    plt.xlabel('position (x)')
    plt.ylabel('velocity (v)')
    plt.xlim([minx, maxx])
    plt.ylim([miny, maxy])

    plt.show()

def calculate_gain_and_error(k, x, y, H, P, W):
    K = P[:, :, k].dot(H.reshape((2, 1))) / (H.dot(P[:, :, k]).dot(H.reshape((2, 1))) + W)
    nu = y[k] - H.dot(x[:, k - 1])

    return K, nu
