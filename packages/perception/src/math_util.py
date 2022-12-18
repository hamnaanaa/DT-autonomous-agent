import numpy as np


def rolling_average(new_point, points):
    if new_point is not None:
        points[1:] = points[:-1]
        points[0] = new_point
    return np.average(points, axis=0, weights=np.exp(np.arange(len(points), 0, -1))), points


def weighted_average(a_weight, a, b_weight, b):
    return (a_weight * a + b_weight * b) / (a_weight + b_weight)


# Test-Main.
if __name__ == "__main__":
    midpoints = np.ones((10, 2))
    avg, midpoints = rolling_average(np.array([10, 10]), midpoints)
    print(avg, midpoints)
    avg, midpoints = rolling_average(np.array([10, 10]), midpoints)
    print(avg, midpoints)
