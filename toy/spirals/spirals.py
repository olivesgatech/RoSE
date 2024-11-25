#!/usr/bin/env python
import argparse
import math
import numpy as np
import matplotlib.pyplot as plt


def rotate_point(point, angle):
    """Rotate two point by an angle.

    Parameters
    ----------
    point: 2d numpy array
        The coordinate to rotate.
    angle: float
        The angle of rotation of the point, in degrees.

    Returns
    -------
    2d numpy array
        Rotated point.
    """
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    rotated_point = rotation_matrix.dot(point)
    return rotated_point


def generate_spiral(samples, start, end, angle, noise):
    """Generate a spiral of points.

    Given a starting end, an end angle and a noise factor, generate a spiral of points along
    an arc.

    Parameters
    ----------
    samples: int
        Number of points to generate.
    start: float
        The starting angle of the spiral in degrees.
    end: float
        The end angle at which to rotate the points, in degrees.
    angle: float
        Angle of rotation in degrees.
    noise: float
        The noisyness of the points inside the spirals. Needs to be less than 1.
    """
    # Generate points from the square root of random data inside an uniform distribution on [0, 1).
    points = math.radians(start) + np.sqrt(np.random.rand(samples, 1)) * math.radians(end)

    # Apply a rotation to the points.
    rotated_x_axis = np.cos(points) * points + np.random.rand(samples, 1) * noise
    rotated_y_axis = np.sin(points) * points + np.random.rand(samples, 1) * noise

    # Stack the vectors inside a samples x 2 matrix.
    rotated_points = np.column_stack((rotated_x_axis, rotated_y_axis))
    return np.apply_along_axis(rotate_point, 1, rotated_points, math.radians(angle))


def main():
    count = 500
    arms = 8
    angle = 360 / arms
    auto_angle = True
    start = 0
    end = 360
    noise = 0.3

    # Create a list of the angles at which to rotate the arms.
    # Either we find the angles automatically by dividing by the number of arms
    # Or we just use the angle given by the user.
    classes = np.empty((0, 3))
    angles = [((360 / arms) if auto_angle else angle) * i for i in range(arms)]

    for i, angle in enumerate(angles):
        points = generate_spiral(count, start, end, angle, noise)
        plt.scatter(points[:, 0], points[:, 1])
        classified_points = np.hstack((points, np.full((count, 1), i)))
        classes = np.concatenate((classes, classified_points))
    plt.show()



if __name__ == '__main__':
    main()