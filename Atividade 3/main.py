import cv2
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def attractive_potential(x, y, goal_x, goal_y):
    return 0.5 * ATTR_GAIN * np.hypot(x - goal_x, y - goal_y)


def repulsive_potential(x, y, obstacles_x_components, obstacles_y_components):
    min_index = -1
    min_distance = float("inf")

    for i in range(len(obstacles_x_components)):
        d = np.hypot(x - obstacles_x_components[i], y - obstacles_y_components[i])
        if min_distance >= d:
            min_distance = d
            min_index = i

    Dq = np.hypot(x - obstacles_x_components[min_index], y - obstacles_y_components[min_index])

    if Dq <= ROBOT_RADIUS:
        if Dq <= 0.1:
            Dq = 0.1

        return 0.5 * REPU_GAIN * (1.0 / Dq - 1.0 / ROBOT_RADIUS) ** 2
    else:
        return 0.0


def add_obstacle(kernel_size, x, y):
    sx = kernel_size[0]
    sy = kernel_size[1]

    for i in range(x, x + sx):
        for j in range(y, y + sy):
            obstacles_x_components.append(i)
            obstacles_y_components.append(j)


def get_minor(pos):
    x = pos[0]
    y = pos[1]

    size = range(-1, 2)
    xx, yy = np.meshgrid(size, size)

    xx = xx[1:].ravel()
    yy = yy[1:].ravel()

    xx = x + xx
    yy = y + yy

    pairs = []

    for i in xx:
        for j in yy:
            if (i, j) not in pairs and 0 < i < MAP_WIDTH and 0 <= j < MAP_WIDTH:
                pairs.append((i, j))

    lower_value = float('inf')
    lower_pair = (-1, -1)

    details = []

    for pair in pairs:
        details.append((full_potential_map[pair], pair))
        if full_potential_map[pair] <= lower_value:
            lower_value = full_potential_map[pair]
            lower_pair = pair

    return lower_pair


ATTR_GAIN = 5
REPU_GAIN = 50
MAP_WIDTH = 30
ROBOT_RADIUS = 1

map = np.ones((MAP_WIDTH, MAP_WIDTH), dtype=np.uint8) * 128
initial = (0, 0)
goal = (MAP_WIDTH - 1, MAP_WIDTH - 1)

attr_potential_map = np.zeros_like(map)
repu_potential_map = np.zeros_like(map)
full_potential_map = np.zeros_like(map)

obstacles_x_components = []
obstacles_y_components = []

add_obstacle((3, 3), 10, 10)
add_obstacle((2, 2), 16, 16)
add_obstacle((3, 14), 15, 0)
add_obstacle((14, 3), 0, 15)
add_obstacle((3, 14), 15, 0)
add_obstacle((7, 1), 16, 24)
add_obstacle((1, 7), 24, 16)
add_obstacle((3, 3), 20, 20)

for i in range(0, MAP_WIDTH):
    for j in range(0, MAP_WIDTH):
        attr_potential_map[i][j] = attractive_potential(i, j, goal[0], goal[1])
        repu_potential_map[i][j] = repulsive_potential(i, j, obstacles_x_components, obstacles_y_components)

full_potential_map = attr_potential_map
full_potential_map[repu_potential_map > 1] = 255

cv2.normalize(full_potential_map, full_potential_map, 0, 255, cv2.NORM_MINMAX)

plane_range = range(0, len(full_potential_map))

X, Y = np.meshgrid(plane_range, plane_range)

Z = np.array(full_potential_map)

fig = plt.figure()
ax = fig.add_subplot(121, projection='3d')
plt.title('3D Plot')

ax.plot_surface(X, Y, Z, cmap='jet')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

full_potential_map_img = cv2.cvtColor(
    cv2.merge((full_potential_map, full_potential_map, full_potential_map)),
    cv2.COLOR_BGR2RGB)

full_potential_map_img[initial] = (255, 0, 0)
full_potential_map_img[goal] = (0, 255, 0)

already_passed = {'values': [], 'indexes': []}
current = np.copy(initial)

plt.subplot('122')

plt.title('2D Plot')
plt.imshow(full_potential_map_img)
plt.pause(4)

while not np.array_equal(current, goal):
    current = get_minor(current)

    plt.plot(current[1], current[0], '.r')
    plt.pause(0.1)

plt.show()
