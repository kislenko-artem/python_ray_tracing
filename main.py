import math
import sys
import time
from typing import Tuple, NamedTuple

import numba
import numpy as np
import pygame
from numba import typed

# Intialize the pygame
pygame.init()

WIDTH = 600
HEIGHT = 600

BLACK = (0.0, 0.0, 0.0)
WHITE = (255.0, 255.0, 255.0)
BACKGROUND_COLOR = WHITE

VIEW_PORT_SIZE = 1
PROJECTION_PLAN_Z = 1.0
CAMERA_ROTATION = [[0.7071, 0, -0.7071],
                   [0, 1, 0],
                   [0.7071, 0, 0.7071]]
CAMERA_POSITION = (0.0, 0.0, 0.0)

MIN_POINT = 1
MAX_POINT = sys.maxsize
RECURSION_DETH = 3

LIGHT_AMBIENT = 0
LIGHT_POINT = 1
LIGHT_DIRECTIONAL = 2


class Sphere(NamedTuple):
    center: Tuple[float, float, float]
    radius: int
    color: Tuple[float, float, float]
    specular: int
    reflective: float


class Light(NamedTuple):
    type: int
    intensity: float
    position: Tuple[float, float, float]


# create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))


@numba.njit(cache=True, fastmath=True)
def canvas_to_viewport(x: int, y: int) -> Tuple[float, float, float]:
    """
    Переводим координаты из заданных в конфиге к размерам нашего холста.
    Так же переводим проекцию 2d в 3d
    :param x: текущая x координата холста
    :param y: текущая y координата холста
    :return:
    """
    new_x = x * VIEW_PORT_SIZE / WIDTH
    new_y = y * VIEW_PORT_SIZE / WIDTH
    return new_x, new_y, PROJECTION_PLAN_Z


@numba.njit(cache=True, fastmath=True)
def multiply_sw(infant: float, v: Tuple[float, float, float]) -> Tuple[
    float, float, float]:
    return infant * v[0], infant * v[1], infant * v[2]


@numba.njit(cache=True, fastmath=True)
def multiply_mv(x: float, y: float, z: float) -> Tuple[float, float, float]:
    """
    Вычисляем поворот камеры для каждого пикселя
    :param x:
    :param y:
    :param z:
    :return:
    """
    new_x = 0.0
    new_y = 0.0
    new_z = 0.0
    for i in range(3):
        new_x += x * CAMERA_ROTATION[i][0]
        new_y += y * CAMERA_ROTATION[i][1]
        new_z += z * CAMERA_ROTATION[i][2]

    return new_x, new_y, new_z


@numba.njit(cache=True, fastmath=True)
def add(
        item: Tuple[float, float, float],
        item2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return item[0] + item2[0], item[1] + item2[1], item[2] + item2[2]


@numba.njit(cache=True, fastmath=True)
def subtract(item: Tuple[float, float, float],
             item2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return item[0] - item2[0], item[1] - item2[1], item[2] - item2[2]


@numba.njit(cache=True, fastmath=True)
def length(vec: Tuple[float, float, float]) -> float:
    return math.sqrt(dot_product(vec, vec))


@numba.njit(cache=True, fastmath=True)
def dot_product(
        vec: Tuple[float, float, float],
        vec2: Tuple[float, float, float]) -> float:
    """
    Скалярное произведение:Ж операция над двумя векторами,
     результатом которой является скаляр, то есть число,
     не зависящее от выбора системы координат
    :param item: вектор
    :param item2: вектор
    :return:
    """
    return vec[0] * vec2[0] + vec[1] * vec2[1] + vec[2] * vec2[2]


@numba.njit(cache=True)
def closest_intersection(spheres: typed.List[Sphere],
                         position: Tuple[float, float, float],
                         min_point: float,
                         x: float,
                         y: float,
                         z: float) -> Tuple[Sphere, float]:
    closest_t: float = float(sys.maxsize)
    closest_sphere = None

    for s in spheres:
        ts = intersect_ray_sphere(position, x, y, z, s)
        # print(ts)
        if ts[0] < closest_t and min_point < ts[0] and ts[0] < MAX_POINT:
            closest_t = ts[0]
            closest_sphere = s
        if ts[1] < closest_t and min_point < ts[1] and ts[1] < MAX_POINT:
            closest_t = ts[1]
            closest_sphere = s
    return closest_sphere, closest_t


@numba.njit(cache=True)
def compute_lighting(lights: typed.List[Light],
                     spheres: typed.List[Sphere],
                     point: Tuple[float, float, float],
                     normal: Tuple[float, float, float],
                     view: Tuple[float, float, float],
                     specular: int):
    i = 0.0
    length_n = length(normal)
    length_v = length(view)
    for light in lights:
        vec_l = light.position
        if light.type == LIGHT_AMBIENT:
            i += light.intensity
        else:
            if light.type == LIGHT_POINT:
                vec_l = subtract(light.position, point)
            n_dot_l = dot_product(normal, vec_l)

            shadow_sphere, shadow_t = closest_intersection(
                spheres, point, 0.0001, vec_l[0], vec_l[1], vec_l[2])
            if shadow_sphere is not None:
                continue

            if n_dot_l > 0:
                i += light.intensity * n_dot_l / (length_n * length(vec_l))

            if specular < 0:
                continue
            vec_r = subtract(
                multiply_sw(2.0 * dot_product(normal, vec_l), normal), vec_l)
            r_dot_v = dot_product(vec_r, view)
            if r_dot_v > 0:
                i += light.intensity * pow(r_dot_v / (length(vec_r) * length_v),
                                           specular)

    return i


@numba.njit(cache=True)
def intersect_ray_sphere(position: Tuple[float, float, float],
                         x: float,
                         y: float,
                         z: float,
                         sphere: Sphere) -> Tuple[
    float, float]:
    oc = subtract(position, sphere.center)

    k1 = dot_product((x, y, z), (x, y, z))
    k2 = 2 * dot_product(oc, (x, y, z))
    k3 = dot_product(oc, oc) - sphere.radius * sphere.radius
    discriminant = k2 * k2 - 4 * k1 * k3
    if discriminant < 0:
        return float(sys.maxsize), float(sys.maxsize)

    t1 = (-k2 + math.sqrt(discriminant)) / (2 * k1)
    t2 = (-k2 - math.sqrt(discriminant)) / (2 * k1)
    return t1, t2


@numba.njit(cache=True)
def trace_ray(
        spheres: typed.List[Sphere],
        lights: typed.List[Light],
        x: float,
        y: float,
        z: float) -> Tuple[
    int, int, int]:
    closest_sphere, closest_t = closest_intersection(
        spheres, CAMERA_POSITION, MIN_POINT, x, y, z)

    if closest_sphere is None:
        return BACKGROUND_COLOR

    # return closest_sphere.color

    point = add(CAMERA_POSITION, multiply_sw(closest_t, (x, y, z)))
    normal = subtract(point, closest_sphere.center)
    normal = multiply_sw(1.0 / length(normal), normal)
    view = multiply_sw(-1, (x, y, z))
    ligth = compute_lighting(lights, spheres, point, normal, view,
                             closest_sphere.specular)
    r, g, b = multiply_sw(ligth, closest_sphere.color)
    if r > 255:
        r = 255
    if g > 255:
        g = 255
    if b > 255:
        b = 255
    return int(r), int(g), int(b)


Coordinates = Tuple[int, int]
Color = Tuple[int, int, int]



# если включить parallel - работает дольше, чем без него
@numba.njit(cache=True)
def get_pix(color_arr: np.ndarray,
            coord_arr: np.ndarray,
            spheres: typed.List[Sphere],
            lights: typed.List[Light]):

    for x in numba.prange(int(WIDTH / 2 * -1), int(WIDTH / 2 * 1)):
        set_x = int(WIDTH / 2) + int(x)
        for y in numba.prange(int(HEIGHT / 2 * -1), int(HEIGHT / 2 * 1)):
            new_x, new_y, z = canvas_to_viewport(x, y)
            color = trace_ray(spheres, lights, new_x, new_y, z)
            set_y = int(HEIGHT / 2) - int(y) - 1
            key = set_y + (set_x * WIDTH)
            coord_arr[key] = np.array([int(set_x), int(set_y)])
            color_arr[key] = np.array(
                [int(color[0]), int(color[1]), int(color[2])])



if __name__ == '__main__':
    clock = pygame.time.Clock()

    spheres = typed.List([
        Sphere((0.0, -1.0, 3.0), 1, (255.0, 0.0, 0.0), 500, 0.2),
        Sphere((2.0, 0.0, 4.0), 1, (0.0, 0.0, 255.0), 500, 0.3),
        Sphere((-2.0, 0.0, 4.0), 1, (0.0, 255.0, 0.0), 10, 0.4),
        Sphere((0.0, -5001.0, 0.0), 5000, (255.0, 255.0, 0.0), 1000, 0.5)]
    )

    lights = typed.List([
        Light(LIGHT_AMBIENT, 0.2, (2.0, 1.0, 0.0)),
        Light(LIGHT_POINT, 0.6, (2.0, 1.0, 0.0)),
        Light(LIGHT_DIRECTIONAL, 0.2, (1.0, 4.0, 4.0))]
    )
    while True:
        clock.tick(60)
        screen.fill(WHITE)
        for event in pygame.event.get():

            if event.type == pygame.QUIT:
                pygame.quit()
                exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    exit()

        s_time = time.time()
        color_array = np.zeros((WIDTH * HEIGHT, 3), dtype=int)
        coord_array = np.zeros((WIDTH * HEIGHT, 2), dtype=int)
        get_pix(color_array, coord_array, spheres, lights)
        print("compute", time.time() - s_time)
        # TODO:
        # 1. Перерисовывать часть экрана (только та, которая изменилась)
        # 2. Добавить элементы управления
        for i in range(0, len(color_array)):
            screen.set_at(
                (coord_array[i][0], coord_array[i][1]),
                (color_array[i][0], color_array[i][1], color_array[i][2])
            )
        print("compute + draw", time.time() - s_time)
        pygame.display.update()
