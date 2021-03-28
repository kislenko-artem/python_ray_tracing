
import sys
import time
from dataclasses import dataclass
from math import sqrt, pow
from typing import Tuple, Optional, List

import pygame

from methods import *

# Intialize the pygame
pygame.init()

WIDTH = 600
HEIGHT = 600

BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BACKGROUND_COLOR = WHITE

VIEW_PORT_SIZE = 1
PROJECTION_PLAN_Z = 1.0
CAMERA_ROTATION = [[0.7071, 0, -0.7071],
                   [0, 1, 0],
                   [0.7071, 0, 0.7071]]
CAMERA_POSITION = (0, 0, 0)

MIN_POINT = 1
MAX_POINT = sys.maxsize
RECURSION_DETH = 3

LIGHT_AMBIENT = 0
LIGHT_POINT = 1
LIGHT_DIRECTIONAL = 2



# create the screen
screen = pygame.display.set_mode((WIDTH, HEIGHT))

@dataclass(frozen=True)
class Sphere:
    center: Tuple[int, int, int]
    radius: int
    color: Tuple[int, int, int]
    specular: int
    reflective: float


@dataclass(frozen=True)
class Light:
    type: int
    intensity: float
    position: Optional[Tuple[int, int, int]]


def closest_intersection(spheres: List[Sphere],
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


# @numba.jit(forceobj=True)
def compute_lighting(spheres: List[Sphere],
                     lights: List[Light],
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


# @numba.jit(forceobj=True)
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
        return sys.maxsize, sys.maxsize

    t1 = (-k2 + sqrt(discriminant)) / (2 * k1)
    t2 = (-k2 - sqrt(discriminant)) / (2 * k1)
    return t1, t2


# @numba.jit(forceobj=True)
def trace_ray(
        spheres: List[Sphere],
        lights: List[Light],
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
    ligth = compute_lighting(spheres, lights, point, normal, view,
                             closest_sphere.specular)
    r, g, b = multiply_sw(ligth, closest_sphere.color)
    if r > 255:
        r = 255
    if g > 255:
        g = 255
    if b > 255:
        b = 255
    return int(r), int(g), int(b)


# @numba.jit(forceobj=True)
def set_pix(screen: pygame.Surface,
            spheres: List[Sphere],
            lights: List[Light]):
    for x in range(int(WIDTH / 2 * -1), int(WIDTH / 2 * 1)):
        for y in range(int(HEIGHT / 2 * -1), int(HEIGHT / 2 * 1)):
            new_x, new_y, z = canvas_to_viewport(x, y)
            r, g, b = trace_ray(spheres, lights, new_x, new_y, z)
            set_x = int(WIDTH / 2) + int(x)
            set_y = int(HEIGHT / 2) - int(y) - 1
            screen.set_at((set_x, set_y), (r, g, b))


if __name__ == '__main__':
    clock = pygame.time.Clock()

    spheres = [
        Sphere((0, -1, 3), 1, (255, 0, 0), 500, 0.2),
        Sphere((2, 0, 4), 1, (0, 0, 255), 500, 0.3),
        Sphere((-2, 0, 4), 1, (0, 255, 0), 10, 0.4),
        Sphere((0, -5001, 0), 5000, (255, 255, 0), 1000, 0.5),
    ]

    lights = [
        Light(LIGHT_AMBIENT, 0.2, None),
        Light(LIGHT_POINT, 0.6, (2, 1, 0)),
        Light(LIGHT_DIRECTIONAL, 0.2, (1, 4, 4)),
    ]
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
        set_pix(screen, spheres, lights)
        print(time.time() - s_time)

        pygame.display.update()
