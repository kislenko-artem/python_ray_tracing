from typing import Tuple, Optional, List
import sys
from math import sqrt, pow

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

# @numba.njit()
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


# @numba.njit()
def multiply_sw(infant: float, v: Tuple[float, float, float]) -> Tuple[
    float, float, float]:
    return infant * v[0], infant * v[1], infant * v[2]


# @numba.njit()
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


# @numba.njit()
def add(
        item: Tuple[float, float, float],
        item2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return item[0] + item2[0], item[1] + item2[1], item[2] + item2[2]


# @numba.njit()
def subtract(item: Tuple[float, float, float],
             item2: Tuple[float, float, float]) -> Tuple[float, float, float]:
    return item[0] - item2[0], item[1] - item2[1], item[2] - item2[2]


# @numba.njit()
def length(vec: Tuple[float, float, float]):
    return sqrt(dot_product(vec, vec))


# @numba.njit()
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

