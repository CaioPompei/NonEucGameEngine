"""
Matemática dos portais: câmera virtual e clipping com near plane oblíquo.

Referência:
    Lengyel, E. (2005). Oblique View Frustum Depth Projection and Clipping.
    Journal of Game Development, 1(2), 5-16.

Convenção de matrizes: pyrr (row-vector). A matriz armazenada em numpy é a
transposta da matriz column-vector usada pelo OpenGL. Por isso, modificações
que Lengyel descreve como "linha 2" da matriz de projeção viram modificações
na "coluna 2" (índice 2) do array numpy.
"""

import math
import numpy as np


# Abaixo desta distância (world units) entre a câmera virtual e o plano do
# portal destino, NÃO aplicamos o near plane oblíquo. Quando a câmera virtual
# encosta no plano — o que acontece justamente ao ATRAVESSAR o portal — o near
# plane oblíquo passa a coincidir com a câmera, `dot(c, q)` tende a zero e a
# matriz de projeção degenera, pintando um "quadrado" de lixo dentro do portal.
# Nesse regime a fatia entre a câmera e o portal é fina demais para vazar
# geometria, então cair para a projeção normal por esse frame é imperceptível.
_OBLIQUE_MIN_DISTANCE = 0.1


def portal_normal_world(rotation_degrees: float) -> np.ndarray:
    """
    Normal frontal do portal em world space.
    Convenção local: front aponta para +Z. Aplica rotação em Y.
    """
    theta = math.radians(rotation_degrees)
    return np.array([math.sin(theta), 0.0, math.cos(theta)], dtype=np.float32)


_FLIP_Y_180 = np.array([
    [-1.0, 0.0,  0.0, 0.0],
    [ 0.0, 1.0,  0.0, 0.0],
    [ 0.0, 0.0, -1.0, 0.0],
    [ 0.0, 0.0,  0.0, 1.0],
], dtype=np.float32)


def calculate_virtual_view(real_view: np.ndarray,
                           origin_transform: np.ndarray,
                           destiny_transform: np.ndarray) -> np.ndarray:
    """
    View matrix da câmera virtual para renderizar o lado destino do portal.

    Inclui uma rotação de 180° em Y entre origem e destino: ao "atravessar"
    um portal, a câmera sai pela FRENTE do destino, não pela costas. Sem
    esse flip, a câmera virtual ficaria espelhada e olharia para o lado
    oposto do esperado.

    Pyrr (row-vector):
        V_virtual = inv(M_destino) @ FlipY180 @ M_origem @ V_real
    """
    inv_destiny = np.linalg.inv(destiny_transform)
    return calculate_virtual_view_cached(real_view, origin_transform, inv_destiny)


def calculate_virtual_view_cached(real_view: np.ndarray,
                                  origin_transform: np.ndarray,
                                  inv_destiny_transform: np.ndarray) -> np.ndarray:
    """
    Mesma fórmula de `calculate_virtual_view`, mas aceita a inversa do
    destiny pré-calculada. Útil quando o portal é estático e a inversa
    pode ser cacheada (evita `np.linalg.inv` por frame, ainda mais
    importante quando o renderer recursa até `max_depth`).
    """
    return (inv_destiny_transform @ _FLIP_Y_180
            @ origin_transform @ real_view).astype(np.float32)


def calculate_traversal_transform(origin_transform: np.ndarray,
                                  destiny_transform: np.ndarray) -> np.ndarray:
    """
    Transform pyrr (row-vector) que leva um ponto/direção em world space do
    lado FRONTAL do portal de origem para o ponto/direção análogo no lado
    frontal do portal de destino, incluindo a rotação de 180° em Y exigida
    pelo pareamento.

    Derivação:
        P_local_orig = P_world @ inv(M_orig)
        P_local_dest = P_local_orig @ FlipY180
        P_world_new  = P_local_dest @ M_dest
      ⇒  T = inv(M_orig) @ FlipY180 @ M_dest

    Aplicação:
        p_new = (px, py, pz, 1) @ T   → pega os 3 primeiros componentes
        d_new = (dx, dy, dz, 0) @ T   → ignora a translação
    """
    inv_origin = np.linalg.inv(origin_transform)
    return (inv_origin @ _FLIP_Y_180 @ destiny_transform).astype(np.float32)


def transform_point(transform: np.ndarray, point) -> np.ndarray:
    """Aplica uma transform pyrr (row-vector) em um ponto 3D."""
    p = np.array([point[0], point[1], point[2], 1.0], dtype=np.float32)
    return (p @ transform)[:3].astype(np.float32)


def transform_direction(transform: np.ndarray, direction) -> np.ndarray:
    """Aplica a parte 3×3 de uma transform pyrr em um vetor direção."""
    d = np.array([direction[0], direction[1], direction[2], 0.0], dtype=np.float32)
    return (d @ transform)[:3].astype(np.float32)


def plane_world_to_view(normal_world: np.ndarray,
                        point_world: np.ndarray,
                        view_matrix: np.ndarray) -> np.ndarray:
    """
    Converte um plano (ponto + normal em world space) para view space,
    retornando (a, b, c, d) tal que a*x + b*y + c*z + d >= 0 é o semi-espaço
    "manter" (não clipado).

    Assume view_matrix ortonormal (rotação + translação, sem escala).
    """
    point_h = np.array([point_world[0], point_world[1], point_world[2], 1.0],
                       dtype=np.float32)
    point_view = point_h @ view_matrix

    normal_h = np.array([normal_world[0], normal_world[1], normal_world[2], 0.0],
                        dtype=np.float32)
    n_view = (normal_h @ view_matrix)[:3]
    n_view = n_view / np.linalg.norm(n_view)

    d = -float(np.dot(n_view, point_view[:3]))
    return np.array([n_view[0], n_view[1], n_view[2], d], dtype=np.float32)


def oblique_near_plane(projection: np.ndarray,
                       clip_plane_view: np.ndarray) -> np.ndarray:
    """
    Modifica a matriz de projeção para que o near plane coincida com
    `clip_plane_view` (plano em view space).

    Implementação da fórmula de Lengyel adaptada à convenção row-vector
    do pyrr.

    Em pyrr, a matriz numpy P satisfaz P[i][j] = M_math[j][i], onde M_math
    é a matriz column-vector do OpenGL. Logo:
      - Lê-se a "coluna 2" de M_math via P[2][*]
      - Escreve-se na "linha 2" de M_math via P[*][2]
    """
    P = projection.copy().astype(np.float32)

    q_x = (np.sign(clip_plane_view[0]) + P[2][0]) / P[0][0]
    q_y = (np.sign(clip_plane_view[1]) + P[2][1]) / P[1][1]
    q_z = -1.0
    q_w = (1.0 + P[2][2]) / P[3][2]
    q = np.array([q_x, q_y, q_z, q_w], dtype=np.float32)

    dot_cq = float(np.dot(clip_plane_view, q))
    if abs(dot_cq) < 1e-6:
        # Plano degenerado em relação ao frustum: devolve projeção original
        # para evitar divisão por zero.
        return P

    c = clip_plane_view * (2.0 / dot_cq)

    P[0][2] = c[0]
    P[1][2] = c[1]
    P[2][2] = c[2] + 1.0
    P[3][2] = c[3]

    return P


def calculate_oblique_projection(projection: np.ndarray,
                                 virtual_view: np.ndarray,
                                 destiny_position: np.ndarray,
                                 destiny_rotation_degrees: float) -> np.ndarray:
    """
    Constrói o plano do portal destino em view space e devolve a projeção
    com o near plane oblíquo aplicado.

    A normal é orientada automaticamente para apontar AFASTANDO da câmera
    virtual (ou seja, para o semi-espaço a ser mantido), de modo que
    qualquer geometria entre a câmera virtual e o plano do portal destino
    seja clipada.
    """
    normal_w = portal_normal_world(destiny_rotation_degrees)

    inv_view = np.linalg.inv(virtual_view)
    virtual_cam_pos = inv_view[3, :3]

    cam_to_portal = destiny_position - virtual_cam_pos
    signed_dist = float(np.dot(normal_w, cam_to_portal))
    if signed_dist < 0.0:
        normal_w = -normal_w
        signed_dist = -signed_dist

    # Câmera virtual coladíssima no plano destino (você está atravessando):
    # a projeção oblíqua degenera. Devolve a projeção normal por este frame.
    if signed_dist < _OBLIQUE_MIN_DISTANCE:
        return projection.astype(np.float32)

    plane_view = plane_world_to_view(normal_w, destiny_position, virtual_view)
    return oblique_near_plane(projection, plane_view)
