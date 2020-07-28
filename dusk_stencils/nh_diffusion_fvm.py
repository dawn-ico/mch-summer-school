from dusk.script import (
    stencil,
    Field,
    Edge,
    Cell,
    Vertex,
    forward,
    backward,
    neighbors,
    reduce,
)


@stencil
def ICON_laplacian_fvm(
    vec: Field[Edge],
    div_vec: Field[Cell],
    rot_vec: Field[Vertex],
    nabla2t1_vec: Field[Edge],
    nabla2t2_vec: Field[Edge],
    nabla2_vec: Field[Edge],
    primal_edge_length: Field[Edge],
    dual_edge_length: Field[Edge],
    tangent_orientation: Field[Edge],
    geofac_rot: Field[Vertex > Edge],
    geofac_div: Field[Cell > Edge],
) -> None:

    for _ in forward:

        # compute curl (on vertices)
        rot_vec = reduce(
            vec * geofac_rot,
            "+",
            0.0,
            Vertex > Edge,
        )

        # compute divergence (on cells)
        div_vec = reduce(
            vec * geofac_div,
            "+",
            0.0,
            Cell > Edge,
        )

        # first term of of nabla2 (gradient of curl)
        nabla2t1_vec = reduce(
            rot_vec,
            "+",
            0.0,
            Edge > Vertex,
            [-1., 1, ],
        )
        nabla2t1_vec = tangent_orientation*nabla2t1_vec/primal_edge_length

        # second term of of nabla2 (gradient of divergence)
        nabla2t2_vec = reduce(
            div_vec,
            "+",
            0.0,
            Edge > Cell,
            [-1., 1, ],
        )
        nabla2t2_vec = tangent_orientation*nabla2t2_vec/dual_edge_length

        # finalize nabla2 (difference between the two gradients)
        nabla2_vec = nabla2t2_vec - nabla2t1_vec
