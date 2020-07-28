from dusk.script import (
    stencil,
    Field,
    Edge,
    Cell,
    Vertex,
    forward,
    backward,
    neighbors,
    reduce_over,
)


@stencil
def ICON_laplacian_fvm(
    vec: Field[Edge, K],
    div_vec: Field[Cell, K],
    rot_vec: Field[Vertex, K],
    nabla2t1_vec: Field[Edge, K],
    nabla2t2_vec: Field[Edge, K],
    nabla2_vec: Field[Edge, K],
    primal_edge_length: Field[Edge],
    dual_edge_length: Field[Edge],
    tangent_orientation: Field[Edge],
    geofac_rot: Field[Vertex > Edge],
    geofac_div: Field[Cell > Edge],
) -> None:

    with levels_upward as k:

        # compute curl (on vertices)
        rot_vec = reduce_over(
            Vertex > Edge,
            vec * geofac_rot,
            sum,
            init=0.0,
        )

        # compute divergence (on cells)
        div_vec = reduce_over(
            Cell > Edge,
            vec * geofac_div,
            sum,
            init=0.0,
        )

        # first term of of nabla2 (gradient of curl)
        nabla2t1_vec = reduce_over(
            Edge > Vertex,
            rot_vec,
            sum,
            init=0.0,
            weights=[-1., 1, ],
        )
        nabla2t1_vec = tangent_orientation*nabla2t1_vec/primal_edge_length

        # second term of of nabla2 (gradient of divergence)
        nabla2t2_vec = reduce_over(
            Edge > Cell,
            div_vec,
            sum,
            init=0.0,
            weights=[-1., 1, ],
        )
        nabla2t2_vec = tangent_orientation*nabla2t2_vec/dual_edge_length

        # finalize nabla2 (difference between the two gradients)
        nabla2_vec = nabla2t2_vec - nabla2t1_vec
