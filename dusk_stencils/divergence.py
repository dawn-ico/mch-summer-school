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
def calc_divergence(
    vec: Field[Edge],
    div_vec: Field[Cell],
    edge_length: Field[Edge],
    cell_area: Field[Cell],
    edge_orientation_cell: Field[Cell > Edge]
) -> None:

    for _ in forward:

        # compute divergence (on cells)
        div_vec = reduce(
            vec * edge_length * edge_orientation_cell,
            "+",
            0.0,
            Cell > Edge,
        )
        div_vec = div_vec / cell_area
