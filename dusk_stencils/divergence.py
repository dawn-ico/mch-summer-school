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
        ##############################################
        # Some examples of what can be done in dusk  #
        ##############################################

        # arithmetic operations
        div_vec = div_vec + cell_area

        # illegal arithmetic operation
        # different location types
        # vec = vec + div_vec  # <- compiler error

        # reductions
        # lhs = reduce(Expr, "Op", "Init", LocChain)
        div_vec = reduce(edge_orientation_cell, "+", 0., Cell > Edge)
        div_vec = reduce(vec + edge_length, "-", 2., Cell > Edge)

        # reductions are typed by the location chain
        # Cell > Edge implies that:
        # - lhs has to be on Cell
        # - Expr contains only field on Edges or Cell > Edge
        # Example of an illegal reduction
        # vec = reduce(div_vec, "+", 0., Cell > Edge)
        #  ^              ^
        # Edge          Cell             Cell > Edge
        #
        # ==> compiler error!

        # TODO
        # compute correct divergence
        # div_vec = reduce(....
