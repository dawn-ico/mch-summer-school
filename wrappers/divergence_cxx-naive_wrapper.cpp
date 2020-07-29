#include "divergence_wrapper.h"

#include <generated/divergence_cxx-naive.cpp>

void divergence_wrapper_naive(
    const dawn::mesh_t<atlasInterface::atlasTag>& mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_length,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& cell_area,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>&
        edge_orientation_cell) {
  dawn_generated::cxxnaiveico::calc_divergence<atlasInterface::atlasTag>(
      mesh, k_size, vec, div_vec, edge_length, cell_area, edge_orientation_cell)
      .run();
}
