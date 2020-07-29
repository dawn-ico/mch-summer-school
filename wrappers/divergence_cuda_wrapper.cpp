#include "divergence_wrapper.h"

#include <generated/divergence_cuda.cpp>

#define C_E_SIZE 3

static dawn_generated::cuda_ico::calc_divergence<atlasInterface::atlasTag, C_E_SIZE>::stencil_26*
    stencil;

void divergence_wrapper_cuda_run(
    const dawn::mesh_t<atlasInterface::atlasTag>& mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_length,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& cell_area,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>&
        edge_orientation_cell) {
  stencil =
      new dawn_generated::cuda_ico::calc_divergence<atlasInterface::atlasTag, C_E_SIZE>::stencil_26(
          mesh, k_size, vec, div_vec, edge_length, cell_area, edge_orientation_cell);

  stencil->run();
}

void divergence_wrapper_cuda_copy_back(
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec) {
  stencil->CopyResultToHost(div_vec);
}