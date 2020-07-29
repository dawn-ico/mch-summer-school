// THESE WRAPPERS ARE NEEDED SINCE THE GENERATED CODE IS TEMPLATED
#pragma once

#include "driver-includes/defs.hpp"

#include "interface/atlas_interface.hpp"

void divergence_wrapper_naive(
    const dawn::mesh_t<atlasInterface::atlasTag>& mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_length,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& cell_area,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_orientation_cell);

void divergence_wrapper_cuda_run(
    const dawn::mesh_t<atlasInterface::atlasTag>& mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_length,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& cell_area,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& edge_orientation_cell);

void divergence_wrapper_cuda_copy_back(
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>& div_vec);