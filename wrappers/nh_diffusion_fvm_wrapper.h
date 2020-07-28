// THESE WRAPPERS ARE NEEDED SINCE THE GENERATED CODE IS TEMPLATED
#pragma once

#include "driver-includes/defs.hpp"

#include "interface/atlas_interface.hpp"

void nh_diffusion_fvm_wrapper_naive(
    const dawn::mesh_t<atlasInterface::atlasTag> &mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type> &vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type> &div_vec,
    dawn::vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type> &rot_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t1_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &primal_edge_length,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &dual_edge_length,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &tangent_orientation,
    dawn::sparse_vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &geofac_rot,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &geofac_div);

void nh_diffusion_fvm_wrapper_cuda_run(
    const dawn::mesh_t<atlasInterface::atlasTag> &mesh, int k_size,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type> &vec,
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type> &div_vec,
    dawn::vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type> &rot_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t1_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &primal_edge_length,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &dual_edge_length,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &tangent_orientation,
    dawn::sparse_vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &geofac_rot,
    dawn::sparse_cell_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &geofac_div);

void nh_diffusion_fvm_wrapper_cuda_copy_back(
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type> &div_vec,
    dawn::vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type> &rot_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t1_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2_vec);