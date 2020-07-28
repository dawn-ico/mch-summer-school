#include "nh_diffusion_fvm_wrapper.h"

#include <generated/nh_diffusion_fvm_cxx-naive.cpp>

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
        &geofac_div) {
  dawn_generated::cxxnaiveico::ICON_laplacian_fvm<atlasInterface::atlasTag>(
      mesh, k_size, vec, div_vec, rot_vec, nabla2t1_vec, nabla2t2_vec,
      nabla2_vec, primal_edge_length, dual_edge_length, tangent_orientation,
      geofac_rot, geofac_div)
      .run();
}
