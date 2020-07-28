#include "nh_diffusion_fvm_wrapper.h"

#include <generated/nh_diffusion_fvm_cuda.cpp>

#define C_E_SIZE 3
#define E_C_SIZE 2
#define E_V_SIZE 2
#define V_E_SIZE 6

static dawn_generated::cuda_ico::ICON_laplacian_fvm<
    atlasInterface::atlasTag, C_E_SIZE, E_C_SIZE, E_V_SIZE,
    V_E_SIZE>::stencil_72 *stencil;

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
        &geofac_div) {
  stencil = new dawn_generated::cuda_ico::ICON_laplacian_fvm<
      atlasInterface::atlasTag, C_E_SIZE, E_C_SIZE, E_V_SIZE,
      V_E_SIZE>::stencil_72(mesh, k_size, vec, div_vec, rot_vec, nabla2t1_vec,
                            nabla2t2_vec, nabla2_vec, primal_edge_length,
                            dual_edge_length, tangent_orientation, geofac_rot,
                            geofac_div);

  stencil->run();
}

void nh_diffusion_fvm_wrapper_cuda_copy_back(
    dawn::cell_field_t<atlasInterface::atlasTag, ::dawn::float_type> &div_vec,
    dawn::vertex_field_t<atlasInterface::atlasTag, ::dawn::float_type> &rot_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t1_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2t2_vec,
    dawn::edge_field_t<atlasInterface::atlasTag, ::dawn::float_type>
        &nabla2_vec) {

  stencil->CopyResultToHost(div_vec, rot_vec, nabla2t1_vec, nabla2t2_vec,
                            nabla2_vec);
}