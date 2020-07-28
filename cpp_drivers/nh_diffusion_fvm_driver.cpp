
// atlas functions
#include <atlas/array.h>
#include <atlas/grid.h>
#include <atlas/mesh.h>
#include <atlas/mesh/actions/BuildEdges.h>
#include <atlas/output/Gmsh.h>
#include <atlas/util/CoordinateEnums.h>

// atlas interface for dawn generated code
#include "interface/atlas_interface.hpp"

// driver includes
#include "driver-includes/defs.hpp"

// atlas utilities
#include "atlas_utils/utils/AtlasCartesianWrapper.h"
#include "atlas_utils/utils/AtlasFromNetcdf.h"
#include "atlas_utils/utils/GenerateRectAtlasMesh.h"
#include "atlas_utils/utils/GenerateStrIndxAtlasMesh.h"

#include <gtest/gtest.h>

#include <cmath>
#include <cstdio>
#include <fenv.h>
#include <optional>
#include <vector>

#include "nh_diffusion_fvm_wrapper.h"

#include "UnstructuredVerifier.h"

template <typename T> static int sgn(T val) {
  return (T(0) < val) - (val < T(0));
}

TEST(nh_diffusion_fvm, manufactured_and_sidebyside) {
  // Setup a 32 by 32 grid of quads and generate a mesh out of it
  const int nx = 32, ny = 32;
  auto mesh = AtlasMeshRect(nx, ny);
  const int k_size = 32;

  // wrapper with various atlas helper functions
  AtlasToCartesian wrapper(mesh, true);

  const int edgesPerVertex = 6;
  const int edgesPerCell = 3;

  //===------------------------------------------------------------------------------------------===//
  // helper lambdas to readily construct atlas fields and views on one line
  //===------------------------------------------------------------------------------------------===//
  auto MakeAtlasField = [&](const std::string &name, int size)
      -> std::tuple<atlas::Field, atlasInterface::Field<dawn::float_type>> {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real32(),
                         atlas::array::make_shape(size, k_size)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real64(),
                         atlas::array::make_shape(size, k_size)};
#else
#error DAWN_PRECISION is invalid
#endif
    return {field_F, atlas::array::make_view<dawn::float_type, 2>(field_F)};
  };

  auto MakeAtlasSparseField = [&](const std::string &name, int size,
                                  int sparseSize)
      -> std::tuple<atlas::Field,
                    atlasInterface::SparseDimension<dawn::float_type>> {
#if DAWN_PRECISION == DAWN_SINGLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real32(),
                         atlas::array::make_shape(size, k_size, sparseSize)};
#elif DAWN_PRECISION == DAWN_DOUBLE_PRECISION
    atlas::Field field_F{name, atlas::array::DataType::real64(),
                         atlas::array::make_shape(size, k_size, sparseSize)};
#else
#error DAWN_PRECISION is invalid
#endif
    return {field_F, atlas::array::make_view<dawn::float_type, 3>(field_F)};
  };

  //===------------------------------------------------------------------------------------------===//
  // input field (field we want to take the laplacian of)
  //===------------------------------------------------------------------------------------------===//
  auto [vec_F, vec] = MakeAtlasField("vec", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // control field holding the analytical solution for the divergence
  //===------------------------------------------------------------------------------------------===//
  auto [divVecSol_F, divVecSol] =
      MakeAtlasField("divVecSol", mesh.cells().size());

  //===------------------------------------------------------------------------------------------===//
  // control field holding the analytical solution for the curl
  //===------------------------------------------------------------------------------------------===//
  auto [rotVecSol_F, rotVecSol] =
      MakeAtlasField("rotVecSol", mesh.nodes().size());

  //===------------------------------------------------------------------------------------------===//
  // control field holding the analytical solution for Laplacian
  //===------------------------------------------------------------------------------------------===//
  auto [lapVecSol_F, lapVecSol] =
      MakeAtlasField("lapVecSol", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // output field (field containing the computed laplacian)
  //===------------------------------------------------------------------------------------------===//
  auto [nabla2_vec_F_gpu, nabla2_vec_gpu] =
      MakeAtlasField("nabla2_vec", mesh.edges().size());
  auto [nabla2_vec_F_cpu, nabla2_vec_cpu] =
      MakeAtlasField("nabla2_vec", mesh.edges().size());
  // term 1 and term 2 of nabla for debugging
  auto [nabla2t1_vec_F, nabla2t1_vec] =
      MakeAtlasField("nabla2t1_vec", mesh.edges().size());
  auto [nabla2t2_vec_F, nabla2t2_vec] =
      MakeAtlasField("nabla2t2_vec", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // intermediary fields (curl/rot and div of vec_e)
  //===------------------------------------------------------------------------------------------===//

  // rotation (more commonly curl) of vec_e on vertices
  auto [rot_vec_F_gpu, rot_vec_gpu] =
      MakeAtlasField("rot_vec_gpu", mesh.nodes().size());

  // divergence of vec_e on cells
  auto [div_vec_F_gpu, div_vec_gpu] =
      MakeAtlasField("div_vec_gpu", mesh.cells().size());

  auto [rot_vec_F_cpu, rot_vec_cpu] =
      MakeAtlasField("rot_vec_cpu", mesh.nodes().size());

  // divergence of vec_e on cells
  auto [div_vec_F_cpu, div_vec_cpu] =
      MakeAtlasField("div_vec_cpu", mesh.cells().size());

  //===------------------------------------------------------------------------------------------===//
  // sparse dimensions for computing intermediary fields
  //===------------------------------------------------------------------------------------------===//

  // needed for the computation of the curl/rotation. according to documentation
  // this needs to be:
  //
  // ! the appropriate dual cell based verts%edge_orientation
  // ! is required to obtain the correct value for the
  // ! application of Stokes theorem (which requires the scalar
  // ! product of the vector field with the tangent unit vectors
  // ! going around dual cell jv COUNTERCLOKWISE;
  // ! since the positive direction for the vec_e components is
  // ! not necessarily the one yelding counterclockwise rotation
  // ! around dual cell jv, a correction coefficient (equal to +-1)
  // ! is necessary, given by g%verts%edge_orientation
  auto [geofac_rot_F, geofac_rot] =
      MakeAtlasSparseField("geofac_rot", mesh.nodes().size(), edgesPerVertex);

  auto [edge_orientation_vertex_F, edge_orientation_vertex] =
      MakeAtlasSparseField("edge_orientation_vertex", mesh.nodes().size(),
                           edgesPerVertex);

  // needed for the computation of the curl/rotation. according to documentation
  // this needs to be:
  //
  //   ! ...the appropriate cell based edge_orientation is required to
  //   ! obtain the correct value for the application of Gauss theorem
  //   ! (which requires the scalar product of the vector field with the
  //   ! OUTWARD pointing unit vector with respect to cell jc; since the
  //   ! positive direction for the vector components is not necessarily
  //   ! the outward pointing one with respect to cell jc, a correction
  //   ! coefficient (equal to +-1) is necessary, given by
  //   ! ptr_patch%grid%cells%edge_orientation)
  auto [geofac_div_F, geofac_div] =
      MakeAtlasSparseField("geofac_div", mesh.cells().size(), edgesPerCell);

  auto [edge_orientation_cell_F, edge_orientation_cell] = MakeAtlasSparseField(
      "edge_orientation_cell", mesh.cells().size(), edgesPerCell);

  //===------------------------------------------------------------------------------------------===//
  // fields containing geometric information
  //===------------------------------------------------------------------------------------------===//
  auto [tangent_orientation_F, tangent_orientation] =
      MakeAtlasField("tangent_orientation", mesh.edges().size());
  auto [primal_edge_length_F, primal_edge_length] =
      MakeAtlasField("primal_edge_length", mesh.edges().size());
  auto [dual_edge_length_F, dual_edge_length] =
      MakeAtlasField("dual_edge_length", mesh.edges().size());
  auto [primal_normal_x_F, primal_normal_x] =
      MakeAtlasField("primal_normal_x", mesh.edges().size());
  auto [primal_normal_y_F, primal_normal_y] =
      MakeAtlasField("primal_normal_y", mesh.edges().size());
  auto [dual_normal_x_F, dual_normal_x] =
      MakeAtlasField("dual_normal_x", mesh.edges().size());
  auto [dual_normal_y_F, dual_normal_y] =
      MakeAtlasField("dual_normal_y", mesh.edges().size());
  auto [cell_area_F, cell_area] =
      MakeAtlasField("cell_area", mesh.cells().size());
  auto [dual_cell_area_F, dual_cell_area] =
      MakeAtlasField("dual_cell_area", mesh.nodes().size());

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on edges
  //===------------------------------------------------------------------------------------------===//
  for (int level = 0; level < k_size; level++) {
    for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      primal_edge_length(edgeIdx, level) = wrapper.edgeLength(mesh, edgeIdx);
      dual_edge_length(edgeIdx, level) = wrapper.dualEdgeLength(mesh, edgeIdx);
      tangent_orientation(edgeIdx, level) =
          wrapper.tangentOrientation(mesh, edgeIdx);
      auto [nx, ny] = wrapper.primalNormal(mesh, edgeIdx);
      primal_normal_x(edgeIdx, level) = nx;
      primal_normal_y(edgeIdx, level) = ny;
      // The primal normal, dual normal
      // forms a left-handed coordinate system
      dual_normal_x(edgeIdx, level) = ny;
      dual_normal_y(edgeIdx, level) = -nx;
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on cells
  //===------------------------------------------------------------------------------------------===//
  for (int level = 0; level < k_size; level++) {
    for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      cell_area(cellIdx, level) = wrapper.cellArea(mesh, cellIdx);
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on vertices
  //===------------------------------------------------------------------------------------------===//
  for (int level = 0; level < k_size; level++) {
    for (int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      dual_cell_area(nodeIdx, level) = wrapper.dualCellArea(mesh, nodeIdx);
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // input (spherical harmonics) and analytical solutions for div, curl and
  // Laplacian
  //===------------------------------------------------------------------------------------------===//

  auto sphericalHarmonic = [](double x,
                              double y) -> std::tuple<double, double> {
    return {0.25 * sqrt(105. / (2 * M_PI)) * cos(2 * x) * cos(y) * cos(y) *
                sin(y),
            0.5 * sqrt(15. / (2 * M_PI)) * cos(x) * cos(y) * sin(y)};
  };
  auto analyticalDivergence = [](double x, double y) {
    return -0.5 * (sqrt(105. / (2 * M_PI))) * sin(2 * x) * cos(y) * cos(y) *
               sin(y) +
           0.5 * sqrt(15. / (2 * M_PI)) * cos(x) *
               (cos(y) * cos(y) - sin(y) * sin(y));
  };
  auto analyticalCurl = [](double x, double y) {
    double c1 = 0.25 * sqrt(105. / (2 * M_PI));
    double c2 = 0.5 * sqrt(15. / (2 * M_PI));
    double dudy =
        c1 * cos(2 * x) * cos(y) * (cos(y) * cos(y) - 2 * sin(y) * sin(y));
    double dvdx = -c2 * cos(y) * sin(x) * sin(y);
    return dvdx - dudy;
  };
  auto analyticalLaplacian = [](double x,
                                double y) -> std::tuple<double, double> {
    double c1 = 0.25 * sqrt(105. / (2 * M_PI));
    double c2 = 0.5 * sqrt(15. / (2 * M_PI));
    return {-4 * c1 * cos(2 * x) * cos(y) * cos(y) * sin(y),
            -4 * c2 * cos(x) * sin(y) * cos(y)};
  };

  for (int level = 0; level < k_size; level++) {
    for (int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
      auto [u, v] = sphericalHarmonic(xm, ym);
      auto [lu, lv] = analyticalLaplacian(xm, ym);
      vec(edgeIdx, level) = primal_normal_x(edgeIdx, level) * u +
                            primal_normal_y(edgeIdx, level) * v;
      lapVecSol(edgeIdx, level) = primal_normal_x(edgeIdx, level) * lu +
                                  primal_normal_y(edgeIdx, level) * lv;
    }
    for (int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      auto [xm, ym] = wrapper.cellMidpoint(mesh, cellIdx);
      divVecSol(cellIdx, level) = analyticalDivergence(xm, ym);
    }
    for (int nodeIdx = 0; nodeIdx < mesh.nodes().size(); nodeIdx++) {
      auto [xm, ym] = wrapper.nodeLocation(nodeIdx);
      rotVecSol(nodeIdx, level) = analyticalCurl(xm, ym);
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // Init geometrical factors (sparse fields)
  //===------------------------------------------------------------------------------------------===//

  // init edge orientations for vertices and cells
  auto dot = [](const Vector &v1, const Vector &v2) {
    return std::get<0>(v1) * std::get<0>(v2) +
           std::get<1>(v1) * std::get<1>(v2);
  };

  //===------------------------------------------------------------------------------------------===//
  // stencil call
  //===------------------------------------------------------------------------------------------===//

  // Run the stencil (naive backend)
  nh_diffusion_fvm_wrapper_naive(mesh, k_size, vec, div_vec_cpu, rot_vec_cpu,
                                 nabla2t1_vec, nabla2t2_vec, nabla2_vec_cpu,
                                 primal_edge_length, dual_edge_length,
                                 tangent_orientation, geofac_rot, geofac_div);
  // Run the stencil (cuda backend)
  nh_diffusion_fvm_wrapper_cuda_run(
      mesh, k_size, vec, div_vec_gpu, rot_vec_gpu, nabla2t1_vec, nabla2t2_vec,
      nabla2_vec_gpu, primal_edge_length, dual_edge_length, tangent_orientation,
      geofac_rot, geofac_div);

  nh_diffusion_fvm_wrapper_cuda_copy_back(
      div_vec_gpu, rot_vec_gpu, nabla2t1_vec, nabla2t2_vec, nabla2_vec_gpu);

  UnstructuredVerifier verif;
  {
    auto div_vec_cpu_v = atlas::array::make_view<double, 2>(div_vec_F_cpu);
    auto div_vec_gpu_v = atlas::array::make_view<double, 2>(div_vec_F_gpu);
    auto div_vec_sol_v = atlas::array::make_view<double, 2>(divVecSol_F);
    EXPECT_TRUE(verif.compareArrayView(wrapper.innerCells(mesh), k_size,
                                       div_vec_cpu_v, div_vec_gpu_v))
        << "while comparing divergence";

    auto [Linf_gpu, L1_gpu, L2_gpu] = UnstructuredVerifier::getErrorNorms(
        wrapper.innerCells(mesh), k_size, div_vec_gpu_v, div_vec_sol_v);
    EXPECT_TRUE(Linf_gpu < 1. / nx * 1.2);
    EXPECT_TRUE(L1_gpu < 10 * 1. / nx);
    EXPECT_TRUE(L2_gpu < 2 * 1. / nx);
  }
  {
    auto rot_vec_cpu_v = atlas::array::make_view<double, 2>(rot_vec_F_cpu);
    auto rot_vec_gpu_v = atlas::array::make_view<double, 2>(rot_vec_F_gpu);
    auto rot_vec_sol_v = atlas::array::make_view<double, 2>(rotVecSol_F);
    EXPECT_TRUE(verif.compareArrayView(wrapper.innerNodes(mesh), k_size,
                                       rot_vec_cpu_v, rot_vec_gpu_v))
        << "while comparing curl";

    auto [Linf_gpu, L1_gpu, L2_gpu] = UnstructuredVerifier::getErrorNorms(
        wrapper.innerNodes(mesh), k_size, rot_vec_gpu_v, rot_vec_sol_v);
    EXPECT_TRUE(Linf_gpu < 1. / nx);
    EXPECT_TRUE(L1_gpu < 10 * 1. / nx);
    EXPECT_TRUE(L2_gpu < 2 * 1. / nx);
  }
  {
    auto lapl_vec_cpu_v = atlas::array::make_view<double, 2>(nabla2_vec_F_gpu);
    auto lapl_vec_gpu_v = atlas::array::make_view<double, 2>(nabla2_vec_F_cpu);
    auto lapl_vec_sol_v = atlas::array::make_view<double, 2>(lapVecSol_F);
    EXPECT_TRUE(verif.compareArrayView(wrapper.innerEdges(mesh), k_size,
                                       lapl_vec_cpu_v, lapl_vec_gpu_v))
        << "while comparing laplacian";

    auto [Linf_gpu, L1_gpu, L2_gpu] = UnstructuredVerifier::getErrorNorms(
        wrapper.innerEdges(mesh), k_size, lapl_vec_gpu_v, lapl_vec_sol_v);

    // constant error (not debendent on nx)
    EXPECT_TRUE(log(Linf_gpu) < 1);
    EXPECT_TRUE(log(L1_gpu) < 0);
    EXPECT_TRUE(log(L2_gpu) < 0);
  }
}
