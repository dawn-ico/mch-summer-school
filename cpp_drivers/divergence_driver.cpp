
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

#include "divergence_wrapper.h"

#include "UnstructuredVerifier.h"

template <typename T>
static int sgn(T val) {
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
  auto MakeAtlasField =
      [&](const std::string& name,
          int size) -> std::tuple<atlas::Field, atlasInterface::Field<dawn::float_type>> {
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

  auto MakeAtlasSparseField = [&](const std::string& name, int size, int sparseSize)
      -> std::tuple<atlas::Field, atlasInterface::SparseDimension<dawn::float_type>> {
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
  // input field (field we want to take the divergence of)
  //===------------------------------------------------------------------------------------------===//
  auto [vec_F, vec] = MakeAtlasField("vec", mesh.edges().size());

  //===------------------------------------------------------------------------------------------===//
  // control field holding the analytical solution for the divergence
  //===------------------------------------------------------------------------------------------===//
  auto [divVecSol_F, divVecSol] = MakeAtlasField("divVecSol", mesh.cells().size());

  //===------------------------------------------------------------------------------------------===//
  // output field (field containing the computed laplacian)
  //===------------------------------------------------------------------------------------------===//

  // divergence of vec_e on cells
  auto [div_vec_F, div_vec] = MakeAtlasField("div_vec", mesh.cells().size());

  //===------------------------------------------------------------------------------------------===//
  // sparse dimensions for computing intermediary fields
  //===------------------------------------------------------------------------------------------===//

  // the mesh may have arbitrary normals. we need to locally flip each normal pointing outside for
  // each cell. we can achieve this using an approriate sparse dimension

  auto [edge_orientation_cell_F, edge_orientation_cell] =
      MakeAtlasSparseField("edge_orientation_cell", mesh.cells().size(), edgesPerCell);

  //===------------------------------------------------------------------------------------------===//
  // fields containing geometric information
  //===------------------------------------------------------------------------------------------===//
  auto [edge_length_F, edge_length] = MakeAtlasField("primal_edge_length", mesh.edges().size());
  auto [normal_x_F, normal_x] = MakeAtlasField("primal_normal_x", mesh.edges().size());
  auto [normal_y_F, normal_y] = MakeAtlasField("primal_normal_y", mesh.edges().size());
  auto [cell_area_F, cell_area] = MakeAtlasField("cell_area", mesh.cells().size());

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on edges
  //===------------------------------------------------------------------------------------------===//
  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      edge_length(edgeIdx, level) = wrapper.edgeLength(mesh, edgeIdx);
      auto [nx, ny] = wrapper.primalNormal(mesh, edgeIdx);
      normal_x(edgeIdx, level) = nx;
      normal_y(edgeIdx, level) = ny;
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // initialize geometrical info on cells
  //===------------------------------------------------------------------------------------------===//
  for(int level = 0; level < k_size; level++) {
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      cell_area(cellIdx, level) = wrapper.cellArea(mesh, cellIdx);
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // Init geometrical factors (sparse fields)
  //===------------------------------------------------------------------------------------------===//

  // init edge orientations for vertices and cells
  auto dot = [](const Vector& v1, const Vector& v2) {
    return std::get<0>(v1) * std::get<0>(v2) + std::get<1>(v1) * std::get<1>(v2);
  };
  const atlas::mesh::HybridElements::Connectivity& cellEdgeConnectivity =
      mesh.cells().edge_connectivity();
  for(int level = 0; level < k_size; level++) {
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {

      auto [xm, ym] = wrapper.cellCircumcenter(mesh, cellIdx);

      for(int nbhIdx = 0; nbhIdx < edgesPerCell; nbhIdx++) {
        int edgeIdx = cellEdgeConnectivity(cellIdx, nbhIdx);
        auto [emX, emY] = wrapper.edgeMidpoint(mesh, edgeIdx);
        Vector toOutsdie{emX - xm, emY - ym};
        Vector primal = {normal_x(edgeIdx, level), normal_y(edgeIdx, level)};
        edge_orientation_cell(cellIdx, nbhIdx, level) = sgn(dot(toOutsdie, primal));
      }
      // explanation: the vector cellMidpoint -> edgeMidpoint is guaranteed to
      // point outside. The dot product checks if the edge normal has the same
      // orientation. edgeMidpoint is arbitrary, any point on e would work just
      // as well
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // input (spherical harmonics) and analytical solutions for divergence
  //===------------------------------------------------------------------------------------------===//

  auto sphericalHarmonic = [](double x, double y) -> std::tuple<double, double> {
    return {0.25 * sqrt(105. / (2 * M_PI)) * cos(2 * x) * cos(y) * cos(y) * sin(y),
            0.5 * sqrt(15. / (2 * M_PI)) * cos(x) * cos(y) * sin(y)};
  };
  auto analyticalDivergence = [](double x, double y) {
    return -0.5 * (sqrt(105. / (2 * M_PI))) * sin(2 * x) * cos(y) * cos(y) * sin(y) +
           0.5 * sqrt(15. / (2 * M_PI)) * cos(x) * (cos(y) * cos(y) - sin(y) * sin(y));
  };

  for(int level = 0; level < k_size; level++) {
    for(int edgeIdx = 0; edgeIdx < mesh.edges().size(); edgeIdx++) {
      auto [xm, ym] = wrapper.edgeMidpoint(mesh, edgeIdx);
      auto [u, v] = sphericalHarmonic(xm, ym);
      vec(edgeIdx, level) = normal_x(edgeIdx, level) * u + normal_y(edgeIdx, level) * v;
    }
    for(int cellIdx = 0; cellIdx < mesh.cells().size(); cellIdx++) {
      auto [xm, ym] = wrapper.cellMidpoint(mesh, cellIdx);
      divVecSol(cellIdx, level) = analyticalDivergence(xm, ym);
    }
  }

  //===------------------------------------------------------------------------------------------===//
  // stencil call
  //===------------------------------------------------------------------------------------------===//

  // Run the stencil (naive backend)
  divergence_wrapper_naive(mesh, k_size, vec, div_vec, edge_length, cell_area,
                           edge_orientation_cell);
  //   // Run the stencil (cuda backend)
  //   divergence_wrapper_cuda_run(mesh, k_size, vec, div_vec, edge_length, cell_area,
  //                               edge_orientation_cell);
  //   divergence_wrapper_cuda_copy_back(div_vec);

  UnstructuredVerifier verif;
  {
    auto div_vec_v = atlas::array::make_view<double, 2>(div_vec_F);
    auto div_vec_sol_v = atlas::array::make_view<double, 2>(divVecSol_F);
    auto [Linf, L1, L2] = UnstructuredVerifier::getErrorNorms(wrapper.innerCells(mesh), k_size,
                                                              div_vec_v, div_vec_sol_v);

    std::cout << "MEASURED ERRORS: L_inf " << Linf << " "
              << " L_1 " << L1 << " L_2 " << L2 << "\n";
    EXPECT_TRUE(Linf < 1. / nx * 1.2);
    EXPECT_TRUE(L1 < 10 * 1. / nx);
    EXPECT_TRUE(L2 < 2 * 1. / nx);
  }
}
