// ---------------------------------------------------------------------
//
// Copyright (C) 1998 - 2021 by the deal.II authors
//
// This file is part of the deal.II library.
//
// The deal.II library is free software; you can use it, redistribute
// it, and/or modify it under the terms of the GNU Lesser General
// Public License as published by the Free Software Foundation; either
// version 2.1 of the License, or (at your option) any later version.
// The full text of the license can be found in the file LICENSE.md at
// the top level directory of deal.II.
//
// ---------------------------------------------------------------------


#include <deal.II/base/geometry_info.h>
#include <deal.II/base/memory_consumption.h>

#include <deal.II/fe/mapping_q1.h>

#include <deal.II/grid/connectivity.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/magic_numbers.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_faces.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_levels.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <numeric>


DEAL_II_NAMESPACE_OPEN


namespace internal
{
  namespace TriangulationImplementation
  {
    NumberCache<1>::NumberCache()
      : n_levels(0)
      , n_lines(0)
      , n_active_lines(0)
    // all other fields are
    // default constructed
    {}



    std::size_t
    NumberCache<1>::memory_consumption() const
    {
      return (MemoryConsumption::memory_consumption(n_levels) +
              MemoryConsumption::memory_consumption(n_lines) +
              MemoryConsumption::memory_consumption(n_lines_level) +
              MemoryConsumption::memory_consumption(n_active_lines) +
              MemoryConsumption::memory_consumption(n_active_lines_level));
    }


    NumberCache<2>::NumberCache()
      : n_quads(0)
      , n_active_quads(0)
    // all other fields are
    // default constructed
    {}



    std::size_t
    NumberCache<2>::memory_consumption() const
    {
      return (NumberCache<1>::memory_consumption() +
              MemoryConsumption::memory_consumption(n_quads) +
              MemoryConsumption::memory_consumption(n_quads_level) +
              MemoryConsumption::memory_consumption(n_active_quads) +
              MemoryConsumption::memory_consumption(n_active_quads_level));
    }



    NumberCache<3>::NumberCache()
      : n_hexes(0)
      , n_active_hexes(0)
    // all other fields are
    // default constructed
    {}



    std::size_t
    NumberCache<3>::memory_consumption() const
    {
      return (NumberCache<2>::memory_consumption() +
              MemoryConsumption::memory_consumption(n_hexes) +
              MemoryConsumption::memory_consumption(n_hexes_level) +
              MemoryConsumption::memory_consumption(n_active_hexes) +
              MemoryConsumption::memory_consumption(n_active_hexes_level));
    }
  } // namespace TriangulationImplementation
} // namespace internal

// anonymous namespace for internal helper functions
namespace
{
  // return whether the given cell is
  // patch_level_1, i.e. determine
  // whether either all or none of
  // its children are further
  // refined. this function can only
  // be called for non-active cells.
  template <int dim, int spacedim>
  bool
  cell_is_patch_level_1(
    const TriaIterator<dealii::CellAccessor<dim, spacedim>> &cell)
  {
    Assert(cell->is_active() == false, ExcInternalError());

    unsigned int n_active_children = 0;
    for (unsigned int i = 0; i < cell->n_children(); ++i)
      if (cell->child(i)->is_active())
        ++n_active_children;

    return (n_active_children == 0) ||
           (n_active_children == cell->n_children());
  }



  // return, whether a given @p cell will be
  // coarsened, which is the case if all
  // children are active and have their coarsen
  // flag set. In case only part of the coarsen
  // flags are set, remove them.
  template <int dim, int spacedim>
  bool
  cell_will_be_coarsened(
    const TriaIterator<dealii::CellAccessor<dim, spacedim>> &cell)
  {
    // only cells with children should be
    // considered for coarsening

    if (cell->has_children())
      {
        unsigned int       children_to_coarsen = 0;
        const unsigned int n_children          = cell->n_children();

        for (unsigned int c = 0; c < n_children; ++c)
          if (cell->child(c)->is_active() && cell->child(c)->coarsen_flag_set())
            ++children_to_coarsen;
        if (children_to_coarsen == n_children)
          return true;
        else
          for (unsigned int c = 0; c < n_children; ++c)
            if (cell->child(c)->is_active())
              cell->child(c)->clear_coarsen_flag();
      }
    // no children, so no coarsening
    // possible. however, no children also
    // means that this cell will be in the same
    // state as if it had children and was
    // coarsened. So, what should we return -
    // false or true?
    // make sure we do not have to do this at
    // all...
    Assert(cell->has_children(), ExcInternalError());
    // ... and then simply return false
    return false;
  }


  // return, whether the face @p face_no of the
  // given @p cell will be refined after the
  // current refinement step, considering
  // refine and coarsen flags and considering
  // only those refinemnts that will be caused
  // by the neighboring cell.

  // this function is used on both active cells
  // and cells with children. on cells with
  // children it also of interest to know 'how'
  // the face will be refined. thus there is an
  // additional third argument @p
  // expected_face_ref_case returning just
  // that. be aware, that this variable will
  // only contain useful information if this
  // function is called for an active cell.
  //
  // thus, this is an internal function, users
  // should call one of the two alternatives
  // following below.
  template <int dim, int spacedim>
  bool
  face_will_be_refined_by_neighbor_internal(
    const TriaIterator<dealii::CellAccessor<dim, spacedim>> &cell,
    const unsigned int                                       face_no,
    RefinementCase<dim - 1> &expected_face_ref_case)
  {
    // first of all: set the default value for
    // expected_face_ref_case, which is no
    // refinement at all
    expected_face_ref_case = RefinementCase<dim - 1>::no_refinement;

    const typename Triangulation<dim, spacedim>::cell_iterator neighbor =
      cell->neighbor(face_no);

    // If we are at the boundary, there is no
    // neighbor which could refine the face
    if (neighbor.state() != IteratorState::valid)
      return false;

    if (neighbor->has_children())
      {
        // if the neighbor is refined, it may be
        // coarsened. if so, then it won't refine
        // the face, no matter what else happens
        if (cell_will_be_coarsened(neighbor))
          return false;
        else
          // if the neighbor is refined, then it
          // is also refined at our current
          // face. It will stay so without
          // coarsening, so return true in that
          // case.
          {
            expected_face_ref_case = cell->face(face_no)->refinement_case();
            return true;
          }
      }

    // now, the neighbor is not refined, but
    // perhaps it will be
    const RefinementCase<dim> nb_ref_flag = neighbor->refine_flag_set();
    if (nb_ref_flag != RefinementCase<dim>::no_refinement)
      {
        // now we need to know, which of the
        // neighbors faces points towards us
        const unsigned int neighbor_neighbor = cell->neighbor_face_no(face_no);
        // check, whether the cell will be
        // refined in a way that refines our
        // face
        const RefinementCase<dim - 1> face_ref_case =
          GeometryInfo<dim>::face_refinement_case(
            nb_ref_flag,
            neighbor_neighbor,
            neighbor->face_orientation(neighbor_neighbor),
            neighbor->face_flip(neighbor_neighbor),
            neighbor->face_rotation(neighbor_neighbor));
        if (face_ref_case != RefinementCase<dim - 1>::no_refinement)
          {
            const typename Triangulation<dim, spacedim>::face_iterator
                      neighbor_face   = neighbor->face(neighbor_neighbor);
            const int this_face_index = cell->face_index(face_no);

            // there are still two basic
            // possibilities here: the neighbor
            // might be coarser or as coarse
            // as we are
            if (neighbor_face->index() == this_face_index)
              // the neighbor is as coarse as
              // we are and will be refined at
              // the face of consideration, so
              // return true
              {
                expected_face_ref_case = face_ref_case;
                return true;
              }
            else
              {
                // the neighbor is coarser.
                // this is the most complicated
                // case. It might be, that the
                // neighbor's face will be
                // refined, but that we will
                // not see this, as we are
                // refined in a similar way.

                // so, the neighbor's face must
                // have children. check, if our
                // cell's face is one of these
                // (it could also be a
                // grand_child)
                for (unsigned int c = 0; c < neighbor_face->n_children(); ++c)
                  if (neighbor_face->child_index(c) == this_face_index)
                    {
                      // if the flagged refine
                      // case of the face is a
                      // subset or the same as
                      // the current refine case,
                      // then the face, as seen
                      // from our cell, won't be
                      // refined by the neighbor
                      if ((neighbor_face->refinement_case() | face_ref_case) ==
                          neighbor_face->refinement_case())
                        return false;
                      else
                        {
                          // if we are active, we
                          // must be an
                          // anisotropic child
                          // and the coming
                          // face_ref_case is
                          // isotropic. Thus,
                          // from our cell we
                          // will see exactly the
                          // opposite refine case
                          // that the face has
                          // now...
                          Assert(
                            face_ref_case ==
                              RefinementCase<dim - 1>::isotropic_refinement,
                            ExcInternalError());
                          expected_face_ref_case =
                            ~neighbor_face->refinement_case();
                          return true;
                        }
                    }

                // so, obviously we were not
                // one of the children, but a
                // grandchild. This is only
                // possible in 3d.
                Assert(dim == 3, ExcInternalError());
                // In that case, however, no
                // matter what the neighbor
                // does, it won't be finer
                // after the next refinement
                // step.
                return false;
              }
          } // if face will be refined
      }     // if neighbor is flagged for refinement

    // no cases left, so the neighbor will not
    // refine the face
    return false;
  }

  // version of above function for both active
  // and non-active cells
  template <int dim, int spacedim>
  bool
  face_will_be_refined_by_neighbor(
    const TriaIterator<dealii::CellAccessor<dim, spacedim>> &cell,
    const unsigned int                                       face_no)
  {
    RefinementCase<dim - 1> dummy = RefinementCase<dim - 1>::no_refinement;
    return face_will_be_refined_by_neighbor_internal(cell, face_no, dummy);
  }

  // version of above function for active cells
  // only. Additionally returning the refine
  // case (to come) of the face under
  // consideration
  template <int dim, int spacedim>
  bool
  face_will_be_refined_by_neighbor(
    const TriaActiveIterator<dealii::CellAccessor<dim, spacedim>> &cell,
    const unsigned int                                             face_no,
    RefinementCase<dim - 1> &expected_face_ref_case)
  {
    return face_will_be_refined_by_neighbor_internal(cell,
                                                     face_no,
                                                     expected_face_ref_case);
  }



  template <int dim, int spacedim>
  bool
  satisfies_level1_at_vertex_rule(
    const Triangulation<dim, spacedim> &triangulation)
  {
    std::vector<unsigned int> min_adjacent_cell_level(
      triangulation.n_vertices(), triangulation.n_levels());
    std::vector<unsigned int> max_adjacent_cell_level(
      triangulation.n_vertices(), 0);

    for (const auto &cell : triangulation.active_cell_iterators())
      for (const unsigned int v : cell->vertex_indices())
        {
          min_adjacent_cell_level[cell->vertex_index(v)] =
            std::min<unsigned int>(
              min_adjacent_cell_level[cell->vertex_index(v)], cell->level());
          max_adjacent_cell_level[cell->vertex_index(v)] =
            std::max<unsigned int>(
              min_adjacent_cell_level[cell->vertex_index(v)], cell->level());
        }

    for (unsigned int k = 0; k < triangulation.n_vertices(); ++k)
      if (triangulation.vertex_used(k))
        if (max_adjacent_cell_level[k] - min_adjacent_cell_level[k] > 1)
          return false;
    return true;
  }



  /**
   * Fill the vector @p line_cell_count
   * needed by @p delete_children with the
   * number of cells bounded by a given
   * line.
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  count_cells_bounded_by_line(const Triangulation<dim, spacedim> &triangulation)
  {
    if (dim >= 2)
      {
        std::vector<unsigned int> line_cell_count(triangulation.n_raw_lines(),
                                                  0);
        for (const auto &cell : triangulation.cell_iterators())
          for (unsigned int l = 0; l < cell->n_lines(); ++l)
            ++line_cell_count[cell->line_index(l)];
        return line_cell_count;
      }
    else
      return std::vector<unsigned int>();
  }



  /**
   * Fill the vector @p quad_cell_count
   * needed by @p delete_children with the
   * number of cells bounded by a given
   * quad.
   */
  template <int dim, int spacedim>
  std::vector<unsigned int>
  count_cells_bounded_by_quad(const Triangulation<dim, spacedim> &triangulation)
  {
    if (dim >= 3)
      {
        std::vector<unsigned int> quad_cell_count(triangulation.n_raw_quads(),
                                                  0);
        for (const auto &cell : triangulation.cell_iterators())
          for (unsigned int q : cell->face_indices())
            ++quad_cell_count[cell->quad_index(q)];
        return quad_cell_count;
      }
    else
      return {};
  }



  /**
   * A set of three functions that
   * reorder the data given to
   * create_triangulation_compatibility
   * from the "classic" to the
   * "current" format of vertex
   * numbering of cells and
   * faces. These functions do the
   * reordering of their arguments
   * in-place.
   */
  void
  reorder_compatibility(const std::vector<CellData<1>> &, const SubCellData &)
  {
    // nothing to do here: the format
    // hasn't changed for 1d
  }


  void
  reorder_compatibility(std::vector<CellData<2>> &cells, const SubCellData &)
  {
    for (auto &cell : cells)
      if (cell.vertices.size() == GeometryInfo<2>::vertices_per_cell)
        std::swap(cell.vertices[2], cell.vertices[3]);
  }


  void
  reorder_compatibility(std::vector<CellData<3>> &cells,
                        SubCellData &             subcelldata)
  {
    unsigned int tmp[GeometryInfo<3>::vertices_per_cell];
    for (auto &cell : cells)
      if (cell.vertices.size() == GeometryInfo<3>::vertices_per_cell)
        {
          for (const unsigned int i : GeometryInfo<3>::vertex_indices())
            tmp[i] = cell.vertices[i];
          for (const unsigned int i : GeometryInfo<3>::vertex_indices())
            cell.vertices[GeometryInfo<3>::ucd_to_deal[i]] = tmp[i];
        }

    // now points in boundary quads
    for (auto &boundary_quad : subcelldata.boundary_quads)
      if (boundary_quad.vertices.size() == GeometryInfo<2>::vertices_per_cell)
        std::swap(boundary_quad.vertices[2], boundary_quad.vertices[3]);
  }



  /**
   * Return the index of the vertex
   * in the middle of this object,
   * if it exists. In order to
   * exist, the object needs to be
   * refined - for 2D and 3D it
   * needs to be refined
   * isotropically or else the
   * anisotropic children have to
   * be refined again. If the
   * middle vertex does not exist,
   * return
   * <tt>numbers::invalid_unsigned_int</tt>.
   *
   * This function should not really be
   * used in application programs.
   */
  template <int dim, int spacedim>
  unsigned int
  middle_vertex_index(
    const typename Triangulation<dim, spacedim>::line_iterator &line)
  {
    if (line->has_children())
      return line->child(0)->vertex_index(1);
    return numbers::invalid_unsigned_int;
  }


  template <int dim, int spacedim>
  unsigned int
  middle_vertex_index(
    const typename Triangulation<dim, spacedim>::quad_iterator &quad)
  {
    switch (static_cast<unsigned char>(quad->refinement_case()))
      {
        case RefinementCase<2>::cut_x:
          return middle_vertex_index<dim, spacedim>(quad->child(0)->line(1));
          break;
        case RefinementCase<2>::cut_y:
          return middle_vertex_index<dim, spacedim>(quad->child(0)->line(3));
          break;
        case RefinementCase<2>::cut_xy:
          return quad->child(0)->vertex_index(3);
          break;
        default:
          break;
      }
    return numbers::invalid_unsigned_int;
  }


  template <int dim, int spacedim>
  unsigned int
  middle_vertex_index(
    const typename Triangulation<dim, spacedim>::hex_iterator &hex)
  {
    switch (static_cast<unsigned char>(hex->refinement_case()))
      {
        case RefinementCase<3>::cut_x:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->quad(1));
          break;
        case RefinementCase<3>::cut_y:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->quad(3));
          break;
        case RefinementCase<3>::cut_z:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->quad(5));
          break;
        case RefinementCase<3>::cut_xy:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->line(11));
          break;
        case RefinementCase<3>::cut_xz:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->line(5));
          break;
        case RefinementCase<3>::cut_yz:
          return middle_vertex_index<dim, spacedim>(hex->child(0)->line(7));
          break;
        case RefinementCase<3>::cut_xyz:
          return hex->child(0)->vertex_index(7);
          break;
        default:
          break;
      }
    return numbers::invalid_unsigned_int;
  }


  /**
   * Collect all coarse mesh cells
   * with at least one vertex at
   * which the determinant of the
   * Jacobian is zero or
   * negative. This is the function
   * for the case dim!=spacedim,
   * where we can not determine
   * whether a cell is twisted as it
   * may, for example, discretize a
   * manifold with a twist.
   */
  template <class TRIANGULATION>
  inline typename TRIANGULATION::DistortedCellList
  collect_distorted_coarse_cells(const TRIANGULATION &)
  {
    return typename TRIANGULATION::DistortedCellList();
  }



  /**
   * Collect all coarse mesh cells
   * with at least one vertex at
   * which the determinant of the
   * Jacobian is zero or
   * negative. This is the function
   * for the case dim==spacedim.
   */
  template <int dim>
  inline typename Triangulation<dim, dim>::DistortedCellList
  collect_distorted_coarse_cells(const Triangulation<dim, dim> &triangulation)
  {
    typename Triangulation<dim, dim>::DistortedCellList distorted_cells;
    for (const auto &cell : triangulation.cell_iterators_on_level(0))
      {
        Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
        for (const unsigned int i : GeometryInfo<dim>::vertex_indices())
          vertices[i] = cell->vertex(i);

        Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
        GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);

        for (const unsigned int i : GeometryInfo<dim>::vertex_indices())
          if (determinants[i] <= 1e-9 * std::pow(cell->diameter(), 1. * dim))
            {
              distorted_cells.distorted_cells.push_back(cell);
              break;
            }
      }

    return distorted_cells;
  }


  /**
   * Return whether any of the
   * children of the given cell is
   * distorted or not. This is the
   * function for dim==spacedim.
   */
  template <int dim>
  bool
  has_distorted_children(
    const typename Triangulation<dim, dim>::cell_iterator &cell)
  {
    Assert(cell->has_children(), ExcInternalError());

    for (unsigned int c = 0; c < cell->n_children(); ++c)
      {
        Point<dim> vertices[GeometryInfo<dim>::vertices_per_cell];
        for (const unsigned int i : GeometryInfo<dim>::vertex_indices())
          vertices[i] = cell->child(c)->vertex(i);

        Tensor<0, dim> determinants[GeometryInfo<dim>::vertices_per_cell];
        GeometryInfo<dim>::alternating_form_at_vertices(vertices, determinants);

        for (const unsigned int i : GeometryInfo<dim>::vertex_indices())
          if (determinants[i] <=
              1e-9 * std::pow(cell->child(c)->diameter(), 1. * dim))
            return true;
      }

    return false;
  }


  /**
   * Function for dim!=spacedim. As
   * for
   * collect_distorted_coarse_cells,
   * there is nothing that we can do
   * in this case.
   */
  template <int dim, int spacedim>
  bool
  has_distorted_children(
    const typename Triangulation<dim, spacedim>::cell_iterator &)
  {
    return false;
  }


  template <int dim, int spacedim>
  void
  update_periodic_face_map_recursively(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell_1,
    const typename Triangulation<dim, spacedim>::cell_iterator &cell_2,
    unsigned int                                                n_face_1,
    unsigned int                                                n_face_2,
    const std::bitset<3> &                                      orientation,
    typename std::map<
      std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                unsigned int>,
      std::pair<std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                          unsigned int>,
                std::bitset<3>>> &periodic_face_map)
  {
    using FaceIterator = typename Triangulation<dim, spacedim>::face_iterator;
    const FaceIterator face_1 = cell_1->face(n_face_1);
    const FaceIterator face_2 = cell_2->face(n_face_2);

    const bool face_orientation = orientation[0];
    const bool face_flip        = orientation[1];
    const bool face_rotation    = orientation[2];

    Assert((dim != 1) || (face_orientation == true && face_flip == false &&
                          face_rotation == false),
           ExcMessage("The supplied orientation "
                      "(face_orientation, face_flip, face_rotation) "
                      "is invalid for 1D"));

    Assert((dim != 2) || (face_orientation == true && face_rotation == false),
           ExcMessage("The supplied orientation "
                      "(face_orientation, face_flip, face_rotation) "
                      "is invalid for 2D"));

    Assert(face_1 != face_2, ExcMessage("face_1 and face_2 are equal!"));

    Assert(face_1->at_boundary() && face_2->at_boundary(),
           ExcMessage("Periodic faces must be on the boundary"));

    // Check if the requirement that each edge can only have at most one hanging
    // node, and as a consequence neighboring cells can differ by at most
    // one refinement level is enforced. In 1d, there are no hanging nodes and
    // so neighboring cells can differ by more than one refinement level.
    Assert(dim == 1 || std::abs(cell_1->level() - cell_2->level()) < 2,
           ExcInternalError());

    // insert periodic face pair for both cells
    using CellFace =
      std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                unsigned int>;
    const CellFace                            cell_face_1(cell_1, n_face_1);
    const CellFace                            cell_face_2(cell_2, n_face_2);
    const std::pair<CellFace, std::bitset<3>> cell_face_orientation_2(
      cell_face_2, orientation);

    const std::pair<CellFace, std::pair<CellFace, std::bitset<3>>>
      periodic_faces(cell_face_1, cell_face_orientation_2);

    // Only one periodic neighbor is allowed
    Assert(periodic_face_map.count(cell_face_1) == 0, ExcInternalError());
    periodic_face_map.insert(periodic_faces);

    if (dim == 1)
      {
        if (cell_1->has_children())
          {
            if (cell_2->has_children())
              {
                update_periodic_face_map_recursively<dim, spacedim>(
                  cell_1->child(n_face_1),
                  cell_2->child(n_face_2),
                  n_face_1,
                  n_face_2,
                  orientation,
                  periodic_face_map);
              }
            else // only face_1 has children
              {
                update_periodic_face_map_recursively<dim, spacedim>(
                  cell_1->child(n_face_1),
                  cell_2,
                  n_face_1,
                  n_face_2,
                  orientation,
                  periodic_face_map);
              }
          }
      }
    else // dim == 2 || dim == 3
      {
        // A lookup table on how to go through the child cells depending on the
        // orientation:
        // see Documentation of GeometryInfo for details

        static const int lookup_table_2d[2][2] =
          //               flip:
          {
            {0, 1}, // false
            {1, 0}  // true
          };

        static const int lookup_table_3d[2][2][2][4] =
          //                           orientation flip  rotation
          {{{
              {0, 2, 1, 3}, // false       false false
              {2, 3, 0, 1}  // false       false true
            },
            {
              {3, 1, 2, 0}, // false       true  false
              {1, 0, 3, 2}  // false       true  true
            }},
           {{
              {0, 1, 2, 3}, // true        false false
              {1, 3, 0, 2}  // true        false true
            },
            {
              {3, 2, 1, 0}, // true        true  false
              {2, 0, 3, 1}  // true        true  true
            }}};

        if (cell_1->has_children())
          {
            if (cell_2->has_children())
              {
                // In the case that both faces have children, we loop over all
                // children and apply update_periodic_face_map_recursively
                // recursively:

                Assert(face_1->n_children() ==
                           GeometryInfo<dim>::max_children_per_face &&
                         face_2->n_children() ==
                           GeometryInfo<dim>::max_children_per_face,
                       ExcNotImplemented());

                for (unsigned int i = 0;
                     i < GeometryInfo<dim>::max_children_per_face;
                     ++i)
                  {
                    // Lookup the index for the second face
                    unsigned int j = 0;
                    switch (dim)
                      {
                        case 2:
                          j = lookup_table_2d[face_flip][i];
                          break;
                        case 3:
                          j = lookup_table_3d[face_orientation][face_flip]
                                             [face_rotation][i];
                          break;
                        default:
                          AssertThrow(false, ExcNotImplemented());
                      }

                    // find subcell ids that belong to the subface indices
                    unsigned int child_cell_1 =
                      GeometryInfo<dim>::child_cell_on_face(
                        cell_1->refinement_case(),
                        n_face_1,
                        i,
                        cell_1->face_orientation(n_face_1),
                        cell_1->face_flip(n_face_1),
                        cell_1->face_rotation(n_face_1),
                        face_1->refinement_case());
                    unsigned int child_cell_2 =
                      GeometryInfo<dim>::child_cell_on_face(
                        cell_2->refinement_case(),
                        n_face_2,
                        j,
                        cell_2->face_orientation(n_face_2),
                        cell_2->face_flip(n_face_2),
                        cell_2->face_rotation(n_face_2),
                        face_2->refinement_case());

                    Assert(cell_1->child(child_cell_1)->face(n_face_1) ==
                             face_1->child(i),
                           ExcInternalError());
                    Assert(cell_2->child(child_cell_2)->face(n_face_2) ==
                             face_2->child(j),
                           ExcInternalError());

                    // precondition: subcell has the same orientation as cell
                    // (so that the face numbers coincide) recursive call
                    update_periodic_face_map_recursively<dim, spacedim>(
                      cell_1->child(child_cell_1),
                      cell_2->child(child_cell_2),
                      n_face_1,
                      n_face_2,
                      orientation,
                      periodic_face_map);
                  }
              }
            else // only face_1 has children
              {
                for (unsigned int i = 0;
                     i < GeometryInfo<dim>::max_children_per_face;
                     ++i)
                  {
                    // find subcell ids that belong to the subface indices
                    unsigned int child_cell_1 =
                      GeometryInfo<dim>::child_cell_on_face(
                        cell_1->refinement_case(),
                        n_face_1,
                        i,
                        cell_1->face_orientation(n_face_1),
                        cell_1->face_flip(n_face_1),
                        cell_1->face_rotation(n_face_1),
                        face_1->refinement_case());

                    // recursive call
                    update_periodic_face_map_recursively<dim, spacedim>(
                      cell_1->child(child_cell_1),
                      cell_2,
                      n_face_1,
                      n_face_2,
                      orientation,
                      periodic_face_map);
                  }
              }
          }
      }
  }


} // end of anonymous namespace


namespace internal
{
  namespace TriangulationImplementation
  {
    // make sure that if in the following we
    // write Triangulation<dim,spacedim>
    // we mean the *class*
    // dealii::Triangulation, not the
    // enclosing namespace
    // internal::TriangulationImplementation
    using dealii::Triangulation;

    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException1(ExcGridHasInvalidCell,
                   int,
                   << "Something went wrong when making cell " << arg1
                   << ". Read the docs and the source code "
                   << "for more information.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException1(ExcInternalErrorOnCell,
                   int,
                   << "Something went wrong upon construction of cell "
                   << arg1);
    /**
     * A cell was entered which has
     * negative measure. In most
     * cases, this is due to a wrong
     * order of the vertices of the
     * cell.
     *
     * @ingroup Exceptions
     */
    DeclException1(ExcCellHasNegativeMeasure,
                   int,
                   << "Cell " << arg1
                   << " has negative measure. This typically "
                   << "indicates some distortion in the cell, or a mistakenly "
                   << "swapped pair of vertices in the input to "
                   << "Triangulation::create_triangulation().");
    /**
     * A cell is created with a
     * vertex number exceeding the
     * vertex array.
     *
     * @ingroup Exceptions
     */
    DeclException3(ExcInvalidVertexIndex,
                   int,
                   int,
                   int,
                   << "Error while creating cell " << arg1
                   << ": the vertex index " << arg2 << " must be between 0 and "
                   << arg3 << ".");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException2(ExcLineInexistant,
                   int,
                   int,
                   << "While trying to assign a boundary indicator to a line: "
                   << "the line with end vertices " << arg1 << " and " << arg2
                   << " does not exist.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException4(ExcQuadInexistant,
                   int,
                   int,
                   int,
                   int,
                   << "While trying to assign a boundary indicator to a quad: "
                   << "the quad with bounding lines " << arg1 << ", " << arg2
                   << ", " << arg3 << ", " << arg4 << " does not exist.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException3(
      ExcInteriorLineCantBeBoundary,
      int,
      int,
      types::boundary_id,
      << "The input data for creating a triangulation contained "
      << "information about a line with indices " << arg1 << " and " << arg2
      << " that is described to have boundary indicator "
      << static_cast<int>(arg3)
      << ". However, this is an internal line not located on the "
      << "boundary. You cannot assign a boundary indicator to it." << std::endl
      << std::endl
      << "If this happened at a place where you call "
      << "Triangulation::create_triangulation() yourself, you need "
      << "to check the SubCellData object you pass to this function."
      << std::endl
      << std::endl
      << "If this happened in a place where you are reading a mesh "
      << "from a file, then you need to investigate why such a line "
      << "ended up in the input file. A typical case is a geometry "
      << "that consisted of multiple parts and for which the mesh "
      << "generator program assumes that the interface between "
      << "two parts is a boundary when that isn't supposed to be "
      << "the case, or where the mesh generator simply assigns "
      << "'geometry indicators' to lines at the perimeter of "
      << "a part that are not supposed to be interpreted as "
      << "'boundary indicators'.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException5(
      ExcInteriorQuadCantBeBoundary,
      int,
      int,
      int,
      int,
      types::boundary_id,
      << "The input data for creating a triangulation contained "
      << "information about a quad with indices " << arg1 << ", " << arg2
      << ", " << arg3 << ", and " << arg4
      << " that is described to have boundary indicator "
      << static_cast<int>(arg5)
      << ". However, this is an internal quad not located on the "
      << "boundary. You cannot assign a boundary indicator to it." << std::endl
      << std::endl
      << "If this happened at a place where you call "
      << "Triangulation::create_triangulation() yourself, you need "
      << "to check the SubCellData object you pass to this function."
      << std::endl
      << std::endl
      << "If this happened in a place where you are reading a mesh "
      << "from a file, then you need to investigate why such a quad "
      << "ended up in the input file. A typical case is a geometry "
      << "that consisted of multiple parts and for which the mesh "
      << "generator program assumes that the interface between "
      << "two parts is a boundary when that isn't supposed to be "
      << "the case, or where the mesh generator simply assigns "
      << "'geometry indicators' to quads at the surface of "
      << "a part that are not supposed to be interpreted as "
      << "'boundary indicators'.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException2(
      ExcMultiplySetLineInfoOfLine,
      int,
      int,
      << "In SubCellData the line info of the line with vertex indices " << arg1
      << " and " << arg2 << " appears more than once. "
      << "This is not allowed.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException3(
      ExcInconsistentLineInfoOfLine,
      int,
      int,
      std::string,
      << "In SubCellData the line info of the line with vertex indices " << arg1
      << " and " << arg2 << " appears multiple times with different (valid) "
      << arg3 << ". This is not allowed.");
    /**
     * Exception
     * @ingroup Exceptions
     */
    DeclException5(
      ExcInconsistentQuadInfoOfQuad,
      int,
      int,
      int,
      int,
      std::string,
      << "In SubCellData the quad info of the quad with line indices " << arg1
      << ", " << arg2 << ", " << arg3 << " and " << arg4
      << " appears multiple times with different (valid) " << arg5
      << ". This is not allowed.");

    /*
     * Reserve space for TriaFaces. Details:
     *
     * Reserve space for line_orientations.
     *
     * @note Used only for dim=3.
     */
    void
    reserve_space(TriaFaces &        tria_faces,
                  const unsigned int new_quads_in_pairs,
                  const unsigned int new_quads_single)
    {
      AssertDimension(tria_faces.dim, 3);

      Assert(new_quads_in_pairs % 2 == 0, ExcInternalError());

      unsigned int next_free_single = 0;
      unsigned int next_free_pair   = 0;

      // count the number of objects, of unused single objects and of
      // unused pairs of objects
      unsigned int n_quads          = 0;
      unsigned int n_unused_pairs   = 0;
      unsigned int n_unused_singles = 0;
      for (unsigned int i = 0; i < tria_faces.quads.used.size(); ++i)
        {
          if (tria_faces.quads.used[i])
            ++n_quads;
          else if (i + 1 < tria_faces.quads.used.size())
            {
              if (tria_faces.quads.used[i + 1])
                {
                  ++n_unused_singles;
                  if (next_free_single == 0)
                    next_free_single = i;
                }
              else
                {
                  ++n_unused_pairs;
                  if (next_free_pair == 0)
                    next_free_pair = i;
                  ++i;
                }
            }
          else
            ++n_unused_singles;
        }
      Assert(n_quads + 2 * n_unused_pairs + n_unused_singles ==
               tria_faces.quads.used.size(),
             ExcInternalError());

      // how many single quads are needed in addition to n_unused_quads?
      const int additional_single_quads = new_quads_single - n_unused_singles;

      unsigned int new_size =
        tria_faces.quads.used.size() + new_quads_in_pairs - 2 * n_unused_pairs;
      if (additional_single_quads > 0)
        new_size += additional_single_quads;

      // see above...
      if (new_size > tria_faces.quads.n_objects())
        {
          // reserve the field of the derived class
          tria_faces.quads_line_orientations.reserve(
            new_size * GeometryInfo<2>::lines_per_cell);
          tria_faces.quads_line_orientations.insert(
            tria_faces.quads_line_orientations.end(),
            new_size * GeometryInfo<2>::lines_per_cell -
              tria_faces.quads_line_orientations.size(),
            1u);

          tria_faces.quad_reference_cell.reserve(new_size);
          tria_faces.quad_reference_cell.insert(
            tria_faces.quad_reference_cell.end(),
            new_size - tria_faces.quad_reference_cell.size(),
            dealii::ReferenceCells::Quadrilateral);
        }
    }



    /**
     * Reserve space for TriaLevel. Details:
     *
     * Reserve enough space to accommodate @p total_cells cells on this
     * level. Since there are no @p used flags on this level, you have to
     * give the total number of cells, not only the number of newly to
     * accommodate ones, like in the <tt>TriaLevel<N>::reserve_space</tt>
     * functions, with <tt>N>0</tt>.
     *
     * Since the number of neighbors per cell depends on the dimensions, you
     * have to pass that additionally.
     */

    void
    reserve_space(TriaLevel &        tria_level,
                  const unsigned int total_cells,
                  const unsigned int dimension,
                  const unsigned int space_dimension)
    {
      // we need space for total_cells cells. Maybe we have more already
      // with those cells which are unused, so only allocate new space if
      // needed.
      //
      // note that all arrays should have equal sizes (checked by
      // @p{monitor_memory}
      if (total_cells > tria_level.refine_flags.size())
        {
          tria_level.refine_flags.reserve(total_cells);
          tria_level.refine_flags.insert(tria_level.refine_flags.end(),
                                         total_cells -
                                           tria_level.refine_flags.size(),
                                         /*RefinementCase::no_refinement=*/0);

          tria_level.coarsen_flags.reserve(total_cells);
          tria_level.coarsen_flags.insert(tria_level.coarsen_flags.end(),
                                          total_cells -
                                            tria_level.coarsen_flags.size(),
                                          false);

          tria_level.active_cell_indices.reserve(total_cells);
          tria_level.active_cell_indices.insert(
            tria_level.active_cell_indices.end(),
            total_cells - tria_level.active_cell_indices.size(),
            numbers::invalid_unsigned_int);

          tria_level.subdomain_ids.reserve(total_cells);
          tria_level.subdomain_ids.insert(tria_level.subdomain_ids.end(),
                                          total_cells -
                                            tria_level.subdomain_ids.size(),
                                          0);

          tria_level.level_subdomain_ids.reserve(total_cells);
          tria_level.level_subdomain_ids.insert(
            tria_level.level_subdomain_ids.end(),
            total_cells - tria_level.level_subdomain_ids.size(),
            0);

          tria_level.global_active_cell_indices.reserve(total_cells);
          tria_level.global_active_cell_indices.insert(
            tria_level.global_active_cell_indices.end(),
            total_cells - tria_level.global_active_cell_indices.size(),
            numbers::invalid_dof_index);

          tria_level.global_level_cell_indices.reserve(total_cells);
          tria_level.global_level_cell_indices.insert(
            tria_level.global_level_cell_indices.end(),
            total_cells - tria_level.global_level_cell_indices.size(),
            numbers::invalid_dof_index);

          if (dimension < space_dimension)
            {
              tria_level.direction_flags.reserve(total_cells);
              tria_level.direction_flags.insert(
                tria_level.direction_flags.end(),
                total_cells - tria_level.direction_flags.size(),
                true);
            }
          else
            tria_level.direction_flags.clear();

          tria_level.parents.reserve((total_cells + 1) / 2);
          tria_level.parents.insert(tria_level.parents.end(),
                                    (total_cells + 1) / 2 -
                                      tria_level.parents.size(),
                                    -1);

          tria_level.neighbors.reserve(total_cells * (2 * dimension));
          tria_level.neighbors.insert(tria_level.neighbors.end(),
                                      total_cells * (2 * dimension) -
                                        tria_level.neighbors.size(),
                                      std::make_pair(-1, -1));

          if (tria_level.dim == 2 || tria_level.dim == 3)
            {
              const unsigned int max_faces_per_cell = 2 * dimension;
              tria_level.face_orientations.reserve(total_cells *
                                                   max_faces_per_cell);
              tria_level.face_orientations.insert(
                tria_level.face_orientations.end(),
                total_cells * max_faces_per_cell -
                  tria_level.face_orientations.size(),
                1u);

              tria_level.reference_cell.reserve(total_cells);
              tria_level.reference_cell.insert(
                tria_level.reference_cell.end(),
                total_cells - tria_level.reference_cell.size(),
                tria_level.dim == 2 ? dealii::ReferenceCells::Quadrilateral :
                                      dealii::ReferenceCells::Hexahedron);
            }
        }
    }



    /**
     * Exception
     */
    DeclException2(ExcMemoryInexact,
                   int,
                   int,
                   << "The containers have sizes " << arg1 << " and " << arg2
                   << ", which is not as expected.");

    /**
     * Check the memory consistency of the different containers. Should only
     * be called with the preprocessor flag @p DEBUG set. The function
     * should be called from the functions of the higher TriaLevel classes.
     */
    void
    monitor_memory(const TriaLevel &  tria_level,
                   const unsigned int true_dimension)
    {
      (void)tria_level;
      (void)true_dimension;
      Assert(2 * true_dimension * tria_level.refine_flags.size() ==
               tria_level.neighbors.size(),
             ExcMemoryInexact(tria_level.refine_flags.size(),
                              tria_level.neighbors.size()));
      Assert(2 * true_dimension * tria_level.coarsen_flags.size() ==
               tria_level.neighbors.size(),
             ExcMemoryInexact(tria_level.coarsen_flags.size(),
                              tria_level.neighbors.size()));
    }



    /**
     * Reserve space for TriaObjects. Details:
     *
     * Assert that enough space is allocated to accommodate
     * <code>new_objs_in_pairs</code> new objects, stored in pairs, plus
     * <code>new_obj_single</code> stored individually. This function does
     * not only call <code>vector::reserve()</code>, but does really append
     * the needed elements.
     *
     * In 2D e.g. refined lines have to be stored in pairs, whereas new
     * lines in the interior of refined cells can be stored as single lines.
     */
    void
    reserve_space(TriaObjects &      tria_objects,
                  const unsigned int new_objects_in_pairs,
                  const unsigned int new_objects_single = 0)
    {
      if (tria_objects.structdim <= 2)
        {
          Assert(new_objects_in_pairs % 2 == 0, ExcInternalError());

          tria_objects.next_free_single               = 0;
          tria_objects.next_free_pair                 = 0;
          tria_objects.reverse_order_next_free_single = false;

          // count the number of objects, of unused single objects and of
          // unused pairs of objects
          unsigned int n_objects        = 0;
          unsigned int n_unused_pairs   = 0;
          unsigned int n_unused_singles = 0;
          for (unsigned int i = 0; i < tria_objects.used.size(); ++i)
            {
              if (tria_objects.used[i])
                ++n_objects;
              else if (i + 1 < tria_objects.used.size())
                {
                  if (tria_objects.used[i + 1])
                    {
                      ++n_unused_singles;
                      if (tria_objects.next_free_single == 0)
                        tria_objects.next_free_single = i;
                    }
                  else
                    {
                      ++n_unused_pairs;
                      if (tria_objects.next_free_pair == 0)
                        tria_objects.next_free_pair = i;
                      ++i;
                    }
                }
              else
                ++n_unused_singles;
            }
          Assert(n_objects + 2 * n_unused_pairs + n_unused_singles ==
                   tria_objects.used.size(),
                 ExcInternalError());

          // how many single objects are needed in addition to
          // n_unused_objects?
          const int additional_single_objects =
            new_objects_single - n_unused_singles;

          unsigned int new_size = tria_objects.used.size() +
                                  new_objects_in_pairs - 2 * n_unused_pairs;
          if (additional_single_objects > 0)
            new_size += additional_single_objects;

          // only allocate space if necessary
          if (new_size > tria_objects.n_objects())
            {
              const unsigned int max_faces_per_cell =
                2 * tria_objects.structdim;
              const unsigned int max_children_per_cell =
                1 << tria_objects.structdim;

              tria_objects.cells.reserve(new_size * max_faces_per_cell);
              tria_objects.cells.insert(tria_objects.cells.end(),
                                        (new_size - tria_objects.n_objects()) *
                                          max_faces_per_cell,
                                        -1);

              tria_objects.used.reserve(new_size);
              tria_objects.used.insert(tria_objects.used.end(),
                                       new_size - tria_objects.used.size(),
                                       false);

              tria_objects.user_flags.reserve(new_size);
              tria_objects.user_flags.insert(tria_objects.user_flags.end(),
                                             new_size -
                                               tria_objects.user_flags.size(),
                                             false);

              const unsigned int factor = max_children_per_cell / 2;
              tria_objects.children.reserve(factor * new_size);
              tria_objects.children.insert(tria_objects.children.end(),
                                           factor * new_size -
                                             tria_objects.children.size(),
                                           -1);

              if (tria_objects.structdim > 1)
                {
                  tria_objects.refinement_cases.reserve(new_size);
                  tria_objects.refinement_cases.insert(
                    tria_objects.refinement_cases.end(),
                    new_size - tria_objects.refinement_cases.size(),
                    /*RefinementCase::no_refinement=*/0);
                }

              // first reserve, then resize. Otherwise the std library can
              // decide to allocate more entries.
              tria_objects.boundary_or_material_id.reserve(new_size);
              tria_objects.boundary_or_material_id.resize(new_size);

              tria_objects.user_data.reserve(new_size);
              tria_objects.user_data.resize(new_size);

              tria_objects.manifold_id.reserve(new_size);
              tria_objects.manifold_id.insert(tria_objects.manifold_id.end(),
                                              new_size -
                                                tria_objects.manifold_id.size(),
                                              numbers::flat_manifold_id);
            }

          if (n_unused_singles == 0)
            {
              tria_objects.next_free_single               = new_size - 1;
              tria_objects.reverse_order_next_free_single = true;
            }
        }
      else
        {
          const unsigned int new_hexes = new_objects_in_pairs;

          const unsigned int new_size =
            new_hexes + std::count(tria_objects.used.begin(),
                                   tria_objects.used.end(),
                                   true);

          // see above...
          if (new_size > tria_objects.n_objects())
            {
              const unsigned int max_faces_per_cell =
                2 * tria_objects.structdim;

              tria_objects.cells.reserve(new_size * max_faces_per_cell);
              tria_objects.cells.insert(tria_objects.cells.end(),
                                        (new_size - tria_objects.n_objects()) *
                                          max_faces_per_cell,
                                        -1);

              tria_objects.used.reserve(new_size);
              tria_objects.used.insert(tria_objects.used.end(),
                                       new_size - tria_objects.used.size(),
                                       false);

              tria_objects.user_flags.reserve(new_size);
              tria_objects.user_flags.insert(tria_objects.user_flags.end(),
                                             new_size -
                                               tria_objects.user_flags.size(),
                                             false);

              tria_objects.children.reserve(4 * new_size);
              tria_objects.children.insert(tria_objects.children.end(),
                                           4 * new_size -
                                             tria_objects.children.size(),
                                           -1);

              // for the following fields, we know exactly how many elements
              // we need, so first reserve then resize (resize itself, at least
              // with some compiler libraries, appears to round up the size it
              // actually reserves)
              tria_objects.boundary_or_material_id.reserve(new_size);
              tria_objects.boundary_or_material_id.resize(new_size);

              tria_objects.manifold_id.reserve(new_size);
              tria_objects.manifold_id.insert(tria_objects.manifold_id.end(),
                                              new_size -
                                                tria_objects.manifold_id.size(),
                                              numbers::flat_manifold_id);

              tria_objects.user_data.reserve(new_size);
              tria_objects.user_data.resize(new_size);

              tria_objects.refinement_cases.reserve(new_size);
              tria_objects.refinement_cases.insert(
                tria_objects.refinement_cases.end(),
                new_size - tria_objects.refinement_cases.size(),
                /*RefinementCase::no_refinement=*/0);
            }
          tria_objects.next_free_single = tria_objects.next_free_pair = 0;
        }
    }



    /**
     * Check the memory consistency of the different containers. Should only
     * be called with the preprocessor flag @p DEBUG set. The function
     * should be called from the functions of the higher TriaLevel classes.
     */
    void
    monitor_memory(const TriaObjects &tria_object, const unsigned int)
    {
      Assert(tria_object.n_objects() == tria_object.used.size(),
             ExcMemoryInexact(tria_object.n_objects(),
                              tria_object.used.size()));
      Assert(tria_object.n_objects() == tria_object.user_flags.size(),
             ExcMemoryInexact(tria_object.n_objects(),
                              tria_object.user_flags.size()));
      Assert(tria_object.n_objects() ==
               tria_object.boundary_or_material_id.size(),
             ExcMemoryInexact(tria_object.n_objects(),
                              tria_object.boundary_or_material_id.size()));
      Assert(tria_object.n_objects() == tria_object.manifold_id.size(),
             ExcMemoryInexact(tria_object.n_objects(),
                              tria_object.manifold_id.size()));
      Assert(tria_object.n_objects() == tria_object.user_data.size(),
             ExcMemoryInexact(tria_object.n_objects(),
                              tria_object.user_data.size()));

      if (tria_object.structdim == 1)
        {
          Assert(1 * tria_object.n_objects() == tria_object.children.size(),
                 ExcMemoryInexact(tria_object.n_objects(),
                                  tria_object.children.size()));
        }
      else if (tria_object.structdim == 2)
        {
          Assert(2 * tria_object.n_objects() == tria_object.children.size(),
                 ExcMemoryInexact(tria_object.n_objects(),
                                  tria_object.children.size()));
        }
      else if (tria_object.structdim == 3)
        {
          Assert(4 * tria_object.n_objects() == tria_object.children.size(),
                 ExcMemoryInexact(tria_object.n_objects(),
                                  tria_object.children.size()));
        }
    }



    /**
     * An interface for algorithms that implement Triangulation-specific tasks
     * related to creation, refinement, and coarsening.
     */
    template <int dim, int spacedim>
    class Policy
    {
    public:
      /**
       * Destructor.
       */
      virtual ~Policy() = default;

      /**
       * Update neighbors.
       */
      virtual void
      update_neighbors(Triangulation<dim, spacedim> &tria) = 0;

      /**
       * Delete children of given cell.
       */
      virtual void
      delete_children(
        Triangulation<dim, spacedim> &                        triangulation,
        typename Triangulation<dim, spacedim>::cell_iterator &cell,
        std::vector<unsigned int> &                           line_cell_count,
        std::vector<unsigned int> &quad_cell_count) = 0;

      /**
       * Execute refinement.
       */
      virtual typename Triangulation<dim, spacedim>::DistortedCellList
      execute_refinement(Triangulation<dim, spacedim> &triangulation,
                         const bool check_for_distorted_cells) = 0;

      /**
       * Prevent distorted boundary cells.
       */
      virtual void
      prevent_distorted_boundary_cells(
        Triangulation<dim, spacedim> &triangulation) = 0;

      /**
       * Prepare refinement.
       */
      virtual void
      prepare_refinement_dim_dependent(
        Triangulation<dim, spacedim> &triangulation) = 0;

      /**
       * Check if coarsening is allowed for the given cell.
       */
      virtual bool
      coarsening_allowed(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell) = 0;

      /**
       * A sort of virtual copy constructor, this function returns a copy of
       * the policy object. Derived classes need to override the function here
       * in this base class and return an object of the same type as the derived
       * class.
       */
      virtual std::unique_ptr<Policy<dim, spacedim>>
      clone() = 0;
    };



    /**
     * A simple implementation of the interface Policy. It simply delegates the
     * task to the functions with the same name provided by class specified by
     * the template argument T.
     */
    template <int dim, int spacedim, typename T>
    class PolicyWrapper : public Policy<dim, spacedim>
    {
    public:
      void
      update_neighbors(Triangulation<dim, spacedim> &tria) override
      {
        T::update_neighbors(tria);
      }

      void
      delete_children(
        Triangulation<dim, spacedim> &                        tria,
        typename Triangulation<dim, spacedim>::cell_iterator &cell,
        std::vector<unsigned int> &                           line_cell_count,
        std::vector<unsigned int> &quad_cell_count) override
      {
        T::delete_children(tria, cell, line_cell_count, quad_cell_count);
      }

      typename Triangulation<dim, spacedim>::DistortedCellList
      execute_refinement(Triangulation<dim, spacedim> &triangulation,
                         const bool check_for_distorted_cells) override
      {
        return T::execute_refinement(triangulation, check_for_distorted_cells);
      }

      void
      prevent_distorted_boundary_cells(
        Triangulation<dim, spacedim> &triangulation) override
      {
        T::prevent_distorted_boundary_cells(triangulation);
      }

      void
      prepare_refinement_dim_dependent(
        Triangulation<dim, spacedim> &triangulation) override
      {
        T::prepare_refinement_dim_dependent(triangulation);
      }

      bool
      coarsening_allowed(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell)
        override
      {
        return T::template coarsening_allowed<dim, spacedim>(cell);
      }

      std::unique_ptr<Policy<dim, spacedim>>
      clone() override
      {
        return std::make_unique<PolicyWrapper<dim, spacedim, T>>();
      }
    };



    /**
     * A class into which we put many of the functions that implement
     * functionality of the Triangulation class. The main reason for this
     * class is as follows: the majority of the functions in Triangulation
     * need to be implemented differently for dim==1, dim==2, and
     * dim==3. However, their implementation is largly independent of the
     * spacedim template parameter. So we would like to write things like
     *
     * @code
     * template <int spacedim>
     * void Triangulation<1,spacedim>::create_triangulation (...) {...}
     * @endcode
     *
     * Unfortunately, C++ doesn't allow this: member functions of class
     * templates have to be either not specialized at all, or fully
     * specialized. No partial specialization is allowed. One possible
     * solution would be to just duplicate the bodies of the functions and
     * have equally implemented functions
     *
     * @code
     * template <>
     * void Triangulation<1,1>::create_triangulation (...) {...}
     *
     * template <>
     * void Triangulation<1,2>::create_triangulation (...) {...}
     * @endcode
     *
     * but that is clearly an unsatisfactory solution. Rather, what we do
     * is introduce the current Implementation class in which we can write
     * these functions as member templates over spacedim, i.e. we can have
     *
     * @code
     * template <int dim_, int spacedim_>
     * template <int spacedim>
     * void Triangulation<dim_,spacedim_>::Implementation::
     *            create_triangulation (...,
     *                                  Triangulation<1,spacedim> &tria ) {...}
     * @endcode
     *
     * The outer template parameters are here unused, only the inner
     * ones are of real interest.
     *
     * One may ask why we put these functions into an class rather
     * than an anonymous namespace, for example?
     *
     * First, these implementation functions need to be friends of the
     * Triangulation class. It is simpler to make the entire class a friend
     * rather than listing all members of an implementation namespace as
     * friends of the Triangulation class (there is no such thing as a "friend
     * namespace XXX" directive).
     *
     * Ideally, we would make this class a member class of the
     * Triangulation<dim,spacedim> class, since then our implementation
     * functions have immediate access to the alias and static functions of
     * the surrounding Triangulation class. I.e., we do not have to write
     * "typename Triangulation<dim,spacedim>::active_cell_iterator" but can
     * write "active_cell_iterator" right away. This is, in fact, the way it was
     * implemented first, but we ran into a bug in gcc4.0:
     * @code
     *  class Triangulation {
     *    struct Implementation;
     *    friend class TriaAccessor;
     *  };
     *
     *  class TriaAccessor {
     *    struct Implementation;
     *    friend class Triangulation;
     *  };
     * @endcode
     *
     * Here, friendship (per C++ standard) is supposed to extend to all members
     * of the befriended class, including its 'Implementation' member class. But
     * gcc4.0 gets this wrong: the members of Triangulation::Implementation are
     * not friends of TriaAccessor and the other way around. Ideally, one would
     * fix this by saying
     * @code
     *  class Triangulation {
     *    struct Implementation;
     *    friend class TriaAccessor;
     *    friend class TriaAccessor::Implementation;   // **
     *  };
     *
     *  class TriaAccessor {
     *    struct Implementation;
     *    friend class Triangulation;
     *    friend class Triangulation::Implementation;
     *  };
     * @endcode
     * but that's not legal because in ** we don't know yet that TriaAccessor
     * has a member class Implementation and so we can't make it a friend. The
     * only way forward at this point was to make Implementation a class in the
     * internal namespace so that we can forward declare it and make it a friend
     * of the respective other outer class -- not quite what we wanted but the
     * only way I could see to make it work...
     */
    struct Implementation
    {
      /**
       * For a given Triangulation, update that part of the number
       * cache that relates to lines. For 1d, we have to deal with the
       * fact that lines have levels, whereas for higher dimensions
       * they do not.
       *
       * The second argument indicates for how many levels the
       * Triangulation has objects, though the highest levels need not
       * contain active cells if they have previously all been
       * coarsened away.
       */
      template <int dim, int spacedim>
      static void
      compute_number_cache(
        const Triangulation<dim, spacedim> &                   triangulation,
        const unsigned int                                     level_objects,
        internal::TriangulationImplementation::NumberCache<1> &number_cache)
      {
        using line_iterator =
          typename Triangulation<dim, spacedim>::line_iterator;

        number_cache.n_levels = 0;
        if (level_objects > 0)
          // find the last level on which there are used cells
          for (unsigned int level = 0; level < level_objects; ++level)
            if (triangulation.begin(level) != triangulation.end(level))
              number_cache.n_levels = level + 1;

        // no cells at all?
        Assert(number_cache.n_levels > 0, ExcInternalError());

        //---------------------------------
        // update the number of lines on the different levels in the
        // cache
        number_cache.n_lines        = 0;
        number_cache.n_active_lines = 0;

        // for 1d, lines have levels so take count the objects per
        // level and globally
        if (dim == 1)
          {
            number_cache.n_lines_level.resize(number_cache.n_levels);
            number_cache.n_active_lines_level.resize(number_cache.n_levels);

            for (unsigned int level = 0; level < number_cache.n_levels; ++level)
              {
                // count lines on this level
                number_cache.n_lines_level[level]        = 0;
                number_cache.n_active_lines_level[level] = 0;

                line_iterator line = triangulation.begin_line(level),
                              endc =
                                (level == number_cache.n_levels - 1 ?
                                   line_iterator(triangulation.end_line()) :
                                   triangulation.begin_line(level + 1));
                for (; line != endc; ++line)
                  {
                    ++number_cache.n_lines_level[level];
                    if (line->has_children() == false)
                      ++number_cache.n_active_lines_level[level];
                  }

                // update total number of lines
                number_cache.n_lines += number_cache.n_lines_level[level];
                number_cache.n_active_lines +=
                  number_cache.n_active_lines_level[level];
              }
          }
        else
          {
            // for dim>1, there are no levels for lines
            number_cache.n_lines_level.clear();
            number_cache.n_active_lines_level.clear();

            line_iterator line = triangulation.begin_line(),
                          endc = triangulation.end_line();
            for (; line != endc; ++line)
              {
                ++number_cache.n_lines;
                if (line->has_children() == false)
                  ++number_cache.n_active_lines;
              }
          }
      }

      /**
       * For a given Triangulation, update that part of the number
       * cache that relates to quads. For 2d, we have to deal with the
       * fact that quads have levels, whereas for higher dimensions
       * they do not.
       *
       * The second argument indicates for how many levels the
       * Triangulation has objects, though the highest levels need not
       * contain active cells if they have previously all been
       * coarsened away.
       *
       * At the beginning of the function, we call the respective
       * function to update the number cache for lines.
       */
      template <int dim, int spacedim>
      static void
      compute_number_cache(
        const Triangulation<dim, spacedim> &                   triangulation,
        const unsigned int                                     level_objects,
        internal::TriangulationImplementation::NumberCache<2> &number_cache)
      {
        // update lines and n_levels in number_cache. since we don't
        // access any of these numbers, we can do this in the
        // background
        Threads::Task<void> update_lines = Threads::new_task(
          static_cast<
            void (*)(const Triangulation<dim, spacedim> &,
                     const unsigned int,
                     internal::TriangulationImplementation::NumberCache<1> &)>(
            &compute_number_cache<dim, spacedim>),
          triangulation,
          level_objects,
          static_cast<internal::TriangulationImplementation::NumberCache<1> &>(
            number_cache));

        using quad_iterator =
          typename Triangulation<dim, spacedim>::quad_iterator;

        //---------------------------------
        // update the number of quads on the different levels in the
        // cache
        number_cache.n_quads        = 0;
        number_cache.n_active_quads = 0;

        // for 2d, quads have levels so take count the objects per
        // level and globally
        if (dim == 2)
          {
            // count the number of levels; the function we called above
            // on a separate Task for lines also does this and puts it into
            // number_cache.n_levels, but this datum may not yet be
            // available as we call the function on a separate task
            unsigned int n_levels = 0;
            if (level_objects > 0)
              // find the last level on which there are used cells
              for (unsigned int level = 0; level < level_objects; ++level)
                if (triangulation.begin(level) != triangulation.end(level))
                  n_levels = level + 1;

            number_cache.n_quads_level.resize(n_levels);
            number_cache.n_active_quads_level.resize(n_levels);

            for (unsigned int level = 0; level < n_levels; ++level)
              {
                // count quads on this level
                number_cache.n_quads_level[level]        = 0;
                number_cache.n_active_quads_level[level] = 0;

                quad_iterator quad = triangulation.begin_quad(level),
                              endc =
                                (level == n_levels - 1 ?
                                   quad_iterator(triangulation.end_quad()) :
                                   triangulation.begin_quad(level + 1));
                for (; quad != endc; ++quad)
                  {
                    ++number_cache.n_quads_level[level];
                    if (quad->has_children() == false)
                      ++number_cache.n_active_quads_level[level];
                  }

                // update total number of quads
                number_cache.n_quads += number_cache.n_quads_level[level];
                number_cache.n_active_quads +=
                  number_cache.n_active_quads_level[level];
              }
          }
        else
          {
            // for dim>2, there are no levels for quads
            number_cache.n_quads_level.clear();
            number_cache.n_active_quads_level.clear();

            quad_iterator quad = triangulation.begin_quad(),
                          endc = triangulation.end_quad();
            for (; quad != endc; ++quad)
              {
                ++number_cache.n_quads;
                if (quad->has_children() == false)
                  ++number_cache.n_active_quads;
              }
          }

        // wait for the background computation for lines
        update_lines.join();
      }

      /**
       * For a given Triangulation, update that part of the number
       * cache that relates to hexes. For 3d, we have to deal with the
       * fact that hexes have levels, whereas for higher dimensions
       * they do not.
       *
       * The second argument indicates for how many levels the
       * Triangulation has objects, though the highest levels need not
       * contain active cells if they have previously all been
       * coarsened away.
       *
       * At the end of the function, we call the respective function
       * to update the number cache for quads, which will in turn call
       * the respective function for lines.
       */
      template <int dim, int spacedim>
      static void
      compute_number_cache(
        const Triangulation<dim, spacedim> &                   triangulation,
        const unsigned int                                     level_objects,
        internal::TriangulationImplementation::NumberCache<3> &number_cache)
      {
        // update quads, lines and n_levels in number_cache. since we
        // don't access any of these numbers, we can do this in the
        // background
        Threads::Task<void> update_quads_and_lines = Threads::new_task(
          static_cast<
            void (*)(const Triangulation<dim, spacedim> &,
                     const unsigned int,
                     internal::TriangulationImplementation::NumberCache<2> &)>(
            &compute_number_cache<dim, spacedim>),
          triangulation,
          level_objects,
          static_cast<internal::TriangulationImplementation::NumberCache<2> &>(
            number_cache));

        using hex_iterator =
          typename Triangulation<dim, spacedim>::hex_iterator;

        //---------------------------------
        // update the number of hexes on the different levels in the
        // cache
        number_cache.n_hexes        = 0;
        number_cache.n_active_hexes = 0;

        // for 3d, hexes have levels so take count the objects per
        // level and globally
        if (dim == 3)
          {
            // count the number of levels; the function we called
            // above on a separate Task for quads (recursively, via
            // the lines function) also does this and puts it into
            // number_cache.n_levels, but this datum may not yet be
            // available as we call the function on a separate task
            unsigned int n_levels = 0;
            if (level_objects > 0)
              // find the last level on which there are used cells
              for (unsigned int level = 0; level < level_objects; ++level)
                if (triangulation.begin(level) != triangulation.end(level))
                  n_levels = level + 1;

            number_cache.n_hexes_level.resize(n_levels);
            number_cache.n_active_hexes_level.resize(n_levels);

            for (unsigned int level = 0; level < n_levels; ++level)
              {
                // count hexes on this level
                number_cache.n_hexes_level[level]        = 0;
                number_cache.n_active_hexes_level[level] = 0;

                hex_iterator hex  = triangulation.begin_hex(level),
                             endc = (level == n_levels - 1 ?
                                       hex_iterator(triangulation.end_hex()) :
                                       triangulation.begin_hex(level + 1));
                for (; hex != endc; ++hex)
                  {
                    ++number_cache.n_hexes_level[level];
                    if (hex->has_children() == false)
                      ++number_cache.n_active_hexes_level[level];
                  }

                // update total number of hexes
                number_cache.n_hexes += number_cache.n_hexes_level[level];
                number_cache.n_active_hexes +=
                  number_cache.n_active_hexes_level[level];
              }
          }
        else
          {
            // for dim>3, there are no levels for hexes
            number_cache.n_hexes_level.clear();
            number_cache.n_active_hexes_level.clear();

            hex_iterator hex  = triangulation.begin_hex(),
                         endc = triangulation.end_hex();
            for (; hex != endc; ++hex)
              {
                ++number_cache.n_hexes;
                if (hex->has_children() == false)
                  ++number_cache.n_active_hexes;
              }
          }

        // wait for the background computation for quads
        update_quads_and_lines.join();
      }



      template <int spacedim>
      static void
      update_neighbors(Triangulation<1, spacedim> &)
      {}


      template <int dim, int spacedim>
      static void
      update_neighbors(Triangulation<dim, spacedim> &triangulation)
      {
        // each face can be neighbored on two sides
        // by cells. according to the face's
        // intrinsic normal we define the left
        // neighbor as the one for which the face
        // normal points outward, and store that
        // one first; the second one is then
        // the right neighbor for which the
        // face normal points inward. This
        // information depends on the type of cell
        // and local number of face for the
        // 'standard ordering and orientation' of
        // faces and then on the face_orientation
        // information for the real mesh. Set up a
        // table to have fast access to those
        // offsets (0 for left and 1 for
        // right). Some of the values are invalid
        // as they reference too large face
        // numbers, but we just leave them at a
        // zero value.
        //
        // Note, that in 2d for lines as faces the
        // normal direction given in the
        // GeometryInfo class is not consistent. We
        // thus define here that the normal for a
        // line points to the right if the line
        // points upwards.
        //
        // There is one more point to
        // consider, however: if we have
        // dim<spacedim, then we may have
        // cases where cells are
        // inverted. In effect, both
        // cells think they are the left
        // neighbor of an edge, for
        // example, which leads us to
        // forget neighborship
        // information (a case that shows
        // this is
        // codim_one/hanging_nodes_02). We
        // store whether a cell is
        // inverted using the
        // direction_flag, so if a cell
        // has a false direction_flag,
        // then we need to invert our
        // selection whether we are a
        // left or right neighbor in all
        // following computations.
        //
        // first index:  dimension (minus 2)
        // second index: local face index
        // third index:  face_orientation (false and true)
        static const unsigned int left_right_offset[2][6][2] = {
          // quadrilateral
          {{0, 1},  // face 0, face_orientation = false and true
           {1, 0},  // face 1, face_orientation = false and true
           {1, 0},  // face 2, face_orientation = false and true
           {0, 1},  // face 3, face_orientation = false and true
           {0, 0},  // face 4, invalid face
           {0, 0}}, // face 5, invalid face
                    // hexahedron
          {{0, 1}, {1, 0}, {0, 1}, {1, 0}, {0, 1}, {1, 0}}};

        // now create a vector of the two active
        // neighbors (left and right) for each face
        // and fill it by looping over all cells. For
        // cases with anisotropic refinement and more
        // then one cell neighboring at a given side
        // of the face we will automatically get the
        // active one on the highest level as we loop
        // over cells from lower levels first.
        const typename Triangulation<dim, spacedim>::cell_iterator dummy;
        std::vector<typename Triangulation<dim, spacedim>::cell_iterator>
          adjacent_cells(2 * triangulation.n_raw_faces(), dummy);

        for (const auto &cell : triangulation.cell_iterators())
          for (auto f : cell->face_indices())
            {
              const typename Triangulation<dim, spacedim>::face_iterator face =
                cell->face(f);

              const unsigned int offset =
                (cell->direction_flag() ?
                   left_right_offset[dim - 2][f][cell->face_orientation(f)] :
                   1 -
                     left_right_offset[dim - 2][f][cell->face_orientation(f)]);

              adjacent_cells[2 * face->index() + offset] = cell;

              // if this cell is not refined, but the
              // face is, then we'll have to set our
              // cell as neighbor for the child faces
              // as well. Fortunately the normal
              // orientation of children will be just
              // the same.
              if (dim == 2)
                {
                  if (cell->is_active() && face->has_children())
                    {
                      adjacent_cells[2 * face->child(0)->index() + offset] =
                        cell;
                      adjacent_cells[2 * face->child(1)->index() + offset] =
                        cell;
                    }
                }
              else // -> dim == 3
                {
                  // We need the same as in 2d
                  // here. Furthermore, if the face is
                  // refined with cut_x or cut_y then
                  // those children again in the other
                  // direction, and if this cell is
                  // refined isotropically (along the
                  // face) then the neighbor will
                  // (probably) be refined as cut_x or
                  // cut_y along the face. For those
                  // neighboring children cells, their
                  // neighbor will be the current,
                  // inactive cell, as our children are
                  // too fine to be neighbors. Catch that
                  // case by also acting on inactive
                  // cells with isotropic refinement
                  // along the face. If the situation
                  // described is not present, the data
                  // will be overwritten later on when we
                  // visit cells on finer levels, so no
                  // harm will be done.
                  if (face->has_children() &&
                      (cell->is_active() ||
                       GeometryInfo<dim>::face_refinement_case(
                         cell->refinement_case(), f) ==
                         RefinementCase<dim - 1>::isotropic_refinement))
                    {
                      for (unsigned int c = 0; c < face->n_children(); ++c)
                        adjacent_cells[2 * face->child(c)->index() + offset] =
                          cell;
                      if (face->child(0)->has_children())
                        {
                          adjacent_cells[2 * face->child(0)->child(0)->index() +
                                         offset] = cell;
                          adjacent_cells[2 * face->child(0)->child(1)->index() +
                                         offset] = cell;
                        }
                      if (face->child(1)->has_children())
                        {
                          adjacent_cells[2 * face->child(1)->child(0)->index() +
                                         offset] = cell;
                          adjacent_cells[2 * face->child(1)->child(1)->index() +
                                         offset] = cell;
                        }
                    } // if cell active and face refined
                }     // else -> dim==3
            }         // for all faces of all cells

        // now loop again over all cells and set the
        // corresponding neighbor cell. Note, that we
        // have to use the opposite of the
        // left_right_offset in this case as we want
        // the offset of the neighbor, not our own.
        for (const auto &cell : triangulation.cell_iterators())
          for (auto f : cell->face_indices())
            {
              const unsigned int offset =
                (cell->direction_flag() ?
                   left_right_offset[dim - 2][f][cell->face_orientation(f)] :
                   1 -
                     left_right_offset[dim - 2][f][cell->face_orientation(f)]);
              cell->set_neighbor(
                f, adjacent_cells[2 * cell->face(f)->index() + 1 - offset]);
            }
      }


      /**
       * Create a triangulation from given data.
       */
      template <int dim, int spacedim>
      static void
      create_triangulation(const std::vector<Point<spacedim>> &vertices,
                           const std::vector<CellData<dim>> &  cells,
                           const SubCellData &                 subcelldata,
                           Triangulation<dim, spacedim> &      tria)
      {
        AssertThrow(vertices.size() > 0, ExcMessage("No vertices given"));
        AssertThrow(cells.size() > 0, ExcMessage("No cells given"));

        // Check that all cells have positive volume.
#ifndef _MSC_VER
        // TODO: The following code does not compile with MSVC. Find a way
        // around it
        if (dim == spacedim)
          for (unsigned int cell_no = 0; cell_no < cells.size(); ++cell_no)
            {
              // If we should check for distorted cells, then we permit them
              // to exist. If a cell has negative measure, then it must be
              // distorted (the converse is not necessarily true); hence
              // throw an exception if no such cells should exist.
              if (tria.check_for_distorted_cells)
                {
                  const double cell_measure = GridTools::cell_measure<spacedim>(
                    vertices,
                    ArrayView<const unsigned int>(cells[cell_no].vertices));
                  AssertThrow(cell_measure > 0, ExcGridHasInvalidCell(cell_no));
                }
            }
#endif

        // clear old content
        tria.levels.clear();
        tria.levels.push_back(
          std::make_unique<
            dealii::internal::TriangulationImplementation::TriaLevel>(dim));

        if (dim > 1)
          tria.faces = std::make_unique<
            dealii::internal::TriangulationImplementation::TriaFaces>(dim);

        // copy vertices
        tria.vertices = vertices;
        tria.vertices_used.assign(vertices.size(), true);

        // compute connectivity
        const auto connectivity   = build_connectivity<unsigned int>(cells);
        const unsigned int n_cell = cells.size();

        // TriaObjects: lines
        if (dim >= 2)
          {
            auto &lines_0 = tria.faces->lines; // data structure to be filled

            // get connectivity between quads and lines
            const auto &       crs     = connectivity.entity_to_entities(1, 0);
            const unsigned int n_lines = crs.ptr.size() - 1;

            // allocate memory
            reserve_space_(lines_0, n_lines);

            // loop over lines
            for (unsigned int line = 0; line < n_lines; ++line)
              for (unsigned int i = crs.ptr[line], j = 0; i < crs.ptr[line + 1];
                   ++i, ++j)
                lines_0.cells[line * GeometryInfo<1>::faces_per_cell + j] =
                  crs.col[i]; // set vertex indices
          }

        // TriaObjects: quads
        if (dim == 3)
          {
            auto &quads_0 = tria.faces->quads; // data structures to be filled
            auto &faces   = *tria.faces;

            // get connectivity between quads and lines
            const auto &       crs     = connectivity.entity_to_entities(2, 1);
            const unsigned int n_quads = crs.ptr.size() - 1;

            // allocate memory
            reserve_space_(quads_0, n_quads);
            reserve_space_(faces, 2 /*structdim*/, n_quads);

            // loop over all quads -> entity type, line indices/orientations
            for (unsigned int q = 0, k = 0; q < n_quads; ++q)
              {
                // set entity type of quads
                faces.quad_reference_cell[q] = connectivity.entity_types(2)[q];

                // loop over all its lines
                for (unsigned int i = crs.ptr[q], j = 0; i < crs.ptr[q + 1];
                     ++i, ++j, ++k)
                  {
                    // set line index
                    quads_0.cells[q * GeometryInfo<2>::faces_per_cell + j] =
                      crs.col[i];

                    // set line orientations
                    faces.quads_line_orientations
                      [q * GeometryInfo<2>::faces_per_cell + j] =
                      connectivity.entity_orientations(1)[k];
                  }
              }
          }

        // TriaObjects/TriaLevel: cell
        {
          auto &cells_0 = tria.levels[0]->cells; // data structure to be filled
          auto &level   = *tria.levels[0];

          // get connectivity between cells/faces and cells/cells
          const auto &crs = connectivity.entity_to_entities(dim, dim - 1);
          const auto &nei = connectivity.entity_to_entities(dim, dim);

          // in 2D optional: since in in pure QUAD meshes same line
          // orientations can be guaranteed
          const bool orientation_needed =
            dim == 3 ||
            (dim == 2 &&
             std::any_of(connectivity.entity_orientations(1).begin(),
                         connectivity.entity_orientations(1).end(),
                         [](const auto &i) { return i == 0; }));

          // allocate memory
          reserve_space_(cells_0, n_cell);
          reserve_space_(level, spacedim, n_cell, orientation_needed);

          // loop over all cells
          for (unsigned int cell = 0; cell < n_cell; ++cell)
            {
              // set material ids
              cells_0.boundary_or_material_id[cell].material_id =
                cells[cell].material_id;

              // set manifold ids
              cells_0.manifold_id[cell] = cells[cell].manifold_id;

              // set entity types
              level.reference_cell[cell] = connectivity.entity_types(dim)[cell];

              // loop over faces
              for (unsigned int i = crs.ptr[cell], j = 0; i < crs.ptr[cell + 1];
                   ++i, ++j)
                {
                  // set neighbor if not at boundary
                  if (nei.col[i] != static_cast<unsigned int>(-1))
                    level.neighbors[cell * GeometryInfo<dim>::faces_per_cell +
                                    j] = {0, nei.col[i]};

                  // set face indices
                  cells_0.cells[cell * GeometryInfo<dim>::faces_per_cell + j] =
                    crs.col[i];

                  // set face orientation if needed
                  if (orientation_needed)
                    {
                      level.face_orientations
                        [cell * GeometryInfo<dim>::faces_per_cell + j] =
                        connectivity.entity_orientations(dim - 1)[i];
                    }
                }
            }
        }

        // TriaFaces: boundary id of boundary faces
        if (dim > 1)
          {
            auto &bids_face = dim == 3 ?
                                tria.faces->quads.boundary_or_material_id :
                                tria.faces->lines.boundary_or_material_id;

            // count number of cells a face is belonging to
            std::vector<unsigned int> count(bids_face.size(), 0);

            // get connectivity between cells/faces
            const auto &crs = connectivity.entity_to_entities(dim, dim - 1);

            // count how many cells are adjacent to the same face
            for (unsigned int cell = 0; cell < cells.size(); ++cell)
              for (unsigned int i = crs.ptr[cell]; i < crs.ptr[cell + 1]; ++i)
                count[crs.col[i]]++;

            // loop over all faces
            for (unsigned int face = 0; face < count.size(); ++face)
              {
                if (count[face] != 1) // inner face
                  continue;

                // boundary faces ...
                bids_face[face].boundary_id = 0;

                if (dim != 3)
                  continue;

                // ... and the lines of quads in 3D
                const auto &crs = connectivity.entity_to_entities(2, 1);
                for (unsigned int i = crs.ptr[face]; i < crs.ptr[face + 1]; ++i)
                  tria.faces->lines.boundary_or_material_id[crs.col[i]]
                    .boundary_id = 0;
              }
          }
        else // 1D
          {
            static const unsigned int t_tba   = static_cast<unsigned int>(-1);
            static const unsigned int t_inner = static_cast<unsigned int>(-2);

            std::vector<unsigned int> type(vertices.size(), t_tba);

            const auto &crs = connectivity.entity_to_entities(1, 0);

            for (unsigned int cell = 0; cell < cells.size(); ++cell)
              for (unsigned int i = crs.ptr[cell], j = 0; i < crs.ptr[cell + 1];
                   ++i, ++j)
                if (type[crs.col[i]] != t_inner)
                  type[crs.col[i]] = type[crs.col[i]] == t_tba ? j : t_inner;

            for (unsigned int face = 0; face < type.size(); ++face)
              {
                // note: we also treat manifolds here!?
                (*tria.vertex_to_manifold_id_map_1d)[face] =
                  numbers::flat_manifold_id;
                if (type[face] != t_inner && type[face] != t_tba)
                  (*tria.vertex_to_boundary_id_map_1d)[face] = type[face];
              }
          }

        // SubCellData: line
        if (dim >= 2)
          process_subcelldata(connectivity.entity_to_entities(1, 0),
                              tria.faces->lines,
                              subcelldata.boundary_lines,
                              vertices);

        // SubCellData: quad
        if (dim == 3)
          process_subcelldata(connectivity.entity_to_entities(2, 0),
                              tria.faces->quads,
                              subcelldata.boundary_quads,
                              vertices);
      }


      template <int structdim, int spacedim, typename T>
      static void
      process_subcelldata(
        const CRS<T> &                          crs,
        TriaObjects &                           obj,
        const std::vector<CellData<structdim>> &boundary_objects_in,
        const std::vector<Point<spacedim>> &    vertex_locations)
      {
        AssertDimension(obj.structdim, structdim);

        if (boundary_objects_in.size() == 0)
          return; // empty subcelldata -> nothing to do

        // pre-sort subcelldata
        auto boundary_objects = boundary_objects_in;

        // ... sort vertices
        for (auto &boundary_object : boundary_objects)
          std::sort(boundary_object.vertices.begin(),
                    boundary_object.vertices.end());

        // ... sort cells
        std::sort(boundary_objects.begin(),
                  boundary_objects.end(),
                  [](const auto &a, const auto &b) {
                    return a.vertices < b.vertices;
                  });

        unsigned int counter = 0;

        std::vector<unsigned int> key;
        key.reserve(GeometryInfo<structdim>::vertices_per_cell);

        for (unsigned int o = 0; o < obj.n_objects(); ++o)
          {
            auto &boundary_id = obj.boundary_or_material_id[o].boundary_id;
            auto &manifold_id = obj.manifold_id[o];

            // assert that object has not been visited yet and its value
            // has not been modified yet
            AssertThrow(boundary_id == 0 ||
                          boundary_id == numbers::internal_face_boundary_id,
                        ExcNotImplemented());
            AssertThrow(manifold_id == numbers::flat_manifold_id,
                        ExcNotImplemented());

            // create key
            key.assign(crs.col.data() + crs.ptr[o],
                       crs.col.data() + crs.ptr[o + 1]);
            std::sort(key.begin(), key.end());

            // is subcelldata provided? -> binary search
            const auto subcell_object =
              std::lower_bound(boundary_objects.begin(),
                               boundary_objects.end(),
                               key,
                               [&](const auto &cell, const auto &key) {
                                 return cell.vertices < key;
                               });

            // no subcelldata provided for this object
            if (subcell_object == boundary_objects.end() ||
                subcell_object->vertices != key)
              continue;

            counter++;

            // set manifold id
            manifold_id = subcell_object->manifold_id;

            // set boundary id
            if (subcell_object->boundary_id !=
                numbers::internal_face_boundary_id)
              {
                (void)vertex_locations;
                AssertThrow(
                  boundary_id != numbers::internal_face_boundary_id,
                  ExcMessage(
                    "The input arguments for creating a triangulation "
                    "specified a boundary id for an internal face. This "
                    "is not allowed."
                    "\n\n"
                    "The object in question has vertex indices " +
                    [subcell_object]() {
                      std::string s;
                      for (const auto v : subcell_object->vertices)
                        s += std::to_string(v) + ',';
                      return s;
                    }() +
                    " which are located at positions " +
                    [vertex_locations, subcell_object]() {
                      std::ostringstream s;
                      for (const auto v : subcell_object->vertices)
                        s << '(' << vertex_locations[v] << ')';
                      return s.str();
                    }() +
                    "."));
                boundary_id = subcell_object->boundary_id;
              }
          }

        // make sure that all subcelldata entries have been processed
        // TODO: this is not guaranteed, why?
        // AssertDimension(counter, boundary_objects_in.size());
      }



      static void
      reserve_space_(TriaFaces &        faces,
                     const unsigned     structdim,
                     const unsigned int size)
      {
        const unsigned int dim = faces.dim;

        const unsigned int max_faces_per_cell = 2 * structdim;

        if (dim == 3 && structdim == 2)
          {
            // quad entity types
            faces.quad_reference_cell.assign(size,
                                             dealii::ReferenceCells::Invalid);

            // quad line orientations
            faces.quads_line_orientations.assign(size * max_faces_per_cell, -1);
          }
      }



      static void
      reserve_space_(TriaLevel &        level,
                     const unsigned int spacedim,
                     const unsigned int size,
                     const bool         orientation_needed)
      {
        const unsigned int dim = level.dim;

        const unsigned int max_faces_per_cell = 2 * dim;

        level.active_cell_indices.assign(size, -1);
        level.subdomain_ids.assign(size, 0);
        level.level_subdomain_ids.assign(size, 0);

        level.refine_flags.assign(size, 0u);
        level.coarsen_flags.assign(size, false);

        level.parents.assign((size + 1) / 2, -1);

        if (dim < spacedim)
          level.direction_flags.assign(size, true);

        level.neighbors.assign(size * max_faces_per_cell, {-1, -1});

        level.reference_cell.assign(size, dealii::ReferenceCells::Invalid);

        if (orientation_needed)
          level.face_orientations.assign(size * max_faces_per_cell, -1);

        level.global_active_cell_indices.assign(size,
                                                numbers::invalid_dof_index);
        level.global_level_cell_indices.assign(size,
                                               numbers::invalid_dof_index);
      }



      static void
      reserve_space_(TriaObjects &obj, const unsigned int size)
      {
        const unsigned int structdim = obj.structdim;

        const unsigned int max_children_per_cell = 1 << structdim;
        const unsigned int max_faces_per_cell    = 2 * structdim;

        obj.used.assign(size, true);
        obj.boundary_or_material_id.assign(
          size,
          internal::TriangulationImplementation::TriaObjects::
            BoundaryOrMaterialId());
        obj.manifold_id.assign(size, -1);
        obj.user_flags.assign(size, false);
        obj.user_data.resize(size);

        if (structdim > 1) // TODO: why?
          obj.refinement_cases.assign(size, 0);

        obj.children.assign(max_children_per_cell / 2 * size, -1);

        obj.cells.assign(max_faces_per_cell * size, -1);

        if (structdim <= 2)
          {
            obj.next_free_single               = size - 1;
            obj.next_free_pair                 = 0;
            obj.reverse_order_next_free_single = true;
          }
        else
          {
            obj.next_free_single = obj.next_free_pair = 0;
          }
      }


      /**
       * Actually delete a cell, or rather all
       * its children, which is the main step for
       * the coarsening process.  This is the
       * dimension dependent part of @p
       * execute_coarsening. The second argument
       * is a vector which gives for each line
       * index the number of cells containing
       * this line. This information is needed to
       * decide whether a refined line may be
       * coarsened or not in 3D. In 1D and 2D
       * this argument is not needed and thus
       * ignored. The same applies for the last
       * argument and quads instead of lines.
       */
      template <int spacedim>
      static void
      delete_children(Triangulation<1, spacedim> &triangulation,
                      typename Triangulation<1, spacedim>::cell_iterator &cell,
                      std::vector<unsigned int> &,
                      std::vector<unsigned int> &)
      {
        const unsigned int dim = 1;

        // first we need to reset the
        // neighbor pointers of the
        // neighbors of this cell's
        // children to this cell. This is
        // different for one dimension,
        // since there neighbors can have a
        // refinement level differing from
        // that of this cell's children by
        // more than one level.

        Assert(!cell->child(0)->has_children() &&
                 !cell->child(1)->has_children(),
               ExcInternalError());

        // first do it for the cells to the
        // left
        if (cell->neighbor(0).state() == IteratorState::valid)
          if (cell->neighbor(0)->has_children())
            {
              typename Triangulation<dim, spacedim>::cell_iterator neighbor =
                cell->neighbor(0);
              Assert(neighbor->level() == cell->level(), ExcInternalError());

              // right child
              neighbor = neighbor->child(1);
              while (true)
                {
                  Assert(neighbor->neighbor(1) == cell->child(0),
                         ExcInternalError());
                  neighbor->set_neighbor(1, cell);

                  // move on to further
                  // children on the
                  // boundary between this
                  // cell and its neighbor
                  if (neighbor->has_children())
                    neighbor = neighbor->child(1);
                  else
                    break;
                }
            }

        // now do it for the cells to the
        // left
        if (cell->neighbor(1).state() == IteratorState::valid)
          if (cell->neighbor(1)->has_children())
            {
              typename Triangulation<dim, spacedim>::cell_iterator neighbor =
                cell->neighbor(1);
              Assert(neighbor->level() == cell->level(), ExcInternalError());

              // left child
              neighbor = neighbor->child(0);
              while (true)
                {
                  Assert(neighbor->neighbor(0) == cell->child(1),
                         ExcInternalError());
                  neighbor->set_neighbor(0, cell);

                  // move on to further
                  // children on the
                  // boundary between this
                  // cell and its neighbor
                  if (neighbor->has_children())
                    neighbor = neighbor->child(0);
                  else
                    break;
                }
            }


        // delete the vertex which will not
        // be needed anymore. This vertex
        // is the second of the first child
        triangulation.vertices_used[cell->child(0)->vertex_index(1)] = false;

        // invalidate children.  clear user
        // pointers, to avoid that they may
        // appear at unwanted places later
        // on...
        for (unsigned int child = 0; child < cell->n_children(); ++child)
          {
            cell->child(child)->clear_user_data();
            cell->child(child)->clear_user_flag();
            cell->child(child)->clear_used_flag();
          }


        // delete pointer to children
        cell->clear_children();
        cell->clear_user_flag();
      }



      template <int spacedim>
      static void
      delete_children(Triangulation<2, spacedim> &triangulation,
                      typename Triangulation<2, spacedim>::cell_iterator &cell,
                      std::vector<unsigned int> &line_cell_count,
                      std::vector<unsigned int> &)
      {
        const unsigned int        dim      = 2;
        const RefinementCase<dim> ref_case = cell->refinement_case();

        Assert(line_cell_count.size() == triangulation.n_raw_lines(),
               ExcInternalError());

        // vectors to hold all lines which
        // may be deleted
        std::vector<typename Triangulation<dim, spacedim>::line_iterator>
          lines_to_delete(0);

        lines_to_delete.reserve(4 * 2 + 4);

        // now we decrease the counters for
        // lines contained in the child
        // cells
        for (unsigned int c = 0; c < cell->n_children(); ++c)
          {
            typename Triangulation<dim, spacedim>::cell_iterator child =
              cell->child(c);
            for (unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
              --line_cell_count[child->line_index(l)];
          }


        // delete the vertex which will not
        // be needed anymore. This vertex
        // is the second of the second line
        // of the first child, if the cell
        // is refined with cut_xy, else there
        // is no inner vertex.
        // additionally delete unneeded inner
        // lines
        if (ref_case == RefinementCase<dim>::cut_xy)
          {
            triangulation
              .vertices_used[cell->child(0)->line(1)->vertex_index(1)] = false;

            lines_to_delete.push_back(cell->child(0)->line(1));
            lines_to_delete.push_back(cell->child(0)->line(3));
            lines_to_delete.push_back(cell->child(3)->line(0));
            lines_to_delete.push_back(cell->child(3)->line(2));
          }
        else
          {
            unsigned int inner_face_no =
              ref_case == RefinementCase<dim>::cut_x ? 1 : 3;

            // the inner line will not be
            // used any more
            lines_to_delete.push_back(cell->child(0)->line(inner_face_no));
          }

        // invalidate children
        for (unsigned int child = 0; child < cell->n_children(); ++child)
          {
            cell->child(child)->clear_user_data();
            cell->child(child)->clear_user_flag();
            cell->child(child)->clear_used_flag();
          }


        // delete pointer to children
        cell->clear_children();
        cell->clear_refinement_case();
        cell->clear_user_flag();

        // look at the refinement of outer
        // lines. if nobody needs those
        // anymore we can add them to the
        // list of lines to be deleted.
        for (unsigned int line_no = 0;
             line_no < GeometryInfo<dim>::lines_per_cell;
             ++line_no)
          {
            typename Triangulation<dim, spacedim>::line_iterator line =
              cell->line(line_no);

            if (line->has_children())
              {
                // if one of the cell counters is
                // zero, the other has to be as well

                Assert((line_cell_count[line->child_index(0)] == 0 &&
                        line_cell_count[line->child_index(1)] == 0) ||
                         (line_cell_count[line->child_index(0)] > 0 &&
                          line_cell_count[line->child_index(1)] > 0),
                       ExcInternalError());

                if (line_cell_count[line->child_index(0)] == 0)
                  {
                    for (unsigned int c = 0; c < 2; ++c)
                      Assert(!line->child(c)->has_children(),
                             ExcInternalError());

                    // we may delete the line's
                    // children and the middle vertex
                    // as no cell references them
                    // anymore
                    triangulation
                      .vertices_used[line->child(0)->vertex_index(1)] = false;

                    lines_to_delete.push_back(line->child(0));
                    lines_to_delete.push_back(line->child(1));

                    line->clear_children();
                  }
              }
          }

        // finally, delete unneeded lines

        // clear user pointers, to avoid that
        // they may appear at unwanted places
        // later on...
        // same for user flags, then finally
        // delete the lines
        typename std::vector<
          typename Triangulation<dim, spacedim>::line_iterator>::iterator
          line    = lines_to_delete.begin(),
          endline = lines_to_delete.end();
        for (; line != endline; ++line)
          {
            (*line)->clear_user_data();
            (*line)->clear_user_flag();
            (*line)->clear_used_flag();
          }
      }



      template <int spacedim>
      static void
      delete_children(Triangulation<3, spacedim> &triangulation,
                      typename Triangulation<3, spacedim>::cell_iterator &cell,
                      std::vector<unsigned int> &line_cell_count,
                      std::vector<unsigned int> &quad_cell_count)
      {
        const unsigned int dim = 3;

        Assert(line_cell_count.size() == triangulation.n_raw_lines(),
               ExcInternalError());
        Assert(quad_cell_count.size() == triangulation.n_raw_quads(),
               ExcInternalError());

        // first of all, we store the RefineCase of
        // this cell
        const RefinementCase<dim> ref_case = cell->refinement_case();
        // vectors to hold all lines and quads which
        // may be deleted
        std::vector<typename Triangulation<dim, spacedim>::line_iterator>
          lines_to_delete(0);
        std::vector<typename Triangulation<dim, spacedim>::quad_iterator>
          quads_to_delete(0);

        lines_to_delete.reserve(12 * 2 + 6 * 4 + 6);
        quads_to_delete.reserve(6 * 4 + 12);

        // now we decrease the counters for lines and
        // quads contained in the child cells
        for (unsigned int c = 0; c < cell->n_children(); ++c)
          {
            typename Triangulation<dim, spacedim>::cell_iterator child =
              cell->child(c);
            for (unsigned int l = 0; l < GeometryInfo<dim>::lines_per_cell; ++l)
              --line_cell_count[child->line_index(l)];
            for (auto f : GeometryInfo<dim>::face_indices())
              --quad_cell_count[child->quad_index(f)];
          }

        //-------------------------------------
        // delete interior quads and lines and the
        // interior vertex, depending on the
        // refinement case of the cell
        //
        // for append quads and lines: only append
        // them to the list of objects to be deleted

        switch (ref_case)
          {
            case RefinementCase<dim>::cut_x:
              quads_to_delete.push_back(cell->child(0)->face(1));
              break;
            case RefinementCase<dim>::cut_y:
              quads_to_delete.push_back(cell->child(0)->face(3));
              break;
            case RefinementCase<dim>::cut_z:
              quads_to_delete.push_back(cell->child(0)->face(5));
              break;
            case RefinementCase<dim>::cut_xy:
              quads_to_delete.push_back(cell->child(0)->face(1));
              quads_to_delete.push_back(cell->child(0)->face(3));
              quads_to_delete.push_back(cell->child(3)->face(0));
              quads_to_delete.push_back(cell->child(3)->face(2));

              lines_to_delete.push_back(cell->child(0)->line(11));
              break;
            case RefinementCase<dim>::cut_xz:
              quads_to_delete.push_back(cell->child(0)->face(1));
              quads_to_delete.push_back(cell->child(0)->face(5));
              quads_to_delete.push_back(cell->child(3)->face(0));
              quads_to_delete.push_back(cell->child(3)->face(4));

              lines_to_delete.push_back(cell->child(0)->line(5));
              break;
            case RefinementCase<dim>::cut_yz:
              quads_to_delete.push_back(cell->child(0)->face(3));
              quads_to_delete.push_back(cell->child(0)->face(5));
              quads_to_delete.push_back(cell->child(3)->face(2));
              quads_to_delete.push_back(cell->child(3)->face(4));

              lines_to_delete.push_back(cell->child(0)->line(7));
              break;
            case RefinementCase<dim>::cut_xyz:
              quads_to_delete.push_back(cell->child(0)->face(1));
              quads_to_delete.push_back(cell->child(2)->face(1));
              quads_to_delete.push_back(cell->child(4)->face(1));
              quads_to_delete.push_back(cell->child(6)->face(1));

              quads_to_delete.push_back(cell->child(0)->face(3));
              quads_to_delete.push_back(cell->child(1)->face(3));
              quads_to_delete.push_back(cell->child(4)->face(3));
              quads_to_delete.push_back(cell->child(5)->face(3));

              quads_to_delete.push_back(cell->child(0)->face(5));
              quads_to_delete.push_back(cell->child(1)->face(5));
              quads_to_delete.push_back(cell->child(2)->face(5));
              quads_to_delete.push_back(cell->child(3)->face(5));

              lines_to_delete.push_back(cell->child(0)->line(5));
              lines_to_delete.push_back(cell->child(0)->line(7));
              lines_to_delete.push_back(cell->child(0)->line(11));
              lines_to_delete.push_back(cell->child(7)->line(0));
              lines_to_delete.push_back(cell->child(7)->line(2));
              lines_to_delete.push_back(cell->child(7)->line(8));
              // delete the vertex which will not
              // be needed anymore. This vertex
              // is the vertex at the heart of
              // this cell, which is the sixth of
              // the first child
              triangulation.vertices_used[cell->child(0)->vertex_index(7)] =
                false;
              break;
            default:
              // only remaining case is
              // no_refinement, thus an error
              Assert(false, ExcInternalError());
              break;
          }


        // invalidate children
        for (unsigned int child = 0; child < cell->n_children(); ++child)
          {
            cell->child(child)->clear_user_data();
            cell->child(child)->clear_user_flag();

            for (auto f : GeometryInfo<dim>::face_indices())
              {
                // set flags denoting deviations from
                // standard orientation of faces back
                // to initialization values
                cell->child(child)->set_face_orientation(f, true);
                cell->child(child)->set_face_flip(f, false);
                cell->child(child)->set_face_rotation(f, false);
              }

            cell->child(child)->clear_used_flag();
          }


        // delete pointer to children
        cell->clear_children();
        cell->clear_refinement_case();
        cell->clear_user_flag();

        // so far we only looked at inner quads,
        // lines and vertices. Now we have to
        // consider outer ones as well. here, we have
        // to check, whether there are other cells
        // still needing these objects. otherwise we
        // can delete them. first for quads (and
        // their inner lines).

        for (const unsigned int quad_no : GeometryInfo<dim>::face_indices())
          {
            typename Triangulation<dim, spacedim>::quad_iterator quad =
              cell->face(quad_no);

            Assert(
              (GeometryInfo<dim>::face_refinement_case(ref_case, quad_no) &&
               quad->has_children()) ||
                GeometryInfo<dim>::face_refinement_case(ref_case, quad_no) ==
                  RefinementCase<dim - 1>::no_refinement,
              ExcInternalError());

            switch (quad->refinement_case())
              {
                case RefinementCase<dim - 1>::no_refinement:
                  // nothing to do as the quad
                  // is not refined
                  break;
                case RefinementCase<dim - 1>::cut_x:
                case RefinementCase<dim - 1>::cut_y:
                  {
                    // if one of the cell counters is
                    // zero, the other has to be as
                    // well
                    Assert((quad_cell_count[quad->child_index(0)] == 0 &&
                            quad_cell_count[quad->child_index(1)] == 0) ||
                             (quad_cell_count[quad->child_index(0)] > 0 &&
                              quad_cell_count[quad->child_index(1)] > 0),
                           ExcInternalError());
                    // it might be, that the quad is
                    // refined twice anisotropically,
                    // first check, whether we may
                    // delete possible grand_children
                    unsigned int deleted_grandchildren       = 0;
                    unsigned int number_of_child_refinements = 0;

                    for (unsigned int c = 0; c < 2; ++c)
                      if (quad->child(c)->has_children())
                        {
                          ++number_of_child_refinements;
                          // if one of the cell counters is
                          // zero, the other has to be as
                          // well
                          Assert(
                            (quad_cell_count[quad->child(c)->child_index(0)] ==
                               0 &&
                             quad_cell_count[quad->child(c)->child_index(1)] ==
                               0) ||
                              (quad_cell_count[quad->child(c)->child_index(0)] >
                                 0 &&
                               quad_cell_count[quad->child(c)->child_index(1)] >
                                 0),
                            ExcInternalError());
                          if (quad_cell_count[quad->child(c)->child_index(0)] ==
                              0)
                            {
                              // Assert, that the two
                              // anisotropic
                              // refinements add up to
                              // isotropic refinement
                              Assert(quad->refinement_case() +
                                         quad->child(c)->refinement_case() ==
                                       RefinementCase<dim>::cut_xy,
                                     ExcInternalError());
                              // we may delete the
                              // quad's children and
                              // the inner line as no
                              // cell references them
                              // anymore
                              quads_to_delete.push_back(
                                quad->child(c)->child(0));
                              quads_to_delete.push_back(
                                quad->child(c)->child(1));
                              if (quad->child(c)->refinement_case() ==
                                  RefinementCase<2>::cut_x)
                                lines_to_delete.push_back(
                                  quad->child(c)->child(0)->line(1));
                              else
                                lines_to_delete.push_back(
                                  quad->child(c)->child(0)->line(3));
                              quad->child(c)->clear_children();
                              quad->child(c)->clear_refinement_case();
                              ++deleted_grandchildren;
                            }
                        }
                    // if no grandchildren are left, we
                    // may as well delete the
                    // refinement of the inner line
                    // between our children and the
                    // corresponding vertex
                    if (number_of_child_refinements > 0 &&
                        deleted_grandchildren == number_of_child_refinements)
                      {
                        typename Triangulation<dim, spacedim>::line_iterator
                          middle_line;
                        if (quad->refinement_case() == RefinementCase<2>::cut_x)
                          middle_line = quad->child(0)->line(1);
                        else
                          middle_line = quad->child(0)->line(3);

                        lines_to_delete.push_back(middle_line->child(0));
                        lines_to_delete.push_back(middle_line->child(1));
                        triangulation
                          .vertices_used[middle_vertex_index<dim, spacedim>(
                            middle_line)] = false;
                        middle_line->clear_children();
                      }

                    // now consider the direct children
                    // of the given quad
                    if (quad_cell_count[quad->child_index(0)] == 0)
                      {
                        // we may delete the quad's
                        // children and the inner line
                        // as no cell references them
                        // anymore
                        quads_to_delete.push_back(quad->child(0));
                        quads_to_delete.push_back(quad->child(1));
                        if (quad->refinement_case() == RefinementCase<2>::cut_x)
                          lines_to_delete.push_back(quad->child(0)->line(1));
                        else
                          lines_to_delete.push_back(quad->child(0)->line(3));

                        // if the counters just dropped
                        // to zero, otherwise the
                        // children would have been
                        // deleted earlier, then this
                        // cell's children must have
                        // contained the anisotropic
                        // quad children. thus, if
                        // those have again anisotropic
                        // children, which are in
                        // effect isotropic children of
                        // the original quad, those are
                        // still needed by a
                        // neighboring cell and we
                        // cannot delete them. instead,
                        // we have to reset this quad's
                        // refine case to isotropic and
                        // set the children
                        // accordingly.
                        if (quad->child(0)->has_children())
                          if (quad->refinement_case() ==
                              RefinementCase<2>::cut_x)
                            {
                              // now evereything is
                              // quite complicated. we
                              // have the children
                              // numbered according to
                              //
                              // *---*---*
                              // |n+1|m+1|
                              // *---*---*
                              // | n | m |
                              // *---*---*
                              //
                              // from the original
                              // anisotropic
                              // refinement. we have to
                              // reorder them as
                              //
                              // *---*---*
                              // | m |m+1|
                              // *---*---*
                              // | n |n+1|
                              // *---*---*
                              //
                              // for isotropic refinement.
                              //
                              // this is a bit ugly, of
                              // course: loop over all
                              // cells on all levels
                              // and look for faces n+1
                              // (switch_1) and m
                              // (switch_2).
                              const typename Triangulation<dim, spacedim>::
                                quad_iterator switch_1 =
                                                quad->child(0)->child(1),
                                              switch_2 =
                                                quad->child(1)->child(0);

                              Assert(!switch_1->has_children(),
                                     ExcInternalError());
                              Assert(!switch_2->has_children(),
                                     ExcInternalError());

                              const int switch_1_index = switch_1->index();
                              const int switch_2_index = switch_2->index();
                              for (unsigned int l = 0;
                                   l < triangulation.levels.size();
                                   ++l)
                                for (unsigned int h = 0;
                                     h <
                                     triangulation.levels[l]->cells.n_objects();
                                     ++h)
                                  for (const unsigned int q :
                                       GeometryInfo<dim>::face_indices())
                                    {
                                      const int index =
                                        triangulation.levels[l]
                                          ->cells.get_bounding_object_indices(
                                            h)[q];
                                      if (index == switch_1_index)
                                        triangulation.levels[l]
                                          ->cells.get_bounding_object_indices(
                                            h)[q] = switch_2_index;
                                      else if (index == switch_2_index)
                                        triangulation.levels[l]
                                          ->cells.get_bounding_object_indices(
                                            h)[q] = switch_1_index;
                                    }
                              // now we have to copy
                              // all information of the
                              // two quads
                              const int switch_1_lines[4] = {
                                static_cast<signed int>(
                                  switch_1->line_index(0)),
                                static_cast<signed int>(
                                  switch_1->line_index(1)),
                                static_cast<signed int>(
                                  switch_1->line_index(2)),
                                static_cast<signed int>(
                                  switch_1->line_index(3))};
                              const bool switch_1_line_orientations[4] = {
                                switch_1->line_orientation(0),
                                switch_1->line_orientation(1),
                                switch_1->line_orientation(2),
                                switch_1->line_orientation(3)};
                              const types::boundary_id switch_1_boundary_id =
                                switch_1->boundary_id();
                              const unsigned int switch_1_user_index =
                                switch_1->user_index();
                              const bool switch_1_user_flag =
                                switch_1->user_flag_set();

                              switch_1->set_bounding_object_indices(
                                {switch_2->line_index(0),
                                 switch_2->line_index(1),
                                 switch_2->line_index(2),
                                 switch_2->line_index(3)});
                              switch_1->set_line_orientation(
                                0, switch_2->line_orientation(0));
                              switch_1->set_line_orientation(
                                1, switch_2->line_orientation(1));
                              switch_1->set_line_orientation(
                                2, switch_2->line_orientation(2));
                              switch_1->set_line_orientation(
                                3, switch_2->line_orientation(3));
                              switch_1->set_boundary_id_internal(
                                switch_2->boundary_id());
                              switch_1->set_manifold_id(
                                switch_2->manifold_id());
                              switch_1->set_user_index(switch_2->user_index());
                              if (switch_2->user_flag_set())
                                switch_1->set_user_flag();
                              else
                                switch_1->clear_user_flag();

                              switch_2->set_bounding_object_indices(
                                {switch_1_lines[0],
                                 switch_1_lines[1],
                                 switch_1_lines[2],
                                 switch_1_lines[3]});
                              switch_2->set_line_orientation(
                                0, switch_1_line_orientations[0]);
                              switch_2->set_line_orientation(
                                1, switch_1_line_orientations[1]);
                              switch_2->set_line_orientation(
                                2, switch_1_line_orientations[2]);
                              switch_2->set_line_orientation(
                                3, switch_1_line_orientations[3]);
                              switch_2->set_boundary_id_internal(
                                switch_1_boundary_id);
                              switch_2->set_manifold_id(
                                switch_1->manifold_id());
                              switch_2->set_user_index(switch_1_user_index);
                              if (switch_1_user_flag)
                                switch_2->set_user_flag();
                              else
                                switch_2->clear_user_flag();

                              const unsigned int child_0 =
                                quad->child(0)->child_index(0);
                              const unsigned int child_2 =
                                quad->child(1)->child_index(0);
                              quad->clear_children();
                              quad->clear_refinement_case();
                              quad->set_refinement_case(
                                RefinementCase<2>::cut_xy);
                              quad->set_children(0, child_0);
                              quad->set_children(2, child_2);
                              std::swap(quad_cell_count[child_0 + 1],
                                        quad_cell_count[child_2]);
                            }
                          else
                            {
                              // the face was refined
                              // with cut_y, thus the
                              // children are already
                              // in correct order. we
                              // only have to set them
                              // correctly, deleting
                              // the indirection of two
                              // anisotropic refinement
                              // and going directly
                              // from the quad to
                              // isotropic children
                              const unsigned int child_0 =
                                quad->child(0)->child_index(0);
                              const unsigned int child_2 =
                                quad->child(1)->child_index(0);
                              quad->clear_children();
                              quad->clear_refinement_case();
                              quad->set_refinement_case(
                                RefinementCase<2>::cut_xy);
                              quad->set_children(0, child_0);
                              quad->set_children(2, child_2);
                            }
                        else
                          {
                            quad->clear_children();
                            quad->clear_refinement_case();
                          }
                      }
                    break;
                  }
                case RefinementCase<dim - 1>::cut_xy:
                  {
                    // if one of the cell counters is
                    // zero, the others have to be as
                    // well

                    Assert((quad_cell_count[quad->child_index(0)] == 0 &&
                            quad_cell_count[quad->child_index(1)] == 0 &&
                            quad_cell_count[quad->child_index(2)] == 0 &&
                            quad_cell_count[quad->child_index(3)] == 0) ||
                             (quad_cell_count[quad->child_index(0)] > 0 &&
                              quad_cell_count[quad->child_index(1)] > 0 &&
                              quad_cell_count[quad->child_index(2)] > 0 &&
                              quad_cell_count[quad->child_index(3)] > 0),
                           ExcInternalError());

                    if (quad_cell_count[quad->child_index(0)] == 0)
                      {
                        // we may delete the quad's
                        // children, the inner lines
                        // and the middle vertex as no
                        // cell references them anymore
                        lines_to_delete.push_back(quad->child(0)->line(1));
                        lines_to_delete.push_back(quad->child(3)->line(0));
                        lines_to_delete.push_back(quad->child(0)->line(3));
                        lines_to_delete.push_back(quad->child(3)->line(2));

                        for (unsigned int child = 0; child < quad->n_children();
                             ++child)
                          quads_to_delete.push_back(quad->child(child));

                        triangulation
                          .vertices_used[quad->child(0)->vertex_index(3)] =
                          false;

                        quad->clear_children();
                        quad->clear_refinement_case();
                      }
                  }
                  break;

                default:
                  Assert(false, ExcInternalError());
                  break;
              }
          }

        // now we repeat a similar procedure
        // for the outer lines of this cell.

        // if in debug mode: check that each
        // of the lines for which we consider
        // deleting the children in fact has
        // children (the bits/coarsening_3d
        // test tripped over this initially)
        for (unsigned int line_no = 0;
             line_no < GeometryInfo<dim>::lines_per_cell;
             ++line_no)
          {
            typename Triangulation<dim, spacedim>::line_iterator line =
              cell->line(line_no);

            Assert(
              (GeometryInfo<dim>::line_refinement_case(ref_case, line_no) &&
               line->has_children()) ||
                GeometryInfo<dim>::line_refinement_case(ref_case, line_no) ==
                  RefinementCase<1>::no_refinement,
              ExcInternalError());

            if (line->has_children())
              {
                // if one of the cell counters is
                // zero, the other has to be as well

                Assert((line_cell_count[line->child_index(0)] == 0 &&
                        line_cell_count[line->child_index(1)] == 0) ||
                         (line_cell_count[line->child_index(0)] > 0 &&
                          line_cell_count[line->child_index(1)] > 0),
                       ExcInternalError());

                if (line_cell_count[line->child_index(0)] == 0)
                  {
                    for (unsigned int c = 0; c < 2; ++c)
                      Assert(!line->child(c)->has_children(),
                             ExcInternalError());

                    // we may delete the line's
                    // children and the middle vertex
                    // as no cell references them
                    // anymore
                    triangulation
                      .vertices_used[line->child(0)->vertex_index(1)] = false;

                    lines_to_delete.push_back(line->child(0));
                    lines_to_delete.push_back(line->child(1));

                    line->clear_children();
                  }
              }
          }

        // finally, delete unneeded quads and lines

        // clear user pointers, to avoid that
        // they may appear at unwanted places
        // later on...
        // same for user flags, then finally
        // delete the quads and lines
        typename std::vector<
          typename Triangulation<dim, spacedim>::line_iterator>::iterator
          line    = lines_to_delete.begin(),
          endline = lines_to_delete.end();
        for (; line != endline; ++line)
          {
            (*line)->clear_user_data();
            (*line)->clear_user_flag();
            (*line)->clear_used_flag();
          }

        typename std::vector<
          typename Triangulation<dim, spacedim>::quad_iterator>::iterator
          quad    = quads_to_delete.begin(),
          endquad = quads_to_delete.end();
        for (; quad != endquad; ++quad)
          {
            (*quad)->clear_user_data();
            (*quad)->clear_children();
            (*quad)->clear_refinement_case();
            (*quad)->clear_user_flag();
            (*quad)->clear_used_flag();
          }
      }


      /**
       * Create the children of a 2d
       * cell. The arguments indicate
       * the next free spots in the
       * vertices, lines, and cells
       * arrays.
       *
       * The faces of the cell have to
       * be refined already, whereas
       * the inner lines in 2D will be
       * created in this
       * function. Therefore iterator
       * pointers into the vectors of
       * lines, quads and cells have to
       * be passed, which point at (or
       * "before") the reserved space.
       */
      template <int spacedim>
      static void
      create_children(
        Triangulation<2, spacedim> &triangulation,
        unsigned int &              next_unused_vertex,
        typename Triangulation<2, spacedim>::raw_line_iterator
          &next_unused_line,
        typename Triangulation<2, spacedim>::raw_cell_iterator
          &next_unused_cell,
        const typename Triangulation<2, spacedim>::cell_iterator &cell)
      {
        const unsigned int dim = 2;
        // clear refinement flag
        const RefinementCase<dim> ref_case = cell->refine_flag_set();
        cell->clear_refine_flag();

        /* For the refinement process: since we go the levels up from the
           lowest, there are (unlike above) only two possibilities: a neighbor
           cell is on the same level or one level up (in both cases, it may or
           may not be refined later on, but we don't care here).

           First:
           Set up an array of the 3x3 vertices, which are distributed on the
           cell (the array consists of indices into the @p{vertices} std::vector

           2--7--3
           |  |  |
           4--8--5
           |  |  |
           0--6--1

           note: in case of cut_x or cut_y not all these vertices are needed for
           the new cells

           Second:
           Set up an array of the new lines (the array consists of iterator
           pointers into the lines arrays)

           .-6-.-7-.         The directions are:  .->-.->-.
           1   9   3                              ^   ^   ^
           .-10.11-.                             .->-.->-.
           0   8   2                              ^   ^   ^
           .-4-.-5-.                              .->-.->-.

           cut_x:
           .-4-.-5-.
           |   |   |
           0   6   1
           |   |   |
           .-2-.-3-.

           cut_y:
           .---5---.
           1       3
           .---6---.
           0       2
           .---4---.


           Third:
           Set up an array of neighbors:

           6  7
           .--.--.
           1|  |  |3
           .--.--.
           0|  |  |2
           .--.--.
           4   5

           We need this array for two reasons: first to get the lines which will
           bound the four subcells (if the neighboring cell is refined, these
           lines already exist), and second to update neighborship information.
           Since if a neighbor is not refined, its neighborship record only
           points to the present, unrefined, cell rather than the children we
           are presently creating, we only need the neighborship information
           if the neighbor cells are refined. In all other cases, we store
           the unrefined neighbor address

           We also need for every neighbor (if refined) which number among its
           neighbors the present (unrefined) cell has, since that number is to
           be replaced and because that also is the number of the subline which
           will be the interface between that neighbor and the to be created
           cell. We will store this number (between 0 and 3) in the field
           @p{neighbors_neighbor}.

           It would be sufficient to use the children of the common line to the
           neighbor, if we only wanted to get the new sublines and the new
           vertex, but because we need to update the neighborship information of
           the two refined subcells of the neighbor, we need to search these
           anyway.

           Convention:
           The created children are numbered like this:

           .--.--.
           |2 . 3|
           .--.--.
           |0 | 1|
           .--.--.
        */
        // collect the indices of the eight surrounding vertices
        //   2--7--3
        //   |  |  |
        //   4--8--5
        //   |  |  |
        //   0--6--1
        int new_vertices[9];
        for (unsigned int vertex_no = 0; vertex_no < 4; ++vertex_no)
          new_vertices[vertex_no] = cell->vertex_index(vertex_no);
        for (unsigned int line_no = 0; line_no < 4; ++line_no)
          if (cell->line(line_no)->has_children())
            new_vertices[4 + line_no] =
              cell->line(line_no)->child(0)->vertex_index(1);

        if (ref_case == RefinementCase<dim>::cut_xy)
          {
            // find the next
            // unused vertex and
            // allocate it for
            // the new vertex we
            // need here
            while (triangulation.vertices_used[next_unused_vertex] == true)
              ++next_unused_vertex;
            Assert(next_unused_vertex < triangulation.vertices.size(),
                   ExcMessage(
                     "Internal error: During refinement, the triangulation "
                     "wants to access an element of the 'vertices' array "
                     "but it turns out that the array is not large enough."));
            triangulation.vertices_used[next_unused_vertex] = true;

            new_vertices[8] = next_unused_vertex;

            // determine middle vertex by transfinite interpolation to be
            // consistent with what happens to quads in a
            // Triangulation<3,3> when they are refined
            triangulation.vertices[next_unused_vertex] =
              cell->center(true, true);
          }


        // Now the lines:
        typename Triangulation<dim, spacedim>::raw_line_iterator new_lines[12];
        unsigned int                                             lmin = 8;
        unsigned int                                             lmax = 12;
        if (ref_case != RefinementCase<dim>::cut_xy)
          {
            lmin = 6;
            lmax = 7;
          }

        for (unsigned int l = lmin; l < lmax; ++l)
          {
            while (next_unused_line->used() == true)
              ++next_unused_line;
            new_lines[l] = next_unused_line;
            ++next_unused_line;

            AssertIsNotUsed(new_lines[l]);
          }

        if (ref_case == RefinementCase<dim>::cut_xy)
          {
            //   .-6-.-7-.
            //   1   9   3
            //   .-10.11-.
            //   0   8   2
            //   .-4-.-5-.

            // lines 0-7 already exist, create only the four interior
            // lines 8-11
            unsigned int l = 0;
            for (const unsigned int face_no : GeometryInfo<dim>::face_indices())
              for (unsigned int c = 0; c < 2; ++c, ++l)
                new_lines[l] = cell->line(face_no)->child(c);
            Assert(l == 8, ExcInternalError());

            new_lines[8]->set_bounding_object_indices(
              {new_vertices[6], new_vertices[8]});
            new_lines[9]->set_bounding_object_indices(
              {new_vertices[8], new_vertices[7]});
            new_lines[10]->set_bounding_object_indices(
              {new_vertices[4], new_vertices[8]});
            new_lines[11]->set_bounding_object_indices(
              {new_vertices[8], new_vertices[5]});
          }
        else if (ref_case == RefinementCase<dim>::cut_x)
          {
            //   .-4-.-5-.
            //   |   |   |
            //   0   6   1
            //   |   |   |
            //   .-2-.-3-.
            new_lines[0] = cell->line(0);
            new_lines[1] = cell->line(1);
            new_lines[2] = cell->line(2)->child(0);
            new_lines[3] = cell->line(2)->child(1);
            new_lines[4] = cell->line(3)->child(0);
            new_lines[5] = cell->line(3)->child(1);
            new_lines[6]->set_bounding_object_indices(
              {new_vertices[6], new_vertices[7]});
          }
        else
          {
            Assert(ref_case == RefinementCase<dim>::cut_y, ExcInternalError());
            //   .---5---.
            //   1       3
            //   .---6---.
            //   0       2
            //   .---4---.
            new_lines[0] = cell->line(0)->child(0);
            new_lines[1] = cell->line(0)->child(1);
            new_lines[2] = cell->line(1)->child(0);
            new_lines[3] = cell->line(1)->child(1);
            new_lines[4] = cell->line(2);
            new_lines[5] = cell->line(3);
            new_lines[6]->set_bounding_object_indices(
              {new_vertices[4], new_vertices[5]});
          }

        for (unsigned int l = lmin; l < lmax; ++l)
          {
            new_lines[l]->set_used_flag();
            new_lines[l]->clear_user_flag();
            new_lines[l]->clear_user_data();
            new_lines[l]->clear_children();
            // interior line
            new_lines[l]->set_boundary_id_internal(
              numbers::internal_face_boundary_id);
            new_lines[l]->set_manifold_id(cell->manifold_id());
          }

        // Now add the four (two)
        // new cells!
        typename Triangulation<dim, spacedim>::raw_cell_iterator
          subcells[GeometryInfo<dim>::max_children_per_cell];
        while (next_unused_cell->used() == true)
          ++next_unused_cell;

        const unsigned int n_children = GeometryInfo<dim>::n_children(ref_case);
        for (unsigned int i = 0; i < n_children; ++i)
          {
            AssertIsNotUsed(next_unused_cell);
            subcells[i] = next_unused_cell;
            ++next_unused_cell;
            if (i % 2 == 1 && i < n_children - 1)
              while (next_unused_cell->used() == true)
                ++next_unused_cell;
          }

        if (ref_case == RefinementCase<dim>::cut_xy)
          {
            // children:
            //   .--.--.
            //   |2 . 3|
            //   .--.--.
            //   |0 | 1|
            //   .--.--.
            // lines:
            //   .-6-.-7-.
            //   1   9   3
            //   .-10.11-.
            //   0   8   2
            //   .-4-.-5-.
            subcells[0]->set_bounding_object_indices({new_lines[0]->index(),
                                                      new_lines[8]->index(),
                                                      new_lines[4]->index(),
                                                      new_lines[10]->index()});
            subcells[1]->set_bounding_object_indices({new_lines[8]->index(),
                                                      new_lines[2]->index(),
                                                      new_lines[5]->index(),
                                                      new_lines[11]->index()});
            subcells[2]->set_bounding_object_indices({new_lines[1]->index(),
                                                      new_lines[9]->index(),
                                                      new_lines[10]->index(),
                                                      new_lines[6]->index()});
            subcells[3]->set_bounding_object_indices({new_lines[9]->index(),
                                                      new_lines[3]->index(),
                                                      new_lines[11]->index(),
                                                      new_lines[7]->index()});
          }
        else if (ref_case == RefinementCase<dim>::cut_x)
          {
            // children:
            //   .--.--.
            //   |  .  |
            //   .0 . 1.
            //   |  |  |
            //   .--.--.
            // lines:
            //   .-4-.-5-.
            //   |   |   |
            //   0   6   1
            //   |   |   |
            //   .-2-.-3-.
            subcells[0]->set_bounding_object_indices({new_lines[0]->index(),
                                                      new_lines[6]->index(),
                                                      new_lines[2]->index(),
                                                      new_lines[4]->index()});
            subcells[1]->set_bounding_object_indices({new_lines[6]->index(),
                                                      new_lines[1]->index(),
                                                      new_lines[3]->index(),
                                                      new_lines[5]->index()});
          }
        else
          {
            Assert(ref_case == RefinementCase<dim>::cut_y, ExcInternalError());
            // children:
            //   .-----.
            //   |  1  |
            //   .-----.
            //   |  0  |
            //   .-----.
            // lines:
            //   .---5---.
            //   1       3
            //   .---6---.
            //   0       2
            //   .---4---.
            subcells[0]->set_bounding_object_indices({new_lines[0]->index(),
                                                      new_lines[2]->index(),
                                                      new_lines[4]->index(),
                                                      new_lines[6]->index()});
            subcells[1]->set_bounding_object_indices({new_lines[1]->index(),
                                                      new_lines[3]->index(),
                                                      new_lines[6]->index(),
                                                      new_lines[5]->index()});
          }

        types::subdomain_id subdomainid = cell->subdomain_id();

        for (unsigned int i = 0; i < n_children; ++i)
          {
            subcells[i]->set_used_flag();
            subcells[i]->clear_refine_flag();
            subcells[i]->clear_user_flag();
            subcells[i]->clear_user_data();
            subcells[i]->clear_children();
            // inherit material properties
            subcells[i]->set_material_id(cell->material_id());
            subcells[i]->set_manifold_id(cell->manifold_id());
            subcells[i]->set_subdomain_id(subdomainid);

            if (i % 2 == 0)
              subcells[i]->set_parent(cell->index());
          }



        // set child index for even children i=0,2 (0)
        for (unsigned int i = 0; i < n_children / 2; ++i)
          cell->set_children(2 * i, subcells[2 * i]->index());
        // set the refine case
        cell->set_refinement_case(ref_case);

        // note that the
        // refinement flag was
        // already cleared at the
        // beginning of this function

        if (dim < spacedim)
          for (unsigned int c = 0; c < n_children; ++c)
            cell->child(c)->set_direction_flag(cell->direction_flag());
      }



      template <int dim, int spacedim>
      static typename Triangulation<dim, spacedim>::DistortedCellList
      execute_refinement_isotropic(Triangulation<dim, spacedim> &triangulation,
                                   const bool check_for_distorted_cells)
      {
        AssertDimension(dim, 2);

        // Check whether a new level is needed. We have to check for
        // this on the highest level only
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               triangulation.levels.size() - 1))
          if (cell->refine_flag_set())
            {
              triangulation.levels.push_back(
                std::make_unique<
                  internal::TriangulationImplementation::TriaLevel>(dim));
              break;
            }

        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          {
            line->clear_user_flag();
            line->clear_user_data();
          }

        unsigned int n_single_lines   = 0;
        unsigned int n_lines_in_pairs = 0;
        unsigned int needed_vertices  = 0;

        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            // count number of flagged cells on this level and compute
            // how many new vertices and new lines will be needed
            unsigned int needed_cells = 0;

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  if (cell->reference_cell() ==
                      dealii::ReferenceCells::Triangle)
                    {
                      needed_cells += 4;
                      needed_vertices += 0;
                      n_single_lines += 3;
                    }
                  else if (cell->reference_cell() ==
                           dealii::ReferenceCells::Quadrilateral)
                    {
                      needed_cells += 4;
                      needed_vertices += 1;
                      n_single_lines += 4;
                    }
                  else
                    {
                      AssertThrow(false, ExcNotImplemented());
                    }

                  for (const auto line_no : cell->face_indices())
                    {
                      auto line = cell->line(line_no);
                      if (line->has_children() == false)
                        line->set_user_flag();
                    }
                }


            const unsigned int used_cells =
              std::count(triangulation.levels[level + 1]->cells.used.begin(),
                         triangulation.levels[level + 1]->cells.used.end(),
                         true);


            reserve_space(*triangulation.levels[level + 1],
                          used_cells + needed_cells,
                          2,
                          spacedim);

            reserve_space(triangulation.levels[level + 1]->cells,
                          needed_cells,
                          0);
          }

        for (auto line = triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          if (line->user_flag_set())
            {
              Assert(line->has_children() == false, ExcInternalError());
              n_lines_in_pairs += 2;
              needed_vertices += 1;
            }

        reserve_space(triangulation.faces->lines, n_lines_in_pairs, 0);

        needed_vertices += std::count(triangulation.vertices_used.begin(),
                                      triangulation.vertices_used.end(),
                                      true);

        if (needed_vertices > triangulation.vertices.size())
          {
            triangulation.vertices.resize(needed_vertices, Point<spacedim>());
            triangulation.vertices_used.resize(needed_vertices, false);
          }

        unsigned int next_unused_vertex = 0;

        {
          typename Triangulation<dim, spacedim>::active_line_iterator
            line = triangulation.begin_active_line(),
            endl = triangulation.end_line();
          typename Triangulation<dim, spacedim>::raw_line_iterator
            next_unused_line = triangulation.begin_raw_line();

          for (; line != endl; ++line)
            if (line->user_flag_set())
              {
                // this line needs to be refined

                // find the next unused vertex and set it
                // appropriately
                while (triangulation.vertices_used[next_unused_vertex] == true)
                  ++next_unused_vertex;
                Assert(
                  next_unused_vertex < triangulation.vertices.size(),
                  ExcMessage(
                    "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));
                triangulation.vertices_used[next_unused_vertex] = true;

                triangulation.vertices[next_unused_vertex] = line->center(true);

                bool pair_found = false;
                (void)pair_found;
                for (; next_unused_line != endl; ++next_unused_line)
                  if (!next_unused_line->used() &&
                      !(++next_unused_line)->used())
                    {
                      --next_unused_line;
                      pair_found = true;
                      break;
                    }
                Assert(pair_found, ExcInternalError());

                line->set_children(0, next_unused_line->index());

                const typename Triangulation<dim, spacedim>::raw_line_iterator
                  children[2] = {next_unused_line, ++next_unused_line};

                AssertIsNotUsed(children[0]);
                AssertIsNotUsed(children[1]);

                children[0]->set_bounding_object_indices(
                  {line->vertex_index(0), next_unused_vertex});
                children[1]->set_bounding_object_indices(
                  {next_unused_vertex, line->vertex_index(1)});

                children[0]->set_used_flag();
                children[1]->set_used_flag();
                children[0]->clear_children();
                children[1]->clear_children();
                children[0]->clear_user_data();
                children[1]->clear_user_data();
                children[0]->clear_user_flag();
                children[1]->clear_user_flag();


                children[0]->set_boundary_id_internal(line->boundary_id());
                children[1]->set_boundary_id_internal(line->boundary_id());

                children[0]->set_manifold_id(line->manifold_id());
                children[1]->set_manifold_id(line->manifold_id());

                line->clear_user_flag();
              }
        }

        reserve_space(triangulation.faces->lines, 0, n_single_lines);

        typename Triangulation<dim, spacedim>::DistortedCellList
          cells_with_distorted_children;

        typename Triangulation<dim, spacedim>::raw_line_iterator
          next_unused_line = triangulation.begin_raw_line();

        const auto create_children = [](auto &        triangulation,
                                        unsigned int &next_unused_vertex,
                                        auto &        next_unused_line,
                                        auto &        next_unused_cell,
                                        const auto &  cell) {
          const auto ref_case = cell->refine_flag_set();
          cell->clear_refine_flag();

          unsigned int n_new_vertices = 0;

          if (cell->reference_cell() == dealii::ReferenceCells::Triangle)
            n_new_vertices = 6;
          else if (cell->reference_cell() ==
                   dealii::ReferenceCells::Quadrilateral)
            n_new_vertices = 9;
          else
            AssertThrow(false, ExcNotImplemented());

          std::vector<int> new_vertices(n_new_vertices);
          for (unsigned int vertex_no = 0; vertex_no < cell->n_vertices();
               ++vertex_no)
            new_vertices[vertex_no] = cell->vertex_index(vertex_no);
          for (unsigned int line_no = 0; line_no < cell->n_lines(); ++line_no)
            if (cell->line(line_no)->has_children())
              new_vertices[cell->n_vertices() + line_no] =
                cell->line(line_no)->child(0)->vertex_index(1);

          if (cell->reference_cell() == dealii::ReferenceCells::Quadrilateral)
            {
              while (triangulation.vertices_used[next_unused_vertex] == true)
                ++next_unused_vertex;
              Assert(
                next_unused_vertex < triangulation.vertices.size(),
                ExcMessage(
                  "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));
              triangulation.vertices_used[next_unused_vertex] = true;

              new_vertices[8] = next_unused_vertex;

              triangulation.vertices[next_unused_vertex] =
                cell->center(true, true);
            }

          std::array<typename Triangulation<dim, spacedim>::raw_line_iterator,
                     12>
                       new_lines;
          unsigned int lmin = 0;
          unsigned int lmax = 0;

          if (cell->reference_cell() == dealii::ReferenceCells::Triangle)
            {
              lmin = 6;
              lmax = 9;
            }
          else if (cell->reference_cell() ==
                   dealii::ReferenceCells::Quadrilateral)
            {
              lmin = 8;
              lmax = 12;
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }

          for (unsigned int l = lmin; l < lmax; ++l)
            {
              while (next_unused_line->used() == true)
                ++next_unused_line;
              new_lines[l] = next_unused_line;
              ++next_unused_line;

              AssertIsNotUsed(new_lines[l]);
            }

          if (true)
            {
              if (cell->reference_cell() == dealii::ReferenceCells::Triangle)
                {
                  // add lines in the right order [TODO: clean up]
                  const auto ref = [&](const unsigned int face_no,
                                       const unsigned int vertex_no) {
                    if (cell->line(face_no)->child(0)->vertex_index(0) ==
                          static_cast<unsigned int>(new_vertices[vertex_no]) ||
                        cell->line(face_no)->child(0)->vertex_index(1) ==
                          static_cast<unsigned int>(new_vertices[vertex_no]))
                      {
                        new_lines[2 * face_no + 0] =
                          cell->line(face_no)->child(0);
                        new_lines[2 * face_no + 1] =
                          cell->line(face_no)->child(1);
                      }
                    else
                      {
                        new_lines[2 * face_no + 0] =
                          cell->line(face_no)->child(1);
                        new_lines[2 * face_no + 1] =
                          cell->line(face_no)->child(0);
                      }
                  };

                  ref(0, 0);
                  ref(1, 1);
                  ref(2, 2);

                  new_lines[6]->set_bounding_object_indices(
                    {new_vertices[3], new_vertices[4]});
                  new_lines[7]->set_bounding_object_indices(
                    {new_vertices[4], new_vertices[5]});
                  new_lines[8]->set_bounding_object_indices(
                    {new_vertices[5], new_vertices[3]});
                }
              else if (cell->reference_cell() ==
                       dealii::ReferenceCells::Quadrilateral)
                {
                  unsigned int l = 0;
                  for (const unsigned int face_no : cell->face_indices())
                    for (unsigned int c = 0; c < 2; ++c, ++l)
                      new_lines[l] = cell->line(face_no)->child(c);

                  new_lines[8]->set_bounding_object_indices(
                    {new_vertices[6], new_vertices[8]});
                  new_lines[9]->set_bounding_object_indices(
                    {new_vertices[8], new_vertices[7]});
                  new_lines[10]->set_bounding_object_indices(
                    {new_vertices[4], new_vertices[8]});
                  new_lines[11]->set_bounding_object_indices(
                    {new_vertices[8], new_vertices[5]});
                }
              else
                {
                  AssertThrow(false, ExcNotImplemented());
                }
            }


          for (unsigned int l = lmin; l < lmax; ++l)
            {
              new_lines[l]->set_used_flag();
              new_lines[l]->clear_user_flag();
              new_lines[l]->clear_user_data();
              new_lines[l]->clear_children();
              // interior line
              new_lines[l]->set_boundary_id_internal(
                numbers::internal_face_boundary_id);
              new_lines[l]->set_manifold_id(cell->manifold_id());
            }

          typename Triangulation<dim, spacedim>::raw_cell_iterator
            subcells[GeometryInfo<dim>::max_children_per_cell];
          while (next_unused_cell->used() == true)
            ++next_unused_cell;

          unsigned int n_children = 0;

          if (cell->reference_cell() == dealii::ReferenceCells::Triangle)
            n_children = 4;
          else if (cell->reference_cell() ==
                   dealii::ReferenceCells::Quadrilateral)
            n_children = 4;
          else
            AssertThrow(false, ExcNotImplemented());

          for (unsigned int i = 0; i < n_children; ++i)
            {
              AssertIsNotUsed(next_unused_cell);
              subcells[i] = next_unused_cell;
              ++next_unused_cell;
              if (i % 2 == 1 && i < n_children - 1)
                while (next_unused_cell->used() == true)
                  ++next_unused_cell;
            }

          if ((dim == 2) &&
              (cell->reference_cell() == dealii::ReferenceCells::Triangle))
            {
              subcells[0]->set_bounding_object_indices({new_lines[0]->index(),
                                                        new_lines[8]->index(),
                                                        new_lines[5]->index()});
              subcells[1]->set_bounding_object_indices({new_lines[1]->index(),
                                                        new_lines[2]->index(),
                                                        new_lines[6]->index()});
              subcells[2]->set_bounding_object_indices({new_lines[7]->index(),
                                                        new_lines[3]->index(),
                                                        new_lines[4]->index()});
              subcells[3]->set_bounding_object_indices({new_lines[6]->index(),
                                                        new_lines[7]->index(),
                                                        new_lines[8]->index()});

              // subcell 0

              const auto ref = [&](const unsigned int line_no,
                                   const unsigned int vertex_no,
                                   const unsigned int subcell_no,
                                   const unsigned int subcell_line_no) {
                if (new_lines[line_no]->vertex_index(1) !=
                    static_cast<unsigned int>(new_vertices[vertex_no]))
                  triangulation.levels[subcells[subcell_no]->level()]
                    ->face_orientations[subcells[subcell_no]->index() *
                                          GeometryInfo<2>::faces_per_cell +
                                        subcell_line_no] = 0;
              };

              ref(0, 3, 0, 0);
              ref(8, 5, 0, 1);
              ref(5, 0, 0, 2);

              ref(1, 1, 1, 0);
              ref(2, 4, 1, 1);
              ref(6, 3, 1, 2);

              ref(7, 4, 2, 0);
              ref(3, 2, 2, 1);
              ref(4, 5, 2, 2);

              ref(6, 4, 3, 0);
              ref(7, 5, 3, 1);
              ref(8, 3, 3, 2);

              // triangulation.levels[subcells[1]->level()]->face_orientations[subcells[1]->index()
              // * GeometryInfo<2>::faces_per_cell + 2] = 0;
              // triangulation.levels[subcells[2]->level()]->face_orientations[subcells[2]->index()
              // * GeometryInfo<2>::faces_per_cell + 0] = 0;
            }
          else if ((dim == 2) && (cell->reference_cell() ==
                                  dealii::ReferenceCells::Quadrilateral))
            {
              subcells[0]->set_bounding_object_indices(
                {new_lines[0]->index(),
                 new_lines[8]->index(),
                 new_lines[4]->index(),
                 new_lines[10]->index()});
              subcells[1]->set_bounding_object_indices(
                {new_lines[8]->index(),
                 new_lines[2]->index(),
                 new_lines[5]->index(),
                 new_lines[11]->index()});
              subcells[2]->set_bounding_object_indices({new_lines[1]->index(),
                                                        new_lines[9]->index(),
                                                        new_lines[10]->index(),
                                                        new_lines[6]->index()});
              subcells[3]->set_bounding_object_indices({new_lines[9]->index(),
                                                        new_lines[3]->index(),
                                                        new_lines[11]->index(),
                                                        new_lines[7]->index()});
            }
          else
            {
              AssertThrow(false, ExcNotImplemented());
            }

          types::subdomain_id subdomainid = cell->subdomain_id();

          for (unsigned int i = 0; i < n_children; ++i)
            {
              subcells[i]->set_used_flag();
              subcells[i]->clear_refine_flag();
              subcells[i]->clear_user_flag();
              subcells[i]->clear_user_data();
              subcells[i]->clear_children();
              // inherit material
              // properties
              subcells[i]->set_material_id(cell->material_id());
              subcells[i]->set_manifold_id(cell->manifold_id());
              subcells[i]->set_subdomain_id(subdomainid);

              // TODO: here we assume that all children have the same reference
              // cell type as the parent! This is justified for 2D.
              triangulation.levels[subcells[i]->level()]
                ->reference_cell[subcells[i]->index()] = cell->reference_cell();

              if (i % 2 == 0)
                subcells[i]->set_parent(cell->index());
            }

          for (unsigned int i = 0; i < n_children / 2; ++i)
            cell->set_children(2 * i, subcells[2 * i]->index());

          cell->set_refinement_case(ref_case);

          if (dim < spacedim)
            for (unsigned int c = 0; c < n_children; ++c)
              cell->child(c)->set_direction_flag(cell->direction_flag());
        };

        for (int level = 0;
             level < static_cast<int>(triangulation.levels.size()) - 1;
             ++level)
          {
            typename Triangulation<dim, spacedim>::raw_cell_iterator
              next_unused_cell = triangulation.begin_raw(level + 1);

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  create_children(triangulation,
                                  next_unused_vertex,
                                  next_unused_line,
                                  next_unused_cell,
                                  cell);

                  if (cell->reference_cell() ==
                        dealii::ReferenceCells::Quadrilateral &&
                      check_for_distorted_cells &&
                      has_distorted_children<dim, spacedim>(cell))
                    cells_with_distorted_children.distorted_cells.push_back(
                      cell);

                  triangulation.signals.post_refinement_on_cell(cell);
                }
          }

        return cells_with_distorted_children;
      }



      /**
       * A function that performs the
       * refinement of a triangulation in 1d.
       */
      template <int spacedim>
      static typename Triangulation<1, spacedim>::DistortedCellList
      execute_refinement(Triangulation<1, spacedim> &triangulation,
                         const bool /*check_for_distorted_cells*/)
      {
        const unsigned int dim = 1;

        // Check whether a new level is needed. We have to check for
        // this on the highest level only
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               triangulation.levels.size() - 1))
          if (cell->refine_flag_set())
            {
              triangulation.levels.push_back(
                std::make_unique<
                  internal::TriangulationImplementation::TriaLevel>(dim));
              break;
            }


        // check how much space is needed on every level. We need not
        // check the highest level since either - on the highest level
        // no cells are flagged for refinement - there are, but
        // prepare_refinement added another empty level
        unsigned int needed_vertices = 0;
        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            // count number of flagged
            // cells on this level
            unsigned int flagged_cells = 0;

            for (const auto &acell :
                 triangulation.active_cell_iterators_on_level(level))
              if (acell->refine_flag_set())
                ++flagged_cells;

            // count number of used cells
            // on the next higher level
            const unsigned int used_cells =
              std::count(triangulation.levels[level + 1]->cells.used.begin(),
                         triangulation.levels[level + 1]->cells.used.end(),
                         true);

            // reserve space for the used_cells cells already existing
            // on the next higher level as well as for the
            // 2*flagged_cells that will be created on that level
            reserve_space(*triangulation.levels[level + 1],
                          used_cells + GeometryInfo<1>::max_children_per_cell *
                                         flagged_cells,
                          1,
                          spacedim);
            // reserve space for 2*flagged_cells new lines on the next
            // higher level
            reserve_space(triangulation.levels[level + 1]->cells,
                          GeometryInfo<1>::max_children_per_cell *
                            flagged_cells,
                          0);

            needed_vertices += flagged_cells;
          }

        // add to needed vertices how many
        // vertices are already in use
        needed_vertices += std::count(triangulation.vertices_used.begin(),
                                      triangulation.vertices_used.end(),
                                      true);
        // if we need more vertices: create them, if not: leave the
        // array as is, since shrinking is not really possible because
        // some of the vertices at the end may be in use
        if (needed_vertices > triangulation.vertices.size())
          {
            triangulation.vertices.resize(needed_vertices, Point<spacedim>());
            triangulation.vertices_used.resize(needed_vertices, false);
          }


        // Do REFINEMENT on every level; exclude highest level as
        // above

        // index of next unused vertex
        unsigned int next_unused_vertex = 0;

        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            typename Triangulation<dim, spacedim>::raw_cell_iterator
              next_unused_cell = triangulation.begin_raw(level + 1);

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  // clear refinement flag
                  cell->clear_refine_flag();

                  // search for next unused
                  // vertex
                  while (triangulation.vertices_used[next_unused_vertex] ==
                         true)
                    ++next_unused_vertex;
                  Assert(
                    next_unused_vertex < triangulation.vertices.size(),
                    ExcMessage(
                      "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));

                  // Now we always ask the cell itself where to put
                  // the new point. The cell in turn will query the
                  // manifold object internally.
                  triangulation.vertices[next_unused_vertex] =
                    cell->center(true);

                  triangulation.vertices_used[next_unused_vertex] = true;

                  // search for next two unused cell (++ takes care of
                  // the end of the vector)
                  typename Triangulation<dim, spacedim>::raw_cell_iterator
                    first_child,
                    second_child;
                  while (next_unused_cell->used() == true)
                    ++next_unused_cell;
                  first_child = next_unused_cell;
                  first_child->set_used_flag();
                  first_child->clear_user_data();
                  ++next_unused_cell;
                  AssertIsNotUsed(next_unused_cell);
                  second_child = next_unused_cell;
                  second_child->set_used_flag();
                  second_child->clear_user_data();

                  types::subdomain_id subdomainid = cell->subdomain_id();

                  // insert first child
                  cell->set_children(0, first_child->index());
                  first_child->clear_children();
                  first_child->set_bounding_object_indices(
                    {cell->vertex_index(0), next_unused_vertex});
                  first_child->set_material_id(cell->material_id());
                  first_child->set_manifold_id(cell->manifold_id());
                  first_child->set_subdomain_id(subdomainid);
                  first_child->set_direction_flag(cell->direction_flag());

                  first_child->set_parent(cell->index());

                  // Set manifold id of the right face. Only do this
                  // on the first child.
                  first_child->face(1)->set_manifold_id(cell->manifold_id());

                  // reset neighborship info (refer to
                  // internal::TriangulationImplementation::TriaLevel<0> for
                  // details)
                  first_child->set_neighbor(1, second_child);
                  if (cell->neighbor(0).state() != IteratorState::valid)
                    first_child->set_neighbor(0, cell->neighbor(0));
                  else if (cell->neighbor(0)->is_active())
                    {
                      // since the neighbors level is always <=level,
                      // if the cell is active, then there are no
                      // cells to the left which may want to know
                      // about this new child cell.
                      Assert(cell->neighbor(0)->level() <= cell->level(),
                             ExcInternalError());
                      first_child->set_neighbor(0, cell->neighbor(0));
                    }
                  else
                    // left neighbor is refined
                    {
                      // set neighbor to cell on same level
                      const unsigned int nbnb = cell->neighbor_of_neighbor(0);
                      first_child->set_neighbor(0,
                                                cell->neighbor(0)->child(nbnb));

                      // reset neighbor info of all right descendant
                      // of the left neighbor of cell
                      typename Triangulation<dim, spacedim>::cell_iterator
                        left_neighbor = cell->neighbor(0);
                      while (left_neighbor->has_children())
                        {
                          left_neighbor = left_neighbor->child(nbnb);
                          left_neighbor->set_neighbor(nbnb, first_child);
                        }
                    }

                  // insert second child
                  second_child->clear_children();
                  second_child->set_bounding_object_indices(
                    {next_unused_vertex, cell->vertex_index(1)});
                  second_child->set_neighbor(0, first_child);
                  second_child->set_material_id(cell->material_id());
                  second_child->set_manifold_id(cell->manifold_id());
                  second_child->set_subdomain_id(subdomainid);
                  second_child->set_direction_flag(cell->direction_flag());

                  if (cell->neighbor(1).state() != IteratorState::valid)
                    second_child->set_neighbor(1, cell->neighbor(1));
                  else if (cell->neighbor(1)->is_active())
                    {
                      Assert(cell->neighbor(1)->level() <= cell->level(),
                             ExcInternalError());
                      second_child->set_neighbor(1, cell->neighbor(1));
                    }
                  else
                    // right neighbor is refined same as above
                    {
                      const unsigned int nbnb = cell->neighbor_of_neighbor(1);
                      second_child->set_neighbor(
                        1, cell->neighbor(1)->child(nbnb));

                      typename Triangulation<dim, spacedim>::cell_iterator
                        right_neighbor = cell->neighbor(1);
                      while (right_neighbor->has_children())
                        {
                          right_neighbor = right_neighbor->child(nbnb);
                          right_neighbor->set_neighbor(nbnb, second_child);
                        }
                    }
                  // inform all listeners that cell refinement is done
                  triangulation.signals.post_refinement_on_cell(cell);
                }
          }

        // in 1d, we can not have distorted children unless the parent
        // was already distorted (that is because we don't use
        // boundary information for 1d triangulations). so return an
        // empty list
        return typename Triangulation<1, spacedim>::DistortedCellList();
      }


      /**
       * A function that performs the refinement of a triangulation in
       * 2d.
       */
      template <int spacedim>
      static typename Triangulation<2, spacedim>::DistortedCellList
      execute_refinement(Triangulation<2, spacedim> &triangulation,
                         const bool                  check_for_distorted_cells)
      {
        const unsigned int dim = 2;


        // First check whether we can get away with isotropic refinement, or
        // whether we need to run through the full anisotropic algorithm
        {
          bool do_isotropic_refinement = true;
          for (const auto &cell : triangulation.active_cell_iterators())
            if (cell->refine_flag_set() == RefinementCase<dim>::cut_x ||
                cell->refine_flag_set() == RefinementCase<dim>::cut_y)
              {
                do_isotropic_refinement = false;
                break;
              }

          if (do_isotropic_refinement)
            return execute_refinement_isotropic(triangulation,
                                                check_for_distorted_cells);
        }

        // Check whether a new level is needed. We have to check for
        // this on the highest level only
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               triangulation.levels.size() - 1))
          if (cell->refine_flag_set())
            {
              triangulation.levels.push_back(
                std::make_unique<
                  internal::TriangulationImplementation::TriaLevel>(dim));
              break;
            }

        // TODO[WB]: we clear user flags and pointers of lines; we're going
        // to use them to flag which lines need refinement
        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          {
            line->clear_user_flag();
            line->clear_user_data();
          }
        // running over all cells and lines count the number
        // n_single_lines of lines which can be stored as single
        // lines, e.g. inner lines
        unsigned int n_single_lines = 0;

        // New lines to be created: number lines which are stored in
        // pairs (the children of lines must be stored in pairs)
        unsigned int n_lines_in_pairs = 0;

        // check how much space is needed on every level. We need not
        // check the highest level since either - on the highest level
        // no cells are flagged for refinement - there are, but
        // prepare_refinement added another empty level
        unsigned int needed_vertices = 0;
        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            // count number of flagged cells on this level and compute
            // how many new vertices and new lines will be needed
            unsigned int needed_cells = 0;

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  if (cell->refine_flag_set() == RefinementCase<dim>::cut_xy)
                    {
                      needed_cells += 4;

                      // new vertex at center of cell is needed in any
                      // case
                      ++needed_vertices;

                      // the four inner lines can be stored as singles
                      n_single_lines += 4;
                    }
                  else // cut_x || cut_y
                    {
                      // set the flag showing that anisotropic
                      // refinement is used for at least one cell
                      triangulation.anisotropic_refinement = true;

                      needed_cells += 2;
                      // no vertex at center

                      // the inner line can be stored as single
                      n_single_lines += 1;
                    }

                  // mark all faces (lines) for refinement; checking
                  // locally whether the neighbor would also like to
                  // refine them is rather difficult for lines so we
                  // only flag them and after visiting all cells, we
                  // decide which lines need refinement;
                  for (const unsigned int line_no :
                       GeometryInfo<dim>::face_indices())
                    {
                      if (GeometryInfo<dim>::face_refinement_case(
                            cell->refine_flag_set(), line_no) ==
                          RefinementCase<1>::cut_x)
                        {
                          typename Triangulation<dim, spacedim>::line_iterator
                            line = cell->line(line_no);
                          if (line->has_children() == false)
                            line->set_user_flag();
                        }
                    }
                }


            // count number of used cells on the next higher level
            const unsigned int used_cells =
              std::count(triangulation.levels[level + 1]->cells.used.begin(),
                         triangulation.levels[level + 1]->cells.used.end(),
                         true);


            // reserve space for the used_cells cells already existing
            // on the next higher level as well as for the
            // needed_cells that will be created on that level
            reserve_space(*triangulation.levels[level + 1],
                          used_cells + needed_cells,
                          2,
                          spacedim);

            // reserve space for needed_cells new quads on the next
            // higher level
            reserve_space(triangulation.levels[level + 1]->cells,
                          needed_cells,
                          0);
          }

        // now count the lines which were flagged for refinement
        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          if (line->user_flag_set())
            {
              Assert(line->has_children() == false, ExcInternalError());
              n_lines_in_pairs += 2;
              needed_vertices += 1;
            }
        // reserve space for n_lines_in_pairs new lines.  note, that
        // we can't reserve space for the single lines here as well,
        // as all the space reserved for lines in pairs would be
        // counted as unused and we would end up with too little space
        // to store all lines. memory reservation for n_single_lines
        // can only be done AFTER we refined the lines of the current
        // cells
        reserve_space(triangulation.faces->lines, n_lines_in_pairs, 0);

        // add to needed vertices how many vertices are already in use
        needed_vertices += std::count(triangulation.vertices_used.begin(),
                                      triangulation.vertices_used.end(),
                                      true);
        // if we need more vertices: create them, if not: leave the
        // array as is, since shrinking is not really possible because
        // some of the vertices at the end may be in use
        if (needed_vertices > triangulation.vertices.size())
          {
            triangulation.vertices.resize(needed_vertices, Point<spacedim>());
            triangulation.vertices_used.resize(needed_vertices, false);
          }


        // Do REFINEMENT on every level; exclude highest level as
        // above

        //  index of next unused vertex
        unsigned int next_unused_vertex = 0;

        // first the refinement of lines.  children are stored
        // pairwise
        {
          // only active objects can be refined further
          typename Triangulation<dim, spacedim>::active_line_iterator
            line = triangulation.begin_active_line(),
            endl = triangulation.end_line();
          typename Triangulation<dim, spacedim>::raw_line_iterator
            next_unused_line = triangulation.begin_raw_line();

          for (; line != endl; ++line)
            if (line->user_flag_set())
              {
                // this line needs to be refined

                // find the next unused vertex and set it
                // appropriately
                while (triangulation.vertices_used[next_unused_vertex] == true)
                  ++next_unused_vertex;
                Assert(
                  next_unused_vertex < triangulation.vertices.size(),
                  ExcMessage(
                    "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));
                triangulation.vertices_used[next_unused_vertex] = true;

                triangulation.vertices[next_unused_vertex] = line->center(true);

                // now that we created the right point, make up the
                // two child lines.  To this end, find a pair of
                // unused lines
                bool pair_found = false;
                (void)pair_found;
                for (; next_unused_line != endl; ++next_unused_line)
                  if (!next_unused_line->used() &&
                      !(++next_unused_line)->used())
                    {
                      // go back to the first of the two unused
                      // lines
                      --next_unused_line;
                      pair_found = true;
                      break;
                    }
                Assert(pair_found, ExcInternalError());

                // there are now two consecutive unused lines, such
                // that the children of a line will be consecutive.
                // then set the child pointer of the present line
                line->set_children(0, next_unused_line->index());

                // set the two new lines
                const typename Triangulation<dim, spacedim>::raw_line_iterator
                  children[2] = {next_unused_line, ++next_unused_line};
                // some tests; if any of the iterators should be
                // invalid, then already dereferencing will fail
                AssertIsNotUsed(children[0]);
                AssertIsNotUsed(children[1]);

                children[0]->set_bounding_object_indices(
                  {line->vertex_index(0), next_unused_vertex});
                children[1]->set_bounding_object_indices(
                  {next_unused_vertex, line->vertex_index(1)});

                children[0]->set_used_flag();
                children[1]->set_used_flag();
                children[0]->clear_children();
                children[1]->clear_children();
                children[0]->clear_user_data();
                children[1]->clear_user_data();
                children[0]->clear_user_flag();
                children[1]->clear_user_flag();


                children[0]->set_boundary_id_internal(line->boundary_id());
                children[1]->set_boundary_id_internal(line->boundary_id());

                children[0]->set_manifold_id(line->manifold_id());
                children[1]->set_manifold_id(line->manifold_id());

                // finally clear flag indicating the need for
                // refinement
                line->clear_user_flag();
              }
        }


        // Now set up the new cells

        // reserve space for inner lines (can be stored as single
        // lines)
        reserve_space(triangulation.faces->lines, 0, n_single_lines);

        typename Triangulation<2, spacedim>::DistortedCellList
          cells_with_distorted_children;

        // reset next_unused_line, as now also single empty places in
        // the vector can be used
        typename Triangulation<dim, spacedim>::raw_line_iterator
          next_unused_line = triangulation.begin_raw_line();

        for (int level = 0;
             level < static_cast<int>(triangulation.levels.size()) - 1;
             ++level)
          {
            typename Triangulation<dim, spacedim>::raw_cell_iterator
              next_unused_cell = triangulation.begin_raw(level + 1);

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  // actually set up the children and update neighbor
                  // information
                  create_children(triangulation,
                                  next_unused_vertex,
                                  next_unused_line,
                                  next_unused_cell,
                                  cell);

                  if (check_for_distorted_cells &&
                      has_distorted_children<dim, spacedim>(cell))
                    cells_with_distorted_children.distorted_cells.push_back(
                      cell);
                  // inform all listeners that cell refinement is done
                  triangulation.signals.post_refinement_on_cell(cell);
                }
          }

        return cells_with_distorted_children;
      }


      template <int spacedim>
      static typename Triangulation<3, spacedim>::DistortedCellList
      execute_refinement_isotropic(Triangulation<3, spacedim> &triangulation,
                                   const bool check_for_distorted_cells)
      {
        static const int          dim = 3;
        static const unsigned int X   = numbers::invalid_unsigned_int;

        Assert(spacedim == 3, ExcNotImplemented());

        Assert(triangulation.vertices.size() ==
                 triangulation.vertices_used.size(),
               ExcInternalError());

        // Check whether a new level is needed. We have to check for
        // this on the highest level only
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               triangulation.levels.size() - 1))
          if (cell->refine_flag_set())
            {
              triangulation.levels.push_back(
                std::make_unique<
                  internal::TriangulationImplementation::TriaLevel>(dim));
              break;
            }

        // first clear user flags for quads and lines; we're going to
        // use them to flag which lines and quads need refinement
        triangulation.faces->quads.clear_user_data();

        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          line->clear_user_flag();

        for (typename Triangulation<dim, spacedim>::quad_iterator quad =
               triangulation.begin_quad();
             quad != triangulation.end_quad();
             ++quad)
          quad->clear_user_flag();

        // check how much space is needed on every level. We need not
        // check the highest level since either
        // - on the highest level no cells are flagged for refinement
        // - there are, but prepare_refinement added another empty
        //   level which then is the highest level

        // variables to hold the number of newly to be created
        // vertices, lines and quads. as these are stored globally,
        // declare them outside the loop over al levels. we need lines
        // and quads in pairs for refinement of old ones and lines and
        // quads, that can be stored as single ones, as they are newly
        // created in the inside of an existing cell
        unsigned int needed_vertices     = 0;
        unsigned int needed_lines_single = 0;
        unsigned int needed_quads_single = 0;
        unsigned int needed_lines_pair   = 0;
        unsigned int needed_quads_pair   = 0;
        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            unsigned int new_cells = 0;

            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(level))
              if (cell->refine_flag_set())
                {
                  // Only support isotropic refinement
                  Assert(cell->refine_flag_set() ==
                           RefinementCase<dim>::cut_xyz,
                         ExcInternalError());

                  // Now count up how many new cells, faces, edges, and vertices
                  // we will need to allocate to do this refinement.
                  new_cells += cell->reference_cell().n_isotropic_children();

                  if (cell->reference_cell() == ReferenceCells::Hexahedron)
                    {
                      ++needed_vertices;
                      needed_lines_single += 6;
                      needed_quads_single += 12;
                    }
                  else if (cell->reference_cell() ==
                           ReferenceCells::Tetrahedron)
                    {
                      needed_lines_single += 1;
                      needed_quads_single += 8;
                    }
                  else
                    {
                      Assert(false, ExcInternalError());
                    }

                  // Also check whether we have to refine any of the faces and
                  // edges that bound this cell. They may of course already be
                  // refined, so we only *mark* them for refinement by setting
                  // the user flags
                  for (const auto face : cell->face_indices())
                    if (cell->face(face)->n_children() == 0)
                      cell->face(face)->set_user_flag();
                    else
                      Assert(cell->face(face)->n_children() ==
                               cell->reference_cell()
                                 .face_reference_cell(face)
                                 .n_isotropic_children(),
                             ExcInternalError());

                  for (const auto line : cell->line_indices())
                    if (cell->line(line)->has_children() == false)
                      cell->line(line)->set_user_flag();
                    else
                      Assert(cell->line(line)->n_children() == 2,
                             ExcInternalError());
                }

            const unsigned int used_cells =
              std::count(triangulation.levels[level + 1]->cells.used.begin(),
                         triangulation.levels[level + 1]->cells.used.end(),
                         true);

            reserve_space(*triangulation.levels[level + 1],
                          used_cells + new_cells,
                          3,
                          spacedim);

            reserve_space(triangulation.levels[level + 1]->cells, new_cells);
          }

        // now count the quads and lines which were flagged for
        // refinement
        for (typename Triangulation<dim, spacedim>::quad_iterator quad =
               triangulation.begin_quad();
             quad != triangulation.end_quad();
             ++quad)
          {
            if (quad->user_flag_set() == false)
              continue;

            if (quad->reference_cell() == ReferenceCells::Quadrilateral)
              {
                needed_quads_pair += 4;
                needed_lines_pair += 4;
                needed_vertices += 1;
              }
            else if (quad->reference_cell() == ReferenceCells::Triangle)
              {
                needed_quads_pair += 4;
                needed_lines_single += 3;
              }
            else
              {
                Assert(false, ExcInternalError());
              }
          }

        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          {
            if (line->user_flag_set() == false)
              continue;

            needed_lines_pair += 2;
            needed_vertices += 1;
          }

        reserve_space(triangulation.faces->lines,
                      needed_lines_pair,
                      needed_lines_single);
        reserve_space(*triangulation.faces,
                      needed_quads_pair,
                      needed_quads_single);
        reserve_space(triangulation.faces->quads,
                      needed_quads_pair,
                      needed_quads_single);


        // add to needed vertices how many vertices are already in use
        needed_vertices += std::count(triangulation.vertices_used.begin(),
                                      triangulation.vertices_used.end(),
                                      true);

        if (needed_vertices > triangulation.vertices.size())
          {
            triangulation.vertices.resize(needed_vertices, Point<spacedim>());
            triangulation.vertices_used.resize(needed_vertices, false);
          }

          //-----------------------------------------
          // Before we start with the actual refinement, we do some
          // sanity checks if in debug mode. especially, we try to catch
          // the notorious problem with lines being twice refined,
          // i.e. there are cells adjacent at one line ("around the
          // edge", but not at a face), with two cells differing by more
          // than one refinement level
          //
          // this check is very simple to implement here, since we have
          // all lines flagged if they shall be refined
#ifdef DEBUG
        for (const auto &cell : triangulation.active_cell_iterators())
          if (!cell->refine_flag_set())
            for (unsigned int line_n = 0; line_n < cell->n_lines(); ++line_n)
              if (cell->line(line_n)->has_children())
                for (unsigned int c = 0; c < 2; ++c)
                  Assert(cell->line(line_n)->child(c)->user_flag_set() == false,
                         ExcInternalError());
#endif

        unsigned int current_vertex = 0;

        // helper function - find the next available vertex number and mark it
        // as used.
        auto get_next_unused_vertex = [](const unsigned int current_vertex,
                                         std::vector<bool> &vertices_used) {
          unsigned int next_vertex = current_vertex;
          while (next_vertex < vertices_used.size() &&
                 vertices_used[next_vertex] == true)
            ++next_vertex;
          Assert(next_vertex < vertices_used.size(), ExcInternalError());
          vertices_used[next_vertex] = true;

          return next_vertex;
        };

        // LINES
        {
          typename Triangulation<dim, spacedim>::active_line_iterator
            line = triangulation.begin_active_line(),
            endl = triangulation.end_line();
          typename Triangulation<dim, spacedim>::raw_line_iterator
            next_unused_line = triangulation.begin_raw_line();

          for (; line != endl; ++line)
            {
              if (line->user_flag_set() == false)
                continue;

              current_vertex =
                get_next_unused_vertex(current_vertex,
                                       triangulation.vertices_used);
              triangulation.vertices[current_vertex] = line->center(true);

              next_unused_line =
                triangulation.faces->lines.template next_free_pair_object<1>(
                  triangulation);
              Assert(next_unused_line.state() == IteratorState::valid,
                     ExcInternalError());

              // now we found two consecutive unused lines, such
              // that the children of a line will be consecutive.
              // then set the child pointer of the present line
              line->set_children(0, next_unused_line->index());

              const typename Triangulation<dim, spacedim>::raw_line_iterator
                children[2] = {next_unused_line, ++next_unused_line};

              AssertIsNotUsed(children[0]);
              AssertIsNotUsed(children[1]);

              children[0]->set_bounding_object_indices(
                {line->vertex_index(0), current_vertex});
              children[1]->set_bounding_object_indices(
                {current_vertex, line->vertex_index(1)});

              children[0]->set_used_flag();
              children[1]->set_used_flag();
              children[0]->clear_children();
              children[1]->clear_children();
              children[0]->clear_user_data();
              children[1]->clear_user_data();
              children[0]->clear_user_flag();
              children[1]->clear_user_flag();

              children[0]->set_boundary_id_internal(line->boundary_id());
              children[1]->set_boundary_id_internal(line->boundary_id());

              children[0]->set_manifold_id(line->manifold_id());
              children[1]->set_manifold_id(line->manifold_id());

              line->clear_user_flag();
            }
        }

        // QUADS
        {
          typename Triangulation<dim, spacedim>::quad_iterator
            quad = triangulation.begin_quad(),
            endq = triangulation.end_quad();
          typename Triangulation<dim, spacedim>::raw_line_iterator
            next_unused_line = triangulation.begin_raw_line();
          typename Triangulation<dim, spacedim>::raw_quad_iterator
            next_unused_quad = triangulation.begin_raw_quad();

          for (; quad != endq; ++quad)
            {
              if (quad->user_flag_set() == false)
                continue;

              const auto reference_face_type = quad->reference_cell();

              // 1) create new vertex (at the center of the face)
              if (reference_face_type == ReferenceCells::Quadrilateral)
                {
                  current_vertex =
                    get_next_unused_vertex(current_vertex,
                                           triangulation.vertices_used);
                  triangulation.vertices[current_vertex] =
                    quad->center(true, true);
                }

              // 2) create new lines (property is set later)
              boost::container::small_vector<
                typename Triangulation<dim, spacedim>::raw_line_iterator,
                GeometryInfo<dim>::lines_per_cell>
                new_lines(quad->n_lines());
              {
                for (unsigned int i = 0; i < new_lines.size(); ++i)
                  {
                    if (reference_face_type == ReferenceCells::Quadrilateral)
                      {
                        if (i % 2 == 0)
                          next_unused_line =
                            triangulation.faces->lines
                              .template next_free_pair_object<1>(triangulation);
                      }
                    else if (reference_face_type == ReferenceCells::Triangle)
                      {
                        next_unused_line =
                          triangulation.faces->lines
                            .template next_free_single_object<1>(triangulation);
                      }
                    else
                      {
                        Assert(false, ExcNotImplemented());
                      }

                    new_lines[i] = next_unused_line;
                    ++next_unused_line;
                    AssertIsNotUsed(new_lines[i]);
                  }
              }

              // 3) create new quads (properties are set below). Both triangles
              // and quads are divided in four.
              std::array<
                typename Triangulation<dim, spacedim>::raw_quad_iterator,
                4>
                new_quads;
              {
                next_unused_quad =
                  triangulation.faces->quads.template next_free_pair_object<2>(
                    triangulation);

                new_quads[0] = next_unused_quad;
                AssertIsNotUsed(new_quads[0]);

                ++next_unused_quad;
                new_quads[1] = next_unused_quad;
                AssertIsNotUsed(new_quads[1]);

                next_unused_quad =
                  triangulation.faces->quads.template next_free_pair_object<2>(
                    triangulation);
                new_quads[2] = next_unused_quad;
                AssertIsNotUsed(new_quads[2]);

                ++next_unused_quad;
                new_quads[3] = next_unused_quad;
                AssertIsNotUsed(new_quads[3]);

                quad->set_children(0, new_quads[0]->index());
                quad->set_children(2, new_quads[2]->index());
                quad->set_refinement_case(RefinementCase<2>::cut_xy);
              }

              // Maximum of 9 vertices per refined quad (9 for Quadrilateral, 6
              // for Triangle)
              std::array<unsigned int, 9> vertex_indices = {};
              {
                unsigned int k = 0;
                for (const auto i : quad->vertex_indices())
                  vertex_indices[k++] = quad->vertex_index(i);

                for (const auto i : quad->line_indices())
                  vertex_indices[k++] =
                    quad->line(i)->child(0)->vertex_index(1);

                vertex_indices[k++] = current_vertex;
              }

              boost::container::small_vector<
                typename Triangulation<dim, spacedim>::raw_line_iterator,
                12>
                lines(reference_face_type == ReferenceCells::Quadrilateral ?
                        12 :
                        9);
              {
                unsigned int k = 0;

                for (unsigned int l = 0; l < quad->n_lines(); ++l)
                  for (unsigned int c = 0; c < 2; ++c)
                    {
                      static constexpr std::array<std::array<unsigned int, 2>,
                                                  2>
                        index = {// child 0, line_orientation=false and true
                                 {{{1, 0}},
                                  // child 1, line_orientation=false and true
                                  {{0, 1}}}};

                      lines[k++] = quad->line(l)->child(
                        index[c][quad->line_orientation(l)]);
                    }

                for (unsigned int l = 0; l < new_lines.size(); ++l)
                  lines[k++] = new_lines[l];
              }

              boost::container::small_vector<int, 12> line_indices(
                lines.size());
              for (unsigned int i = 0; i < line_indices.size(); ++i)
                line_indices[i] = lines[i]->index();

              static constexpr std::array<std::array<unsigned int, 2>, 12>
                line_vertices_quad{{{{0, 4}},
                                    {{4, 2}},
                                    {{1, 5}},
                                    {{5, 3}},
                                    {{0, 6}},
                                    {{6, 1}},
                                    {{2, 7}},
                                    {{7, 3}},
                                    {{6, 8}},
                                    {{8, 7}},
                                    {{4, 8}},
                                    {{8, 5}}}};

              static constexpr std::array<std::array<unsigned int, 4>, 4>
                quad_lines_quad{{{{0, 8, 4, 10}},
                                 {{8, 2, 5, 11}},
                                 {{1, 9, 10, 6}},
                                 {{9, 3, 11, 7}}}};

              static constexpr std::
                array<std::array<std::array<unsigned int, 2>, 4>, 4>
                  quad_line_vertices_quad{
                    {{{{{0, 4}}, {{6, 8}}, {{0, 6}}, {{4, 8}}}},
                     {{{{6, 8}}, {{1, 5}}, {{6, 1}}, {{8, 5}}}},
                     {{{{4, 2}}, {{8, 7}}, {{4, 8}}, {{2, 7}}}},
                     {{{{8, 7}}, {{5, 3}}, {{8, 5}}, {{7, 3}}}}}};

              static constexpr std::array<std::array<unsigned int, 2>, 12>
                line_vertices_tri{{{{0, 3}},
                                   {{3, 1}},
                                   {{1, 4}},
                                   {{4, 2}},
                                   {{2, 5}},
                                   {{5, 0}},
                                   {{3, 4}},
                                   {{4, 5}},
                                   {{3, 5}},
                                   {{X, X}},
                                   {{X, X}},
                                   {{X, X}}}};

              static constexpr std::array<std::array<unsigned int, 4>, 4>
                quad_lines_tri{{{{0, 8, 5, X}},
                                {{1, 2, 6, X}},
                                {{7, 3, 4, X}},
                                {{6, 7, 8, X}}}};

              static constexpr std::
                array<std::array<std::array<unsigned int, 2>, 4>, 4>
                  quad_line_vertices_tri{
                    {{{{{0, 3}}, {{3, 5}}, {{5, 0}}, {{X, X}}}},
                     {{{{3, 1}}, {{1, 4}}, {{4, 3}}, {{X, X}}}},
                     {{{{5, 4}}, {{4, 2}}, {{2, 5}}, {{X, X}}}},
                     {{{{3, 4}}, {{4, 5}}, {{5, 3}}, {{X, X}}}}}};

              const auto &line_vertices =
                (reference_face_type == ReferenceCells::Quadrilateral) ?
                  line_vertices_quad :
                  line_vertices_tri;
              const auto &quad_lines =
                (reference_face_type == ReferenceCells::Quadrilateral) ?
                  quad_lines_quad :
                  quad_lines_tri;
              const auto &quad_line_vertices =
                (reference_face_type == ReferenceCells::Quadrilateral) ?
                  quad_line_vertices_quad :
                  quad_line_vertices_tri;

              // 4) set properties of lines
              for (unsigned int i = 0, j = lines.size() - new_lines.size();
                   i < new_lines.size();
                   ++i, ++j)
                {
                  auto &new_line = new_lines[i];
                  new_line->set_bounding_object_indices(
                    {vertex_indices[line_vertices[j][0]],
                     vertex_indices[line_vertices[j][1]]});
                  new_line->set_used_flag();
                  new_line->clear_user_flag();
                  new_line->clear_user_data();
                  new_line->clear_children();
                  new_line->set_boundary_id_internal(quad->boundary_id());
                  new_line->set_manifold_id(quad->manifold_id());
                }

              // 5) set properties of quads
              for (unsigned int i = 0; i < new_quads.size(); ++i)
                {
                  auto &new_quad = new_quads[i];

                  // TODO: we assume here that all children have the same type
                  // as the parent
                  triangulation.faces->quad_reference_cell[new_quad->index()] =
                    reference_face_type;

                  if (new_quad->n_lines() == 3)
                    new_quad->set_bounding_object_indices(
                      {line_indices[quad_lines[i][0]],
                       line_indices[quad_lines[i][1]],
                       line_indices[quad_lines[i][2]]});
                  else if (new_quad->n_lines() == 4)
                    new_quad->set_bounding_object_indices(
                      {line_indices[quad_lines[i][0]],
                       line_indices[quad_lines[i][1]],
                       line_indices[quad_lines[i][2]],
                       line_indices[quad_lines[i][3]]});
                  else
                    Assert(false, ExcNotImplemented());

                  new_quad->set_used_flag();
                  new_quad->clear_user_flag();
                  new_quad->clear_user_data();
                  new_quad->clear_children();
                  new_quad->set_boundary_id_internal(quad->boundary_id());
                  new_quad->set_manifold_id(quad->manifold_id());

#ifdef DEBUG
                  std::set<unsigned int> s;
#endif

                  // ... and fix orientation of faces (lines) of quad
                  for (const auto f : new_quad->line_indices())
                    {
                      std::array<unsigned int, 2> vertices_0, vertices_1;

                      for (unsigned int v = 0; v < 2; ++v)
                        vertices_0[v] =
                          lines[quad_lines[i][f]]->vertex_index(v);

                      for (unsigned int v = 0; v < 2; ++v)
                        vertices_1[v] =
                          vertex_indices[quad_line_vertices[i][f][v]];

                      const auto orientation =
                        ReferenceCells::Line.compute_orientation(vertices_0,
                                                                 vertices_1);

#ifdef DEBUG
                      for (const auto i : vertices_0)
                        s.insert(i);
                      for (const auto i : vertices_1)
                        s.insert(i);
#endif

                      new_quad->set_line_orientation(f, orientation);
                    }
#ifdef DEBUG
                  AssertDimension(
                    s.size(),
                    (reference_face_type == ReferenceCells::Quadrilateral ? 4 :
                                                                            3));
#endif
                }

              quad->clear_user_flag();
            }
        }

        typename Triangulation<3, spacedim>::DistortedCellList
          cells_with_distorted_children;

        for (unsigned int level = 0; level != triangulation.levels.size() - 1;
             ++level)
          {
            typename Triangulation<dim, spacedim>::active_hex_iterator
              hex  = triangulation.begin_active_hex(level),
              endh = triangulation.begin_active_hex(level + 1);
            typename Triangulation<dim, spacedim>::raw_hex_iterator
              next_unused_hex = triangulation.begin_raw_hex(level + 1);

            for (; hex != endh; ++hex)
              {
                if (hex->refine_flag_set() ==
                    RefinementCase<dim>::no_refinement)
                  continue;

                const auto &reference_cell_type = hex->reference_cell();

                const RefinementCase<dim> ref_case = hex->refine_flag_set();
                hex->clear_refine_flag();
                hex->set_refinement_case(ref_case);

                unsigned int n_new_lines = 0;
                unsigned int n_new_quads = 0;
                unsigned int n_new_hexes = 0;

                if (reference_cell_type == ReferenceCells::Hexahedron)
                  {
                    n_new_lines = 6;
                    n_new_quads = 12;
                    n_new_hexes = 8;
                  }
                else if (reference_cell_type == ReferenceCells::Tetrahedron)
                  {
                    n_new_lines = 1;
                    n_new_quads = 8;
                    n_new_hexes = 8;
                  }
                else
                  Assert(false, ExcNotImplemented());

                // Hexes add a single new internal vertex
                if (reference_cell_type == ReferenceCells::Hexahedron)
                  {
                    current_vertex =
                      get_next_unused_vertex(current_vertex,
                                             triangulation.vertices_used);
                    triangulation.vertices[current_vertex] =
                      hex->center(true, true);
                  }

                boost::container::small_vector<
                  typename Triangulation<dim, spacedim>::raw_line_iterator,
                  6>
                  new_lines(n_new_lines);
                for (unsigned int i = 0; i < n_new_lines; ++i)
                  {
                    new_lines[i] =
                      triangulation.faces->lines
                        .template next_free_single_object<1>(triangulation);

                    AssertIsNotUsed(new_lines[i]);
                    new_lines[i]->set_used_flag();
                    new_lines[i]->clear_user_flag();
                    new_lines[i]->clear_user_data();
                    new_lines[i]->clear_children();
                    new_lines[i]->set_boundary_id_internal(
                      numbers::internal_face_boundary_id);
                    new_lines[i]->set_manifold_id(hex->manifold_id());
                  }

                boost::container::small_vector<
                  typename Triangulation<dim, spacedim>::raw_quad_iterator,
                  12>
                  new_quads(n_new_quads);
                for (unsigned int i = 0; i < n_new_quads; ++i)
                  {
                    new_quads[i] =
                      triangulation.faces->quads
                        .template next_free_single_object<2>(triangulation);

                    auto &new_quad = new_quads[i];

                    // TODO: faces of children have the same type as the faces
                    //  of the parent
                    triangulation.faces
                      ->quad_reference_cell[new_quad->index()] =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        ReferenceCells::Quadrilateral :
                        ReferenceCells::Triangle;

                    AssertIsNotUsed(new_quad);
                    new_quad->set_used_flag();
                    new_quad->clear_user_flag();
                    new_quad->clear_user_data();
                    new_quad->clear_children();
                    new_quad->set_boundary_id_internal(
                      numbers::internal_face_boundary_id);
                    new_quad->set_manifold_id(hex->manifold_id());
                    for (const auto j : new_quads[i]->line_indices())
                      new_quad->set_line_orientation(j, true);
                  }

                // we always get 8 children per refined cell
                std::array<
                  typename Triangulation<dim, spacedim>::raw_hex_iterator,
                  8>
                  new_hexes;
                {
                  for (unsigned int i = 0; i < n_new_hexes; ++i)
                    {
                      if (i % 2 == 0)
                        next_unused_hex =
                          triangulation.levels[level + 1]->cells.next_free_hex(
                            triangulation, level + 1);
                      else
                        ++next_unused_hex;

                      new_hexes[i] = next_unused_hex;

                      auto &new_hex = new_hexes[i];

                      // TODO: children have the same type as the parent
                      triangulation.levels[new_hex->level()]
                        ->reference_cell[new_hex->index()] =
                        reference_cell_type;

                      AssertIsNotUsed(new_hex);
                      new_hex->set_used_flag();
                      new_hex->clear_user_flag();
                      new_hex->clear_user_data();
                      new_hex->clear_children();
                      new_hex->set_material_id(hex->material_id());
                      new_hex->set_manifold_id(hex->manifold_id());
                      new_hex->set_subdomain_id(hex->subdomain_id());

                      if (i % 2)
                        new_hex->set_parent(hex->index());
                      // set the face_orientation flag to true for all
                      // faces initially, as this is the default value
                      // which is true for all faces interior to the
                      // hex. later on go the other way round and
                      // reset faces that are at the boundary of the
                      // mother cube
                      //
                      // the same is true for the face_flip and
                      // face_rotation flags. however, the latter two
                      // are set to false by default as this is the
                      // standard value
                      for (const auto f : new_hex->face_indices())
                        {
                          new_hex->set_face_orientation(f, true);
                          new_hex->set_face_flip(f, false);
                          new_hex->set_face_rotation(f, false);
                        }
                    }
                  for (unsigned int i = 0; i < n_new_hexes / 2; ++i)
                    hex->set_children(2 * i, new_hexes[2 * i]->index());
                }

                {
                  // load vertex indices
                  std::array<unsigned int, 27> vertex_indices = {};

                  {
                    unsigned int k = 0;

                    for (const unsigned int i : hex->vertex_indices())
                      vertex_indices[k++] = hex->vertex_index(i);

                    for (const unsigned int i : hex->line_indices())
                      vertex_indices[k++] =
                        hex->line(i)->child(0)->vertex_index(1);

                    if (reference_cell_type == ReferenceCells::Hexahedron)
                      {
                        for (const unsigned int i : hex->face_indices())
                          vertex_indices[k++] =
                            middle_vertex_index<dim, spacedim>(hex->face(i));

                        vertex_indices[k++] = current_vertex;
                      }
                  }

                  // set up new lines
                  {
                    static constexpr std::array<std::array<unsigned int, 2>, 6>
                      new_line_vertices_hex = {{{{22, 26}},
                                                {{26, 23}},
                                                {{20, 26}},
                                                {{26, 21}},
                                                {{24, 26}},
                                                {{26, 25}}}};

                    static constexpr std::array<std::array<unsigned int, 2>, 6>
                      new_line_vertices_tet = {{{{6, 8}},
                                                {{X, X}},
                                                {{X, X}},
                                                {{X, X}},
                                                {{X, X}},
                                                {{X, X}}}};

                    const auto &new_line_vertices =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        new_line_vertices_hex :
                        new_line_vertices_tet;

                    for (unsigned int i = 0; i < new_lines.size(); ++i)
                      new_lines[i]->set_bounding_object_indices(
                        {vertex_indices[new_line_vertices[i][0]],
                         vertex_indices[new_line_vertices[i][1]]});
                  }

                  // set up new quads
                  {
                    boost::container::small_vector<
                      typename Triangulation<dim, spacedim>::raw_line_iterator,
                      30>
                      relevant_lines(0);

                    if (reference_cell_type == ReferenceCells::Hexahedron)
                      {
                        relevant_lines.resize(30);
                        for (unsigned int f = 0, k = 0; f < 6; ++f)
                          for (unsigned int c = 0; c < 4; ++c, ++k)
                            {
                              static constexpr std::
                                array<std::array<unsigned int, 2>, 4>
                                  temp = {
                                    {{{0, 1}}, {{3, 0}}, {{0, 3}}, {{3, 2}}}};

                              relevant_lines[k] =
                                hex->face(f)
                                  ->isotropic_child(
                                    GeometryInfo<dim>::
                                      standard_to_real_face_vertex(
                                        temp[c][0],
                                        hex->face_orientation(f),
                                        hex->face_flip(f),
                                        hex->face_rotation(f)))
                                  ->line(GeometryInfo<dim>::
                                           standard_to_real_face_line(
                                             temp[c][1],
                                             hex->face_orientation(f),
                                             hex->face_flip(f),
                                             hex->face_rotation(f)));
                            }

                        for (unsigned int i = 0, k = 24; i < 6; ++i, ++k)
                          relevant_lines[k] = new_lines[i];
                      }
                    else if (reference_cell_type == ReferenceCells::Tetrahedron)
                      {
                        relevant_lines.resize(13);

                        unsigned int k = 0;
                        for (unsigned int f = 0; f < 4; ++f)
                          for (unsigned int l = 0; l < 3; ++l, ++k)
                            {
                              // TODO: add comment
                              static const std::
                                array<std::array<unsigned int, 3>, 6>
                                  table = {{{{1, 0, 2}}, // 0
                                            {{0, 1, 2}},
                                            {{0, 2, 1}}, // 2
                                            {{1, 2, 0}},
                                            {{2, 1, 0}}, // 4
                                            {{2, 0, 1}}}};

                              relevant_lines[k] =
                                hex->face(f)
                                  ->child(3 /*center triangle*/)
                                  ->line(
                                    table[triangulation.levels[hex->level()]
                                            ->face_orientations
                                              [hex->index() *
                                                 GeometryInfo<
                                                   dim>::faces_per_cell +
                                               f]][l]);
                            }

                        relevant_lines[k++] = new_lines[0];

                        AssertDimension(k, 13);
                      }
                    else
                      Assert(false, ExcNotImplemented());

                    boost::container::small_vector<unsigned int, 30>
                      relevant_line_indices(relevant_lines.size());
                    for (unsigned int i = 0; i < relevant_line_indices.size();
                         ++i)
                      relevant_line_indices[i] = relevant_lines[i]->index();

                    static constexpr std::array<std::array<unsigned int, 4>, 12>
                      new_quad_lines_hex = {{{{10, 28, 16, 24}},
                                             {{28, 14, 17, 25}},
                                             {{11, 29, 24, 20}},
                                             {{29, 15, 25, 21}},
                                             {{18, 26, 0, 28}},
                                             {{26, 22, 1, 29}},
                                             {{19, 27, 28, 4}},
                                             {{27, 23, 29, 5}},
                                             {{2, 24, 8, 26}},
                                             {{24, 6, 9, 27}},
                                             {{3, 25, 26, 12}},
                                             {{25, 7, 27, 13}}}};

                    static constexpr std::array<std::array<unsigned int, 4>, 12>
                      new_quad_lines_tet = {{{{2, 3, 8, X}},
                                             {{0, 9, 5, X}},
                                             {{1, 6, 11, X}},
                                             {{4, 10, 7, X}},
                                             {{2, 12, 5, X}},
                                             {{1, 9, 12, X}},
                                             {{4, 8, 12, X}},
                                             {{6, 12, 10, X}},
                                             {{X, X, X, X}},
                                             {{X, X, X, X}},
                                             {{X, X, X, X}},
                                             {{X, X, X, X}}}};

                    static constexpr std::
                      array<std::array<std::array<unsigned int, 2>, 4>, 12>
                        table_hex = {
                          {{{{{10, 22}}, {{24, 26}}, {{10, 24}}, {{22, 26}}}},
                           {{{{24, 26}}, {{11, 23}}, {{24, 11}}, {{26, 23}}}},
                           {{{{22, 14}}, {{26, 25}}, {{22, 26}}, {{14, 25}}}},
                           {{{{26, 25}}, {{23, 15}}, {{26, 23}}, {{25, 15}}}},
                           {{{{8, 24}}, {{20, 26}}, {{8, 20}}, {{24, 26}}}},
                           {{{{20, 26}}, {{12, 25}}, {{20, 12}}, {{26, 25}}}},
                           {{{{24, 9}}, {{26, 21}}, {{24, 26}}, {{9, 21}}}},
                           {{{{26, 21}}, {{25, 13}}, {{26, 25}}, {{21, 13}}}},
                           {{{{16, 20}}, {{22, 26}}, {{16, 22}}, {{20, 26}}}},
                           {{{{22, 26}}, {{17, 21}}, {{22, 17}}, {{26, 21}}}},
                           {{{{20, 18}}, {{26, 23}}, {{20, 26}}, {{18, 23}}}},
                           {{{{26, 23}}, {{21, 19}}, {{26, 21}}, {{23, 19}}}}}};

                    static constexpr std::
                      array<std::array<std::array<unsigned int, 2>, 4>, 12>
                        table_tet = {
                          {{{{{6, 4}}, {{4, 7}}, {{7, 6}}, {{X, X}}}},
                           {{{{4, 5}}, {{5, 8}}, {{8, 4}}, {{X, X}}}},
                           {{{{5, 6}}, {{6, 9}}, {{9, 5}}, {{X, X}}}},
                           {{{{7, 8}}, {{8, 9}}, {{9, 7}}, {{X, X}}}},
                           {{{{4, 6}}, {{6, 8}}, {{8, 4}}, {{X, X}}}},
                           {{{{6, 5}}, {{5, 8}}, {{8, 6}}, {{X, X}}}},
                           {{{{8, 7}}, {{7, 6}}, {{6, 8}}, {{X, X}}}},
                           {{{{9, 6}}, {{6, 8}}, {{8, 9}}, {{X, X}}}},
                           {{{{X, X}}, {{X, X}}, {{X, X}}, {{X, X}}}},
                           {{{{X, X}}, {{X, X}}, {{X, X}}, {{X, X}}}},
                           {{{{X, X}}, {{X, X}}, {{X, X}}, {{X, X}}}},
                           {{{{X, X}}, {{X, X}}, {{X, X}}, {{X, X}}}}}};

                    const auto &new_quad_lines =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        new_quad_lines_hex :
                        new_quad_lines_tet;

                    const auto &table =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        table_hex :
                        table_tet;

                    for (unsigned int q = 0; q < new_quads.size(); ++q)
                      {
                        for (unsigned int l = 0; l < 3; ++l)
                          {
                            std::array<unsigned int, 2> vertices_0, vertices_1;

                            for (unsigned int v = 0; v < 2; ++v)
                              vertices_0[v] =
                                relevant_lines[new_quad_lines[q][l]]
                                  ->vertex_index(v);

                            for (unsigned int v = 0; v < 2; ++v)
                              vertices_1[v] = vertex_indices[table[q][l][v]];
                          }
                      }

                    for (unsigned int q = 0; q < new_quads.size(); ++q)
                      {
                        auto &new_quad = new_quads[q];

                        if (new_quad->n_lines() == 3)
                          new_quad->set_bounding_object_indices(
                            {relevant_line_indices[new_quad_lines[q][0]],
                             relevant_line_indices[new_quad_lines[q][1]],
                             relevant_line_indices[new_quad_lines[q][2]]});
                        else if (new_quad->n_lines() == 4)
                          new_quad->set_bounding_object_indices(
                            {relevant_line_indices[new_quad_lines[q][0]],
                             relevant_line_indices[new_quad_lines[q][1]],
                             relevant_line_indices[new_quad_lines[q][2]],
                             relevant_line_indices[new_quad_lines[q][3]]});
                        else
                          Assert(false, ExcNotImplemented());

                        for (const auto l : new_quad->line_indices())
                          {
                            std::array<unsigned int, 2> vertices_0, vertices_1;

                            for (unsigned int v = 0; v < 2; ++v)
                              vertices_0[v] =
                                relevant_lines[new_quad_lines[q][l]]
                                  ->vertex_index(v);

                            for (unsigned int v = 0; v < 2; ++v)
                              vertices_1[v] = vertex_indices[table[q][l][v]];

                            const auto orientation =
                              ReferenceCells::Line.compute_orientation(
                                vertices_0, vertices_1);

                            new_quad->set_line_orientation(l, orientation);
                          }
                      }
                  }

                  // set up new hex
                  {
                    std::array<int, 36> quad_indices;

                    if (reference_cell_type == ReferenceCells::Hexahedron)
                      {
                        for (unsigned int i = 0; i < new_quads.size(); ++i)
                          quad_indices[i] = new_quads[i]->index();

                        for (unsigned int f = 0, k = new_quads.size(); f < 6;
                             ++f)
                          for (unsigned int c = 0; c < 4; ++c, ++k)
                            quad_indices[k] =
                              hex->face(f)->isotropic_child_index(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  c,
                                  hex->face_orientation(f),
                                  hex->face_flip(f),
                                  hex->face_rotation(f)));
                      }
                    else if (reference_cell_type == ReferenceCells::Tetrahedron)
                      {
                        for (unsigned int i = 0; i < new_quads.size(); ++i)
                          quad_indices[i] = new_quads[i]->index();

                        for (unsigned int f = 0, k = new_quads.size(); f < 4;
                             ++f)
                          for (unsigned int c = 0; c < 4; ++c, ++k)
                            {
                              quad_indices[k] = hex->face(f)->child_index(
                                (c == 3) ?
                                  3 :
                                  reference_cell_type
                                    .standard_to_real_face_vertex(
                                      c,
                                      f,
                                      triangulation.levels[hex->level()]
                                        ->face_orientations
                                          [hex->index() *
                                             GeometryInfo<dim>::faces_per_cell +
                                           f]));
                            }
                      }
                    else
                      {
                        Assert(false, ExcNotImplemented());
                      }

                    static constexpr std::array<std::array<unsigned int, 6>, 8>
                      cell_quads_hex = {{
                        {{12, 0, 20, 4, 28, 8}},  // bottom children
                        {{0, 16, 22, 6, 29, 9}},  //
                        {{13, 1, 4, 24, 30, 10}}, //
                        {{1, 17, 6, 26, 31, 11}}, //
                        {{14, 2, 21, 5, 8, 32}},  // top children
                        {{2, 18, 23, 7, 9, 33}},  //
                        {{15, 3, 5, 25, 10, 34}}, //
                        {{3, 19, 7, 27, 11, 35}}  //
                      }};

                    static constexpr std::array<std::array<unsigned int, 6>, 8>
                      cell_quads_tet{{{{8, 13, 16, 0, X, X}},
                                      {{9, 12, 1, 21, X, X}},
                                      {{10, 2, 17, 20, X, X}},
                                      {{3, 14, 18, 22, X, X}},
                                      {{11, 1, 4, 5, X, X}},
                                      {{15, 0, 4, 6, X, X}},
                                      {{19, 7, 6, 3, X, X}},
                                      {{23, 5, 2, 7, X, X}}}};

                    static constexpr std::
                      array<std::array<std::array<unsigned int, 4>, 6>, 8>
                        cell_face_vertices_hex{{{{{{0, 8, 16, 20}},
                                                  {{10, 24, 22, 26}},
                                                  {{0, 16, 10, 22}},
                                                  {{8, 20, 24, 26}},
                                                  {{0, 10, 8, 24}},
                                                  {{16, 22, 20, 26}}}},
                                                {{{{10, 24, 22, 26}},
                                                  {{1, 9, 17, 21}},
                                                  {{10, 22, 1, 17}},
                                                  {{24, 26, 9, 21}},
                                                  {{10, 1, 24, 9}},
                                                  {{22, 17, 26, 21}}}},
                                                {{{{8, 2, 20, 18}},
                                                  {{24, 11, 26, 23}},
                                                  {{8, 20, 24, 26}},
                                                  {{2, 18, 11, 23}},
                                                  {{8, 24, 2, 11}},
                                                  {{20, 26, 18, 23}}}},
                                                {{{{24, 11, 26, 23}},
                                                  {{9, 3, 21, 19}},
                                                  {{24, 26, 9, 21}},
                                                  {{11, 23, 3, 19}},
                                                  {{24, 9, 11, 3}},
                                                  {{26, 21, 23, 19}}}},
                                                {{{{16, 20, 4, 12}},
                                                  {{22, 26, 14, 25}},
                                                  {{16, 4, 22, 14}},
                                                  {{20, 12, 26, 25}},
                                                  {{16, 22, 20, 26}},
                                                  {{4, 14, 12, 25}}}},
                                                {{{{22, 26, 14, 25}},
                                                  {{17, 21, 5, 13}},
                                                  {{22, 14, 17, 5}},
                                                  {{26, 25, 21, 13}},
                                                  {{22, 17, 26, 21}},
                                                  {{14, 5, 25, 13}}}},
                                                {{{{20, 18, 12, 6}},
                                                  {{26, 23, 25, 15}},
                                                  {{20, 12, 26, 25}},
                                                  {{18, 6, 23, 15}},
                                                  {{20, 26, 18, 23}},
                                                  {{12, 25, 6, 15}}}},
                                                {{{{26, 23, 25, 15}},
                                                  {{21, 19, 13, 7}},
                                                  {{26, 25, 21, 13}},
                                                  {{23, 15, 19, 7}},
                                                  {{26, 21, 23, 19}},
                                                  {{25, 13, 15, 7}}}}}};

                    static constexpr std::
                      array<std::array<std::array<unsigned int, 4>, 6>, 8>
                        cell_face_vertices_tet{{{{{{0, 4, 6, X}},
                                                  {{4, 0, 7, X}},
                                                  {{0, 6, 7, X}},
                                                  {{6, 4, 7, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{4, 1, 5, X}},
                                                  {{1, 4, 8, X}},
                                                  {{4, 5, 8, X}},
                                                  {{5, 1, 8, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{6, 5, 2, X}},
                                                  {{5, 6, 9, X}},
                                                  {{6, 2, 9, X}},
                                                  {{2, 5, 9, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{7, 8, 9, X}},
                                                  {{8, 7, 3, X}},
                                                  {{7, 9, 3, X}},
                                                  {{9, 8, 3, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{4, 5, 6, X}},
                                                  {{5, 4, 8, X}},
                                                  {{4, 6, 8, X}},
                                                  {{6, 5, 8, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{4, 7, 8, X}},
                                                  {{7, 4, 6, X}},
                                                  {{4, 8, 6, X}},
                                                  {{8, 7, 6, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{6, 9, 7, X}},
                                                  {{9, 6, 8, X}},
                                                  {{6, 7, 8, X}},
                                                  {{7, 9, 8, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}},
                                                {{{{5, 8, 9, X}},
                                                  {{8, 5, 6, X}},
                                                  {{5, 9, 6, X}},
                                                  {{9, 8, 6, X}},
                                                  {{X, X, X, X}},
                                                  {{X, X, X, X}}}}}};

                    const auto &cell_quads =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        cell_quads_hex :
                        cell_quads_tet;

                    const auto &cell_face_vertices =
                      (reference_cell_type == ReferenceCells::Hexahedron) ?
                        cell_face_vertices_hex :
                        cell_face_vertices_tet;

                    for (unsigned int c = 0;
                         c < GeometryInfo<dim>::max_children_per_cell;
                         ++c)
                      {
                        auto &new_hex = new_hexes[c];

                        if (new_hex->n_faces() == 4)
                          new_hex->set_bounding_object_indices(
                            {quad_indices[cell_quads[c][0]],
                             quad_indices[cell_quads[c][1]],
                             quad_indices[cell_quads[c][2]],
                             quad_indices[cell_quads[c][3]]});
                        else if (new_hex->n_faces() == 6)
                          new_hex->set_bounding_object_indices(
                            {quad_indices[cell_quads[c][0]],
                             quad_indices[cell_quads[c][1]],
                             quad_indices[cell_quads[c][2]],
                             quad_indices[cell_quads[c][3]],
                             quad_indices[cell_quads[c][4]],
                             quad_indices[cell_quads[c][5]]});
                        else
                          Assert(false, ExcNotImplemented());

                        for (const auto f : new_hex->face_indices())
                          {
                            std::array<unsigned int, 4> vertices_0, vertices_1;

                            const auto &face = new_hex->face(f);

                            for (const auto i : face->vertex_indices())
                              vertices_0[i] = face->vertex_index(i);

                            for (const auto i : face->vertex_indices())
                              vertices_1[i] =
                                vertex_indices[cell_face_vertices[c][f][i]];

                            const auto orientation =
                              face->reference_cell().compute_orientation(
                                vertices_1, vertices_0);

                            new_hex->set_face_orientation(
                              f, Utilities::get_bit(orientation, 0));
                            new_hex->set_face_flip(
                              f, Utilities::get_bit(orientation, 2));
                            new_hex->set_face_rotation(
                              f, Utilities::get_bit(orientation, 1));
                          }
                      }
                  }
                }

                if (check_for_distorted_cells &&
                    has_distorted_children<dim, spacedim>(hex))
                  cells_with_distorted_children.distorted_cells.push_back(hex);

                triangulation.signals.post_refinement_on_cell(hex);
              }
          }

        triangulation.faces->quads.clear_user_data();

        return cells_with_distorted_children;
      }

      /**
       * A function that performs the refinement of a triangulation in
       * 3d.
       */
      template <int spacedim>
      static typename Triangulation<3, spacedim>::DistortedCellList
      execute_refinement(Triangulation<3, spacedim> &triangulation,
                         const bool                  check_for_distorted_cells)
      {
        const unsigned int dim = 3;

        {
          bool flag_isotropic_mesh = true;
          typename Triangulation<dim, spacedim>::raw_cell_iterator
            cell = triangulation.begin(),
            endc = triangulation.end();
          for (; cell != endc; ++cell)
            if (cell->used())
              if (triangulation.get_anisotropic_refinement_flag() ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_x ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_y ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_z ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_xy ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_xz ||
                  cell->refine_flag_set() == RefinementCase<dim>::cut_yz)
                {
                  flag_isotropic_mesh = false;
                  break;
                }

          if (flag_isotropic_mesh)
            return execute_refinement_isotropic(triangulation,
                                                check_for_distorted_cells);
        }

        // this function probably also works for spacedim>3 but it
        // isn't tested. it will probably be necessary to pull new
        // vertices onto the manifold just as we do for the other
        // functions above.
        Assert(spacedim == 3, ExcNotImplemented());

        // Check whether a new level is needed. We have to check for
        // this on the highest level only
        for (const auto &cell : triangulation.active_cell_iterators_on_level(
               triangulation.levels.size() - 1))
          if (cell->refine_flag_set())
            {
              triangulation.levels.push_back(
                std::make_unique<
                  internal::TriangulationImplementation::TriaLevel>(dim));
              break;
            }


        // first clear user flags for quads and lines; we're going to
        // use them to flag which lines and quads need refinement
        triangulation.faces->quads.clear_user_data();

        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          line->clear_user_flag();
        for (typename Triangulation<dim, spacedim>::quad_iterator quad =
               triangulation.begin_quad();
             quad != triangulation.end_quad();
             ++quad)
          quad->clear_user_flag();

        // create an array of face refine cases. User indices of faces
        // will be set to values corresponding with indices in this
        // array.
        const RefinementCase<dim - 1> face_refinement_cases[4] = {
          RefinementCase<dim - 1>::no_refinement,
          RefinementCase<dim - 1>::cut_x,
          RefinementCase<dim - 1>::cut_y,
          RefinementCase<dim - 1>::cut_xy};

        // check how much space is needed on every level. We need not
        // check the highest level since either
        // - on the highest level no cells are flagged for refinement
        // - there are, but prepare_refinement added another empty
        //   level which then is the highest level

        // variables to hold the number of newly to be created
        // vertices, lines and quads. as these are stored globally,
        // declare them outside the loop over al levels. we need lines
        // and quads in pairs for refinement of old ones and lines and
        // quads, that can be stored as single ones, as they are newly
        // created in the inside of an existing cell
        unsigned int needed_vertices     = 0;
        unsigned int needed_lines_single = 0;
        unsigned int needed_quads_single = 0;
        unsigned int needed_lines_pair   = 0;
        unsigned int needed_quads_pair   = 0;
        for (int level = triangulation.levels.size() - 2; level >= 0; --level)
          {
            // count number of flagged cells on this level and compute
            // how many new vertices and new lines will be needed
            unsigned int new_cells = 0;

            for (const auto &acell :
                 triangulation.active_cell_iterators_on_level(level))
              if (acell->refine_flag_set())
                {
                  RefinementCase<dim> ref_case = acell->refine_flag_set();

                  // now for interior vertices, lines and quads, which
                  // are needed in any case
                  if (ref_case == RefinementCase<dim>::cut_x ||
                      ref_case == RefinementCase<dim>::cut_y ||
                      ref_case == RefinementCase<dim>::cut_z)
                    {
                      ++needed_quads_single;
                      new_cells += 2;
                      triangulation.anisotropic_refinement = true;
                    }
                  else if (ref_case == RefinementCase<dim>::cut_xy ||
                           ref_case == RefinementCase<dim>::cut_xz ||
                           ref_case == RefinementCase<dim>::cut_yz)
                    {
                      ++needed_lines_single;
                      needed_quads_single += 4;
                      new_cells += 4;
                      triangulation.anisotropic_refinement = true;
                    }
                  else if (ref_case == RefinementCase<dim>::cut_xyz)
                    {
                      ++needed_vertices;
                      needed_lines_single += 6;
                      needed_quads_single += 12;
                      new_cells += 8;
                    }
                  else
                    {
                      // we should never get here
                      Assert(false, ExcInternalError());
                    }

                  // mark all faces for refinement; checking locally
                  // if and how the neighbor would like to refine
                  // these is difficult so we only flag them and after
                  // visiting all cells, we decide which faces need
                  // which refinement;
                  for (const unsigned int face :
                       GeometryInfo<dim>::face_indices())
                    {
                      typename Triangulation<dim, spacedim>::face_iterator
                        aface = acell->face(face);
                      // get the RefineCase this faces has for the
                      // given RefineCase of the cell
                      RefinementCase<dim - 1> face_ref_case =
                        GeometryInfo<dim>::face_refinement_case(
                          ref_case,
                          face,
                          acell->face_orientation(face),
                          acell->face_flip(face),
                          acell->face_rotation(face));
                      // only do something, if this face has to be
                      // refined
                      if (face_ref_case)
                        {
                          if (face_ref_case ==
                              RefinementCase<dim - 1>::isotropic_refinement)
                            {
                              if (aface->n_active_descendants() < 4)
                                // we use user_flags to denote needed
                                // isotropic refinement
                                aface->set_user_flag();
                            }
                          else if (aface->refinement_case() != face_ref_case)
                            // we use user_indices to denote needed
                            // anisotropic refinement. note, that we
                            // can have at most one anisotropic
                            // refinement case for this face, as
                            // otherwise prepare_refinement() would
                            // have changed one of the cells to yield
                            // isotropic refinement at this
                            // face. therefore we set the user_index
                            // uniquely
                            {
                              Assert(aface->refinement_case() ==
                                         RefinementCase<
                                           dim - 1>::isotropic_refinement ||
                                       aface->refinement_case() ==
                                         RefinementCase<dim - 1>::no_refinement,
                                     ExcInternalError());
                              aface->set_user_index(face_ref_case);
                            }
                        }
                    } // for all faces

                  // flag all lines, that have to be refined
                  for (unsigned int line = 0;
                       line < GeometryInfo<dim>::lines_per_cell;
                       ++line)
                    if (GeometryInfo<dim>::line_refinement_case(ref_case,
                                                                line) &&
                        !acell->line(line)->has_children())
                      acell->line(line)->set_user_flag();

                } // if refine_flag set and for all cells on this level


            // count number of used cells on the next higher level
            const unsigned int used_cells =
              std::count(triangulation.levels[level + 1]->cells.used.begin(),
                         triangulation.levels[level + 1]->cells.used.end(),
                         true);


            // reserve space for the used_cells cells already existing
            // on the next higher level as well as for the
            // 8*flagged_cells that will be created on that level
            reserve_space(*triangulation.levels[level + 1],
                          used_cells + new_cells,
                          3,
                          spacedim);
            // reserve space for 8*flagged_cells new hexes on the next
            // higher level
            reserve_space(triangulation.levels[level + 1]->cells, new_cells);
          } // for all levels
        // now count the quads and lines which were flagged for
        // refinement
        for (typename Triangulation<dim, spacedim>::quad_iterator quad =
               triangulation.begin_quad();
             quad != triangulation.end_quad();
             ++quad)
          {
            if (quad->user_flag_set())
              {
                // isotropic refinement: 1 interior vertex, 4 quads
                // and 4 interior lines. we store the interior lines
                // in pairs in case the face is already or will be
                // refined anisotropically
                needed_quads_pair += 4;
                needed_lines_pair += 4;
                needed_vertices += 1;
              }
            if (quad->user_index())
              {
                // anisotropic refinement: 1 interior
                // line and two quads
                needed_quads_pair += 2;
                needed_lines_single += 1;
                // there is a kind of complicated situation here which
                // requires our attention. if the quad is refined
                // isotropcally, two of the interior lines will get a
                // new mother line - the interior line of our
                // anisotropically refined quad. if those two lines
                // are not consecutive, we cannot do so and have to
                // replace them by two lines that are consecutive. we
                // try to avoid that situation, but it may happen
                // nevertheless through repeated refinement and
                // coarsening. thus we have to check here, as we will
                // need some additional space to store those new lines
                // in case we need them...
                if (quad->has_children())
                  {
                    Assert(quad->refinement_case() ==
                             RefinementCase<dim - 1>::isotropic_refinement,
                           ExcInternalError());
                    if ((face_refinement_cases[quad->user_index()] ==
                           RefinementCase<dim - 1>::cut_x &&
                         (quad->child(0)->line_index(1) + 1 !=
                          quad->child(2)->line_index(1))) ||
                        (face_refinement_cases[quad->user_index()] ==
                           RefinementCase<dim - 1>::cut_y &&
                         (quad->child(0)->line_index(3) + 1 !=
                          quad->child(1)->line_index(3))))
                      needed_lines_pair += 2;
                  }
              }
          }

        for (typename Triangulation<dim, spacedim>::line_iterator line =
               triangulation.begin_line();
             line != triangulation.end_line();
             ++line)
          if (line->user_flag_set())
            {
              needed_lines_pair += 2;
              needed_vertices += 1;
            }

        // reserve space for needed_lines new lines stored in pairs
        reserve_space(triangulation.faces->lines,
                      needed_lines_pair,
                      needed_lines_single);
        // reserve space for needed_quads new quads stored in pairs
        reserve_space(*triangulation.faces,
                      needed_quads_pair,
                      needed_quads_single);
        reserve_space(triangulation.faces->quads,
                      needed_quads_pair,
                      needed_quads_single);


        // add to needed vertices how many vertices are already in use
        needed_vertices += std::count(triangulation.vertices_used.begin(),
                                      triangulation.vertices_used.end(),
                                      true);
        // if we need more vertices: create them, if not: leave the
        // array as is, since shrinking is not really possible because
        // some of the vertices at the end may be in use
        if (needed_vertices > triangulation.vertices.size())
          {
            triangulation.vertices.resize(needed_vertices, Point<spacedim>());
            triangulation.vertices_used.resize(needed_vertices, false);
          }


          //-----------------------------------------
          // Before we start with the actual refinement, we do some
          // sanity checks if in debug mode. especially, we try to catch
          // the notorious problem with lines being twice refined,
          // i.e. there are cells adjacent at one line ("around the
          // edge", but not at a face), with two cells differing by more
          // than one refinement level
          //
          // this check is very simple to implement here, since we have
          // all lines flagged if they shall be refined
#ifdef DEBUG
        for (const auto &cell : triangulation.active_cell_iterators())
          if (!cell->refine_flag_set())
            for (unsigned int line = 0;
                 line < GeometryInfo<dim>::lines_per_cell;
                 ++line)
              if (cell->line(line)->has_children())
                for (unsigned int c = 0; c < 2; ++c)
                  Assert(cell->line(line)->child(c)->user_flag_set() == false,
                         ExcInternalError());
#endif

        //-----------------------------------------
        // Do refinement on every level
        //
        // To make life a bit easier, we first refine those lines and
        // quads that were flagged for refinement and then compose the
        // newly to be created cells.
        //
        // index of next unused vertex
        unsigned int next_unused_vertex = 0;

        // first for lines
        {
          // only active objects can be refined further
          typename Triangulation<dim, spacedim>::active_line_iterator
            line = triangulation.begin_active_line(),
            endl = triangulation.end_line();
          typename Triangulation<dim, spacedim>::raw_line_iterator
            next_unused_line = triangulation.begin_raw_line();

          for (; line != endl; ++line)
            if (line->user_flag_set())
              {
                // this line needs to be refined

                // find the next unused vertex and set it
                // appropriately
                while (triangulation.vertices_used[next_unused_vertex] == true)
                  ++next_unused_vertex;
                Assert(
                  next_unused_vertex < triangulation.vertices.size(),
                  ExcMessage(
                    "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));
                triangulation.vertices_used[next_unused_vertex] = true;

                triangulation.vertices[next_unused_vertex] = line->center(true);

                // now that we created the right point, make up the
                // two child lines (++ takes care of the end of the
                // vector)
                next_unused_line =
                  triangulation.faces->lines.template next_free_pair_object<1>(
                    triangulation);
                Assert(next_unused_line.state() == IteratorState::valid,
                       ExcInternalError());

                // now we found two consecutive unused lines, such
                // that the children of a line will be consecutive.
                // then set the child pointer of the present line
                line->set_children(0, next_unused_line->index());

                // set the two new lines
                const typename Triangulation<dim, spacedim>::raw_line_iterator
                  children[2] = {next_unused_line, ++next_unused_line};

                // some tests; if any of the iterators should be
                // invalid, then already dereferencing will fail
                AssertIsNotUsed(children[0]);
                AssertIsNotUsed(children[1]);

                children[0]->set_bounding_object_indices(
                  {line->vertex_index(0), next_unused_vertex});
                children[1]->set_bounding_object_indices(
                  {next_unused_vertex, line->vertex_index(1)});

                children[0]->set_used_flag();
                children[1]->set_used_flag();
                children[0]->clear_children();
                children[1]->clear_children();
                children[0]->clear_user_data();
                children[1]->clear_user_data();
                children[0]->clear_user_flag();
                children[1]->clear_user_flag();

                children[0]->set_boundary_id_internal(line->boundary_id());
                children[1]->set_boundary_id_internal(line->boundary_id());

                children[0]->set_manifold_id(line->manifold_id());
                children[1]->set_manifold_id(line->manifold_id());

                // finally clear flag
                // indicating the need
                // for refinement
                line->clear_user_flag();
              }
        }


        //-------------------------------------
        // now refine marked quads
        //-------------------------------------

        // here we encounter several cases:

        // a) the quad is unrefined and shall be refined isotropically

        // b) the quad is unrefined and shall be refined
        // anisotropically

        // c) the quad is unrefined and shall be refined both
        // anisotropically and isotropically (this is reduced to case
        // b) and then case b) for the children again)

        // d) the quad is refined anisotropically and shall be refined
        // isotropically (this is reduced to case b) for the
        // anisotropic children)

        // e) the quad is refined isotropically and shall be refined
        // anisotropically (this is transformed to case c), however we
        // might have to renumber/rename children...)

        // we need a loop in cases c) and d), as the anisotropic
        // children migt have a lower index than the mother quad
        for (unsigned int loop = 0; loop < 2; ++loop)
          {
            // usually, only active objects can be refined
            // further. however, in cases d) and e) that is not true,
            // so we have to use 'normal' iterators here
            typename Triangulation<dim, spacedim>::quad_iterator
              quad = triangulation.begin_quad(),
              endq = triangulation.end_quad();
            typename Triangulation<dim, spacedim>::raw_line_iterator
              next_unused_line = triangulation.begin_raw_line();
            typename Triangulation<dim, spacedim>::raw_quad_iterator
              next_unused_quad = triangulation.begin_raw_quad();

            for (; quad != endq; ++quad)
              {
                if (quad->user_index())
                  {
                    RefinementCase<dim - 1> aniso_quad_ref_case =
                      face_refinement_cases[quad->user_index()];
                    // there is one unlikely event here, where we
                    // already have refind the face: if the face was
                    // refined anisotropically and we want to refine
                    // it isotropically, both children are flagged for
                    // anisotropic refinement. however, if those
                    // children were already flagged for anisotropic
                    // refinement, they might already be processed and
                    // refined.
                    if (aniso_quad_ref_case == quad->refinement_case())
                      continue;

                    Assert(quad->refinement_case() ==
                               RefinementCase<dim - 1>::cut_xy ||
                             quad->refinement_case() ==
                               RefinementCase<dim - 1>::no_refinement,
                           ExcInternalError());

                    // this quad needs to be refined anisotropically
                    Assert(quad->user_index() ==
                               RefinementCase<dim - 1>::cut_x ||
                             quad->user_index() ==
                               RefinementCase<dim - 1>::cut_y,
                           ExcInternalError());

                    // make the new line interior to the quad
                    typename Triangulation<dim, spacedim>::raw_line_iterator
                      new_line;

                    new_line =
                      triangulation.faces->lines
                        .template next_free_single_object<1>(triangulation);
                    AssertIsNotUsed(new_line);

                    // first collect the
                    // indices of the vertices:
                    // *--1--*
                    // |  |  |
                    // |  |  |    cut_x
                    // |  |  |
                    // *--0--*
                    //
                    // *-----*
                    // |     |
                    // 0-----1    cut_y
                    // |     |
                    // *-----*
                    unsigned int vertex_indices[2];
                    if (aniso_quad_ref_case == RefinementCase<dim - 1>::cut_x)
                      {
                        vertex_indices[0] =
                          quad->line(2)->child(0)->vertex_index(1);
                        vertex_indices[1] =
                          quad->line(3)->child(0)->vertex_index(1);
                      }
                    else
                      {
                        vertex_indices[0] =
                          quad->line(0)->child(0)->vertex_index(1);
                        vertex_indices[1] =
                          quad->line(1)->child(0)->vertex_index(1);
                      }

                    new_line->set_bounding_object_indices(
                      {vertex_indices[0], vertex_indices[1]});
                    new_line->set_used_flag();
                    new_line->clear_user_flag();
                    new_line->clear_user_data();
                    new_line->clear_children();
                    new_line->set_boundary_id_internal(quad->boundary_id());
                    new_line->set_manifold_id(quad->manifold_id());

                    // child 0 and 1 of a line are switched if the
                    // line orientation is false. set up a miniature
                    // table, indicating which child to take for line
                    // orientations false and true. first index: child
                    // index in standard orientation, second index:
                    // line orientation
                    const unsigned int index[2][2] = {
                      {1, 0},  // child 0, line_orientation=false and true
                      {0, 1}}; // child 1, line_orientation=false and true

                    // find some space (consecutive) for the two newly
                    // to be created quads.
                    typename Triangulation<dim, spacedim>::raw_quad_iterator
                      new_quads[2];

                    next_unused_quad =
                      triangulation.faces->quads
                        .template next_free_pair_object<2>(triangulation);
                    new_quads[0] = next_unused_quad;
                    AssertIsNotUsed(new_quads[0]);

                    ++next_unused_quad;
                    new_quads[1] = next_unused_quad;
                    AssertIsNotUsed(new_quads[1]);

                    if (aniso_quad_ref_case == RefinementCase<dim - 1>::cut_x)
                      {
                        new_quads[0]->set_bounding_object_indices(
                          {static_cast<int>(quad->line_index(0)),
                           new_line->index(),
                           quad->line(2)
                             ->child(index[0][quad->line_orientation(2)])
                             ->index(),
                           quad->line(3)
                             ->child(index[0][quad->line_orientation(3)])
                             ->index()});
                        new_quads[1]->set_bounding_object_indices(
                          {new_line->index(),
                           static_cast<int>(quad->line_index(1)),
                           quad->line(2)
                             ->child(index[1][quad->line_orientation(2)])
                             ->index(),
                           quad->line(3)
                             ->child(index[1][quad->line_orientation(3)])
                             ->index()});
                      }
                    else
                      {
                        new_quads[0]->set_bounding_object_indices(
                          {quad->line(0)
                             ->child(index[0][quad->line_orientation(0)])
                             ->index(),
                           quad->line(1)
                             ->child(index[0][quad->line_orientation(1)])
                             ->index(),
                           static_cast<int>(quad->line_index(2)),
                           new_line->index()});
                        new_quads[1]->set_bounding_object_indices(
                          {quad->line(0)
                             ->child(index[1][quad->line_orientation(0)])
                             ->index(),
                           quad->line(1)
                             ->child(index[1][quad->line_orientation(1)])
                             ->index(),
                           new_line->index(),
                           static_cast<int>(quad->line_index(3))});
                      }

                    for (const auto &new_quad : new_quads)
                      {
                        new_quad->set_used_flag();
                        new_quad->clear_user_flag();
                        new_quad->clear_user_data();
                        new_quad->clear_children();
                        new_quad->set_boundary_id_internal(quad->boundary_id());
                        new_quad->set_manifold_id(quad->manifold_id());
                        // set all line orientations to true, change
                        // this after the loop, as we have to consider
                        // different lines for each child
                        for (unsigned int j = 0;
                             j < GeometryInfo<dim>::lines_per_face;
                             ++j)
                          new_quad->set_line_orientation(j, true);
                      }
                    // now set the line orientation of children of
                    // outer lines correctly, the lines in the
                    // interior of the refined quad are automatically
                    // oriented conforming to the standard
                    new_quads[0]->set_line_orientation(
                      0, quad->line_orientation(0));
                    new_quads[0]->set_line_orientation(
                      2, quad->line_orientation(2));
                    new_quads[1]->set_line_orientation(
                      1, quad->line_orientation(1));
                    new_quads[1]->set_line_orientation(
                      3, quad->line_orientation(3));
                    if (aniso_quad_ref_case == RefinementCase<dim - 1>::cut_x)
                      {
                        new_quads[0]->set_line_orientation(
                          3, quad->line_orientation(3));
                        new_quads[1]->set_line_orientation(
                          2, quad->line_orientation(2));
                      }
                    else
                      {
                        new_quads[0]->set_line_orientation(
                          1, quad->line_orientation(1));
                        new_quads[1]->set_line_orientation(
                          0, quad->line_orientation(0));
                      }

                    // test, whether this face is refined
                    // isotropically already. if so, set the correct
                    // children pointers.
                    if (quad->refinement_case() ==
                        RefinementCase<dim - 1>::cut_xy)
                      {
                        // we will put a new refinemnt level of
                        // anisotropic refinement between the
                        // unrefined and isotropically refined quad
                        // ending up with the same fine quads but
                        // introducing anisotropically refined ones as
                        // children of the unrefined quad and mother
                        // cells of the original fine ones.

                        // this process includes the creation of a new
                        // middle line which we will assign as the
                        // mother line of two of the existing inner
                        // lines. If those inner lines are not
                        // consecutive in memory, we won't find them
                        // later on, so we have to create new ones
                        // instead and replace all occurrences of the
                        // old ones with those new ones. As this is
                        // kind of ugly, we hope we don't have to do
                        // it often...
                        typename Triangulation<dim, spacedim>::line_iterator
                          old_child[2];
                        if (aniso_quad_ref_case ==
                            RefinementCase<dim - 1>::cut_x)
                          {
                            old_child[0] = quad->child(0)->line(1);
                            old_child[1] = quad->child(2)->line(1);
                          }
                        else
                          {
                            Assert(aniso_quad_ref_case ==
                                     RefinementCase<dim - 1>::cut_y,
                                   ExcInternalError());

                            old_child[0] = quad->child(0)->line(3);
                            old_child[1] = quad->child(1)->line(3);
                          }

                        if (old_child[0]->index() + 1 != old_child[1]->index())
                          {
                            // this is exactly the ugly case we taked
                            // about. so, no coimplaining, lets get
                            // two new lines and copy all info
                            typename Triangulation<dim,
                                                   spacedim>::raw_line_iterator
                              new_child[2];

                            new_child[0] = new_child[1] =
                              triangulation.faces->lines
                                .template next_free_pair_object<1>(
                                  triangulation);
                            ++new_child[1];

                            new_child[0]->set_used_flag();
                            new_child[1]->set_used_flag();

                            const int old_index_0 = old_child[0]->index(),
                                      old_index_1 = old_child[1]->index(),
                                      new_index_0 = new_child[0]->index(),
                                      new_index_1 = new_child[1]->index();

                            // loop over all quads and replace the old
                            // lines
                            for (unsigned int q = 0;
                                 q < triangulation.faces->quads.n_objects();
                                 ++q)
                              for (unsigned int l = 0;
                                   l < GeometryInfo<dim>::lines_per_face;
                                   ++l)
                                {
                                  const int this_index =
                                    triangulation.faces->quads
                                      .get_bounding_object_indices(q)[l];
                                  if (this_index == old_index_0)
                                    triangulation.faces->quads
                                      .get_bounding_object_indices(q)[l] =
                                      new_index_0;
                                  else if (this_index == old_index_1)
                                    triangulation.faces->quads
                                      .get_bounding_object_indices(q)[l] =
                                      new_index_1;
                                }
                            // now we have to copy all information of
                            // the two lines
                            for (unsigned int i = 0; i < 2; ++i)
                              {
                                Assert(!old_child[i]->has_children(),
                                       ExcInternalError());

                                new_child[i]->set_bounding_object_indices(
                                  {old_child[i]->vertex_index(0),
                                   old_child[i]->vertex_index(1)});
                                new_child[i]->set_boundary_id_internal(
                                  old_child[i]->boundary_id());
                                new_child[i]->set_manifold_id(
                                  old_child[i]->manifold_id());
                                new_child[i]->set_user_index(
                                  old_child[i]->user_index());
                                if (old_child[i]->user_flag_set())
                                  new_child[i]->set_user_flag();
                                else
                                  new_child[i]->clear_user_flag();

                                new_child[i]->clear_children();

                                old_child[i]->clear_user_flag();
                                old_child[i]->clear_user_index();
                                old_child[i]->clear_used_flag();
                              }
                          }
                        // now that we cared about the lines, go on
                        // with the quads themselves, where we might
                        // encounter similar situations...
                        if (aniso_quad_ref_case ==
                            RefinementCase<dim - 1>::cut_x)
                          {
                            new_line->set_children(
                              0, quad->child(0)->line_index(1));
                            Assert(new_line->child(1) ==
                                     quad->child(2)->line(1),
                                   ExcInternalError());
                            // now evereything is quite
                            // complicated. we have the children
                            // numbered according to
                            //
                            // *---*---*
                            // |n+2|n+3|
                            // *---*---*
                            // | n |n+1|
                            // *---*---*
                            //
                            // from the original isotropic
                            // refinement. we have to reorder them as
                            //
                            // *---*---*
                            // |n+1|n+3|
                            // *---*---*
                            // | n |n+2|
                            // *---*---*
                            //
                            // such that n and n+1 are consecutive
                            // children of m and n+2 and n+3 are
                            // consecutive children of m+1, where m
                            // and m+1 are given as in
                            //
                            // *---*---*
                            // |   |   |
                            // | m |m+1|
                            // |   |   |
                            // *---*---*
                            //
                            // this is a bit ugly, of course: loop
                            // over all cells on all levels and look
                            // for faces n+1 (switch_1) and n+2
                            // (switch_2).
                            const typename Triangulation<dim, spacedim>::
                              quad_iterator switch_1 = quad->child(1),
                                            switch_2 = quad->child(2);
                            const int switch_1_index = switch_1->index();
                            const int switch_2_index = switch_2->index();
                            for (unsigned int l = 0;
                                 l < triangulation.levels.size();
                                 ++l)
                              for (unsigned int h = 0;
                                   h <
                                   triangulation.levels[l]->cells.n_objects();
                                   ++h)
                                for (const unsigned int q :
                                     GeometryInfo<dim>::face_indices())
                                  {
                                    const int face_index =
                                      triangulation.levels[l]
                                        ->cells.get_bounding_object_indices(
                                          h)[q];
                                    if (face_index == switch_1_index)
                                      triangulation.levels[l]
                                        ->cells.get_bounding_object_indices(
                                          h)[q] = switch_2_index;
                                    else if (face_index == switch_2_index)
                                      triangulation.levels[l]
                                        ->cells.get_bounding_object_indices(
                                          h)[q] = switch_1_index;
                                  }
                            // now we have to copy all information of
                            // the two quads
                            const unsigned int switch_1_lines[4] = {
                              switch_1->line_index(0),
                              switch_1->line_index(1),
                              switch_1->line_index(2),
                              switch_1->line_index(3)};
                            const bool switch_1_line_orientations[4] = {
                              switch_1->line_orientation(0),
                              switch_1->line_orientation(1),
                              switch_1->line_orientation(2),
                              switch_1->line_orientation(3)};
                            const types::boundary_id switch_1_boundary_id =
                              switch_1->boundary_id();
                            const unsigned int switch_1_user_index =
                              switch_1->user_index();
                            const bool switch_1_user_flag =
                              switch_1->user_flag_set();
                            const RefinementCase<dim - 1>
                              switch_1_refinement_case =
                                switch_1->refinement_case();
                            const int switch_1_first_child_pair =
                              (switch_1_refinement_case ?
                                 switch_1->child_index(0) :
                                 -1);
                            const int switch_1_second_child_pair =
                              (switch_1_refinement_case ==
                                   RefinementCase<dim - 1>::cut_xy ?
                                 switch_1->child_index(2) :
                                 -1);

                            switch_1->set_bounding_object_indices(
                              {switch_2->line_index(0),
                               switch_2->line_index(1),
                               switch_2->line_index(2),
                               switch_2->line_index(3)});
                            switch_1->set_line_orientation(
                              0, switch_2->line_orientation(0));
                            switch_1->set_line_orientation(
                              1, switch_2->line_orientation(1));
                            switch_1->set_line_orientation(
                              2, switch_2->line_orientation(2));
                            switch_1->set_line_orientation(
                              3, switch_2->line_orientation(3));
                            switch_1->set_boundary_id_internal(
                              switch_2->boundary_id());
                            switch_1->set_manifold_id(switch_2->manifold_id());
                            switch_1->set_user_index(switch_2->user_index());
                            if (switch_2->user_flag_set())
                              switch_1->set_user_flag();
                            else
                              switch_1->clear_user_flag();
                            switch_1->clear_refinement_case();
                            switch_1->set_refinement_case(
                              switch_2->refinement_case());
                            switch_1->clear_children();
                            if (switch_2->refinement_case())
                              switch_1->set_children(0,
                                                     switch_2->child_index(0));
                            if (switch_2->refinement_case() ==
                                RefinementCase<dim - 1>::cut_xy)
                              switch_1->set_children(2,
                                                     switch_2->child_index(2));

                            switch_2->set_bounding_object_indices(
                              {switch_1_lines[0],
                               switch_1_lines[1],
                               switch_1_lines[2],
                               switch_1_lines[3]});
                            switch_2->set_line_orientation(
                              0, switch_1_line_orientations[0]);
                            switch_2->set_line_orientation(
                              1, switch_1_line_orientations[1]);
                            switch_2->set_line_orientation(
                              2, switch_1_line_orientations[2]);
                            switch_2->set_line_orientation(
                              3, switch_1_line_orientations[3]);
                            switch_2->set_boundary_id_internal(
                              switch_1_boundary_id);
                            switch_2->set_manifold_id(switch_1->manifold_id());
                            switch_2->set_user_index(switch_1_user_index);
                            if (switch_1_user_flag)
                              switch_2->set_user_flag();
                            else
                              switch_2->clear_user_flag();
                            switch_2->clear_refinement_case();
                            switch_2->set_refinement_case(
                              switch_1_refinement_case);
                            switch_2->clear_children();
                            switch_2->set_children(0,
                                                   switch_1_first_child_pair);
                            switch_2->set_children(2,
                                                   switch_1_second_child_pair);

                            new_quads[0]->set_refinement_case(
                              RefinementCase<2>::cut_y);
                            new_quads[0]->set_children(0, quad->child_index(0));
                            new_quads[1]->set_refinement_case(
                              RefinementCase<2>::cut_y);
                            new_quads[1]->set_children(0, quad->child_index(2));
                          }
                        else
                          {
                            new_quads[0]->set_refinement_case(
                              RefinementCase<2>::cut_x);
                            new_quads[0]->set_children(0, quad->child_index(0));
                            new_quads[1]->set_refinement_case(
                              RefinementCase<2>::cut_x);
                            new_quads[1]->set_children(0, quad->child_index(2));
                            new_line->set_children(
                              0, quad->child(0)->line_index(3));
                            Assert(new_line->child(1) ==
                                     quad->child(1)->line(3),
                                   ExcInternalError());
                          }
                        quad->clear_children();
                      }

                    // note these quads as children to the present one
                    quad->set_children(0, new_quads[0]->index());

                    quad->set_refinement_case(aniso_quad_ref_case);

                    // finally clear flag indicating the need for
                    // refinement
                    quad->clear_user_data();
                  } // if (anisotropic refinement)

                if (quad->user_flag_set())
                  {
                    // this quad needs to be refined isotropically

                    // first of all: we only get here in the first run
                    // of the loop
                    Assert(loop == 0, ExcInternalError());

                    // find the next unused vertex. we'll need this in
                    // any case
                    while (triangulation.vertices_used[next_unused_vertex] ==
                           true)
                      ++next_unused_vertex;
                    Assert(
                      next_unused_vertex < triangulation.vertices.size(),
                      ExcMessage(
                        "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));

                    // now: if the quad is refined anisotropically
                    // already, set the anisotropic refinement flag
                    // for both children. Additionally, we have to
                    // refine the inner line, as it is an outer line
                    // of the two (anisotropic) children
                    const RefinementCase<dim - 1> quad_ref_case =
                      quad->refinement_case();

                    if (quad_ref_case == RefinementCase<dim - 1>::cut_x ||
                        quad_ref_case == RefinementCase<dim - 1>::cut_y)
                      {
                        // set the 'opposite' refine case for children
                        quad->child(0)->set_user_index(
                          RefinementCase<dim - 1>::cut_xy - quad_ref_case);
                        quad->child(1)->set_user_index(
                          RefinementCase<dim - 1>::cut_xy - quad_ref_case);
                        // refine the inner line
                        typename Triangulation<dim, spacedim>::line_iterator
                          middle_line;
                        if (quad_ref_case == RefinementCase<dim - 1>::cut_x)
                          middle_line = quad->child(0)->line(1);
                        else
                          middle_line = quad->child(0)->line(3);

                        // if the face has been refined
                        // anisotropically in the last refinement step
                        // it might be, that it is flagged already and
                        // that the middle line is thus refined
                        // already. if not create children.
                        if (!middle_line->has_children())
                          {
                            // set the middle vertex
                            // appropriately. double refinement of
                            // quads can only happen in the interior
                            // of the domain, so we need not care
                            // about boundary quads here
                            triangulation.vertices[next_unused_vertex] =
                              middle_line->center(true);
                            triangulation.vertices_used[next_unused_vertex] =
                              true;

                            // now search a slot for the two
                            // child lines
                            next_unused_line =
                              triangulation.faces->lines
                                .template next_free_pair_object<1>(
                                  triangulation);

                            // set the child pointer of the present
                            // line
                            middle_line->set_children(
                              0, next_unused_line->index());

                            // set the two new lines
                            const typename Triangulation<dim, spacedim>::
                              raw_line_iterator children[2] = {
                                next_unused_line, ++next_unused_line};

                            // some tests; if any of the iterators
                            // should be invalid, then already
                            // dereferencing will fail
                            AssertIsNotUsed(children[0]);
                            AssertIsNotUsed(children[1]);

                            children[0]->set_bounding_object_indices(
                              {middle_line->vertex_index(0),
                               next_unused_vertex});
                            children[1]->set_bounding_object_indices(
                              {next_unused_vertex,
                               middle_line->vertex_index(1)});

                            children[0]->set_used_flag();
                            children[1]->set_used_flag();
                            children[0]->clear_children();
                            children[1]->clear_children();
                            children[0]->clear_user_data();
                            children[1]->clear_user_data();
                            children[0]->clear_user_flag();
                            children[1]->clear_user_flag();

                            children[0]->set_boundary_id_internal(
                              middle_line->boundary_id());
                            children[1]->set_boundary_id_internal(
                              middle_line->boundary_id());

                            children[0]->set_manifold_id(
                              middle_line->manifold_id());
                            children[1]->set_manifold_id(
                              middle_line->manifold_id());
                          }
                        // now remove the flag from the quad and go to
                        // the next quad, the actual refinement of the
                        // quad takes place later on in this pass of
                        // the loop or in the next one
                        quad->clear_user_flag();
                        continue;
                      } // if (several refinement cases)

                    // if we got here, we have an unrefined quad and
                    // have to do the usual work like in an purely
                    // isotropic refinement
                    Assert(quad_ref_case ==
                             RefinementCase<dim - 1>::no_refinement,
                           ExcInternalError());

                    // set the middle vertex appropriately: it might be that
                    // the quad itself is not at the boundary, but that one of
                    // its lines actually is. in this case, the newly created
                    // vertices at the centers of the lines are not
                    // necessarily the mean values of the adjacent vertices,
                    // so do not compute the new vertex as the mean value of
                    // the 4 vertices of the face, but rather as a weighted
                    // mean value of the 8 vertices which we already have (the
                    // four old ones, and the four ones inserted as middle
                    // points for the four lines). summing up some more points
                    // is generally cheaper than first asking whether one of
                    // the lines is at the boundary
                    //
                    // note that the exact weights are chosen such as to
                    // minimize the distortion of the four new quads from the
                    // optimal shape. their description uses the formulas
                    // underlying the TransfiniteInterpolationManifold
                    // implementation
                    triangulation.vertices[next_unused_vertex] =
                      quad->center(true, true);
                    triangulation.vertices_used[next_unused_vertex] = true;

                    // now that we created the right point, make up
                    // the four lines interior to the quad (++ takes
                    // care of the end of the vector)
                    typename Triangulation<dim, spacedim>::raw_line_iterator
                      new_lines[4];

                    for (unsigned int i = 0; i < 4; ++i)
                      {
                        if (i % 2 == 0)
                          // search a free pair of lines for 0. and
                          // 2. line, so that two of them end up
                          // together, which is necessary if later on
                          // we want to refine the quad
                          // anisotropically and the two lines end up
                          // as children of new line
                          next_unused_line =
                            triangulation.faces->lines
                              .template next_free_pair_object<1>(triangulation);

                        new_lines[i] = next_unused_line;
                        ++next_unused_line;

                        AssertIsNotUsed(new_lines[i]);
                      }

                    // set the data of the four lines.  first collect
                    // the indices of the five vertices:
                    //
                    // *--3--*
                    // |  |  |
                    // 0--4--1
                    // |  |  |
                    // *--2--*
                    //
                    // the lines are numbered as follows:
                    //
                    // *--*--*
                    // |  1  |
                    // *2-*-3*
                    // |  0  |
                    // *--*--*

                    const unsigned int vertex_indices[5] = {
                      quad->line(0)->child(0)->vertex_index(1),
                      quad->line(1)->child(0)->vertex_index(1),
                      quad->line(2)->child(0)->vertex_index(1),
                      quad->line(3)->child(0)->vertex_index(1),
                      next_unused_vertex};

                    new_lines[0]->set_bounding_object_indices(
                      {vertex_indices[2], vertex_indices[4]});
                    new_lines[1]->set_bounding_object_indices(
                      {vertex_indices[4], vertex_indices[3]});
                    new_lines[2]->set_bounding_object_indices(
                      {vertex_indices[0], vertex_indices[4]});
                    new_lines[3]->set_bounding_object_indices(
                      {vertex_indices[4], vertex_indices[1]});

                    for (const auto &new_line : new_lines)
                      {
                        new_line->set_used_flag();
                        new_line->clear_user_flag();
                        new_line->clear_user_data();
                        new_line->clear_children();
                        new_line->set_boundary_id_internal(quad->boundary_id());
                        new_line->set_manifold_id(quad->manifold_id());
                      }

                    // now for the quads. again, first collect some
                    // data about the indices of the lines, with the
                    // following numbering:
                    //
                    //   .-6-.-7-.
                    //   1   9   3
                    //   .-10.11-.
                    //   0   8   2
                    //   .-4-.-5-.

                    // child 0 and 1 of a line are switched if the
                    // line orientation is false. set up a miniature
                    // table, indicating which child to take for line
                    // orientations false and true. first index: child
                    // index in standard orientation, second index:
                    // line orientation
                    const unsigned int index[2][2] = {
                      {1, 0},  // child 0, line_orientation=false and true
                      {0, 1}}; // child 1, line_orientation=false and true

                    const int line_indices[12] = {
                      quad->line(0)
                        ->child(index[0][quad->line_orientation(0)])
                        ->index(),
                      quad->line(0)
                        ->child(index[1][quad->line_orientation(0)])
                        ->index(),
                      quad->line(1)
                        ->child(index[0][quad->line_orientation(1)])
                        ->index(),
                      quad->line(1)
                        ->child(index[1][quad->line_orientation(1)])
                        ->index(),
                      quad->line(2)
                        ->child(index[0][quad->line_orientation(2)])
                        ->index(),
                      quad->line(2)
                        ->child(index[1][quad->line_orientation(2)])
                        ->index(),
                      quad->line(3)
                        ->child(index[0][quad->line_orientation(3)])
                        ->index(),
                      quad->line(3)
                        ->child(index[1][quad->line_orientation(3)])
                        ->index(),
                      new_lines[0]->index(),
                      new_lines[1]->index(),
                      new_lines[2]->index(),
                      new_lines[3]->index()};

                    // find some space (consecutive)
                    // for the first two newly to be
                    // created quads.
                    typename Triangulation<dim, spacedim>::raw_quad_iterator
                      new_quads[4];

                    next_unused_quad =
                      triangulation.faces->quads
                        .template next_free_pair_object<2>(triangulation);

                    new_quads[0] = next_unused_quad;
                    AssertIsNotUsed(new_quads[0]);

                    ++next_unused_quad;
                    new_quads[1] = next_unused_quad;
                    AssertIsNotUsed(new_quads[1]);

                    next_unused_quad =
                      triangulation.faces->quads
                        .template next_free_pair_object<2>(triangulation);
                    new_quads[2] = next_unused_quad;
                    AssertIsNotUsed(new_quads[2]);

                    ++next_unused_quad;
                    new_quads[3] = next_unused_quad;
                    AssertIsNotUsed(new_quads[3]);

                    // note these quads as children to the present one
                    quad->set_children(0, new_quads[0]->index());
                    quad->set_children(2, new_quads[2]->index());
                    quad->set_refinement_case(RefinementCase<2>::cut_xy);

                    new_quads[0]->set_bounding_object_indices(
                      {line_indices[0],
                       line_indices[8],
                       line_indices[4],
                       line_indices[10]});
                    new_quads[1]->set_bounding_object_indices(
                      {line_indices[8],
                       line_indices[2],
                       line_indices[5],
                       line_indices[11]});
                    new_quads[2]->set_bounding_object_indices(
                      {line_indices[1],
                       line_indices[9],
                       line_indices[10],
                       line_indices[6]});
                    new_quads[3]->set_bounding_object_indices(
                      {line_indices[9],
                       line_indices[3],
                       line_indices[11],
                       line_indices[7]});
                    for (const auto &new_quad : new_quads)
                      {
                        new_quad->set_used_flag();
                        new_quad->clear_user_flag();
                        new_quad->clear_user_data();
                        new_quad->clear_children();
                        new_quad->set_boundary_id_internal(quad->boundary_id());
                        new_quad->set_manifold_id(quad->manifold_id());
                        // set all line orientations to true, change
                        // this after the loop, as we have to consider
                        // different lines for each child
                        for (unsigned int j = 0;
                             j < GeometryInfo<dim>::lines_per_face;
                             ++j)
                          new_quad->set_line_orientation(j, true);
                      }
                    // now set the line orientation of children of
                    // outer lines correctly, the lines in the
                    // interior of the refined quad are automatically
                    // oriented conforming to the standard
                    new_quads[0]->set_line_orientation(
                      0, quad->line_orientation(0));
                    new_quads[0]->set_line_orientation(
                      2, quad->line_orientation(2));
                    new_quads[1]->set_line_orientation(
                      1, quad->line_orientation(1));
                    new_quads[1]->set_line_orientation(
                      2, quad->line_orientation(2));
                    new_quads[2]->set_line_orientation(
                      0, quad->line_orientation(0));
                    new_quads[2]->set_line_orientation(
                      3, quad->line_orientation(3));
                    new_quads[3]->set_line_orientation(
                      1, quad->line_orientation(1));
                    new_quads[3]->set_line_orientation(
                      3, quad->line_orientation(3));

                    // finally clear flag indicating the need for
                    // refinement
                    quad->clear_user_flag();
                  } // if (isotropic refinement)
              }     // for all quads
          }         // looped two times over all quads, all quads refined now

        //---------------------------------
        // Now, finally, set up the new
        // cells
        //---------------------------------

        typename Triangulation<3, spacedim>::DistortedCellList
          cells_with_distorted_children;

        for (unsigned int level = 0; level != triangulation.levels.size() - 1;
             ++level)
          {
            // only active objects can be refined further; remember
            // that we won't operate on the finest level, so
            // triangulation.begin_*(level+1) is allowed
            typename Triangulation<dim, spacedim>::active_hex_iterator
              hex  = triangulation.begin_active_hex(level),
              endh = triangulation.begin_active_hex(level + 1);
            typename Triangulation<dim, spacedim>::raw_hex_iterator
              next_unused_hex = triangulation.begin_raw_hex(level + 1);

            for (; hex != endh; ++hex)
              if (hex->refine_flag_set())
                {
                  // this hex needs to be refined

                  // clear flag indicating the need for refinement. do
                  // it here already, since we can't do it anymore
                  // once the cell has children
                  const RefinementCase<dim> ref_case = hex->refine_flag_set();
                  hex->clear_refine_flag();
                  hex->set_refinement_case(ref_case);

                  // depending on the refine case we might have to
                  // create additional vertices, lines and quads
                  // interior of the hex before the actual children
                  // can be set up.

                  // in a first step: reserve the needed space for
                  // lines, quads and hexes and initialize them
                  // correctly

                  unsigned int n_new_lines = 0;
                  unsigned int n_new_quads = 0;
                  unsigned int n_new_hexes = 0;
                  switch (ref_case)
                    {
                      case RefinementCase<dim>::cut_x:
                      case RefinementCase<dim>::cut_y:
                      case RefinementCase<dim>::cut_z:
                        n_new_lines = 0;
                        n_new_quads = 1;
                        n_new_hexes = 2;
                        break;
                      case RefinementCase<dim>::cut_xy:
                      case RefinementCase<dim>::cut_xz:
                      case RefinementCase<dim>::cut_yz:
                        n_new_lines = 1;
                        n_new_quads = 4;
                        n_new_hexes = 4;
                        break;
                      case RefinementCase<dim>::cut_xyz:
                        n_new_lines = 6;
                        n_new_quads = 12;
                        n_new_hexes = 8;
                        break;
                      default:
                        Assert(false, ExcInternalError());
                        break;
                    }

                  // find some space for the newly to be created
                  // interior lines and initialize them.
                  std::vector<
                    typename Triangulation<dim, spacedim>::raw_line_iterator>
                    new_lines(n_new_lines);
                  for (unsigned int i = 0; i < n_new_lines; ++i)
                    {
                      new_lines[i] =
                        triangulation.faces->lines
                          .template next_free_single_object<1>(triangulation);

                      AssertIsNotUsed(new_lines[i]);
                      new_lines[i]->set_used_flag();
                      new_lines[i]->clear_user_flag();
                      new_lines[i]->clear_user_data();
                      new_lines[i]->clear_children();
                      // interior line
                      new_lines[i]->set_boundary_id_internal(
                        numbers::internal_face_boundary_id);
                      // they inherit geometry description of the hex they
                      // belong to
                      new_lines[i]->set_manifold_id(hex->manifold_id());
                    }

                  // find some space for the newly to be created
                  // interior quads and initialize them.
                  std::vector<
                    typename Triangulation<dim, spacedim>::raw_quad_iterator>
                    new_quads(n_new_quads);
                  for (unsigned int i = 0; i < n_new_quads; ++i)
                    {
                      new_quads[i] =
                        triangulation.faces->quads
                          .template next_free_single_object<2>(triangulation);

                      AssertIsNotUsed(new_quads[i]);
                      new_quads[i]->set_used_flag();
                      new_quads[i]->clear_user_flag();
                      new_quads[i]->clear_user_data();
                      new_quads[i]->clear_children();
                      // interior quad
                      new_quads[i]->set_boundary_id_internal(
                        numbers::internal_face_boundary_id);
                      // they inherit geometry description of the hex they
                      // belong to
                      new_quads[i]->set_manifold_id(hex->manifold_id());
                      // set all line orientation flags to true by
                      // default, change this afterwards, if necessary
                      for (unsigned int j = 0;
                           j < GeometryInfo<dim>::lines_per_face;
                           ++j)
                        new_quads[i]->set_line_orientation(j, true);
                    }

                  types::subdomain_id subdomainid = hex->subdomain_id();

                  // find some space for the newly to be created hexes
                  // and initialize them.
                  std::vector<
                    typename Triangulation<dim, spacedim>::raw_hex_iterator>
                    new_hexes(n_new_hexes);
                  for (unsigned int i = 0; i < n_new_hexes; ++i)
                    {
                      if (i % 2 == 0)
                        next_unused_hex =
                          triangulation.levels[level + 1]->cells.next_free_hex(
                            triangulation, level + 1);
                      else
                        ++next_unused_hex;

                      new_hexes[i] = next_unused_hex;

                      AssertIsNotUsed(new_hexes[i]);
                      new_hexes[i]->set_used_flag();
                      new_hexes[i]->clear_user_flag();
                      new_hexes[i]->clear_user_data();
                      new_hexes[i]->clear_children();
                      // inherit material
                      // properties
                      new_hexes[i]->set_material_id(hex->material_id());
                      new_hexes[i]->set_manifold_id(hex->manifold_id());
                      new_hexes[i]->set_subdomain_id(subdomainid);

                      if (i % 2)
                        new_hexes[i]->set_parent(hex->index());
                      // set the face_orientation flag to true for all
                      // faces initially, as this is the default value
                      // which is true for all faces interior to the
                      // hex. later on go the other way round and
                      // reset faces that are at the boundary of the
                      // mother cube
                      //
                      // the same is true for the face_flip and
                      // face_rotation flags. however, the latter two
                      // are set to false by default as this is the
                      // standard value
                      for (const unsigned int f :
                           GeometryInfo<dim>::face_indices())
                        {
                          new_hexes[i]->set_face_orientation(f, true);
                          new_hexes[i]->set_face_flip(f, false);
                          new_hexes[i]->set_face_rotation(f, false);
                        }
                    }
                  // note these hexes as children to the present cell
                  for (unsigned int i = 0; i < n_new_hexes / 2; ++i)
                    hex->set_children(2 * i, new_hexes[2 * i]->index());

                  // we have to take into account whether the
                  // different faces are oriented correctly or in the
                  // opposite direction, so store that up front

                  // face_orientation
                  const bool f_or[6] = {hex->face_orientation(0),
                                        hex->face_orientation(1),
                                        hex->face_orientation(2),
                                        hex->face_orientation(3),
                                        hex->face_orientation(4),
                                        hex->face_orientation(5)};

                  // face_flip
                  const bool f_fl[6] = {hex->face_flip(0),
                                        hex->face_flip(1),
                                        hex->face_flip(2),
                                        hex->face_flip(3),
                                        hex->face_flip(4),
                                        hex->face_flip(5)};

                  // face_rotation
                  const bool f_ro[6] = {hex->face_rotation(0),
                                        hex->face_rotation(1),
                                        hex->face_rotation(2),
                                        hex->face_rotation(3),
                                        hex->face_rotation(4),
                                        hex->face_rotation(5)};

                  // little helper table, indicating, whether the
                  // child with index 0 or with index 1 can be found
                  // at the standard origin of an anisotropically
                  // refined quads in real orientation index 1:
                  // (RefineCase - 1) index 2: face_flip

                  // index 3: face rotation
                  // note: face orientation has no influence
                  const unsigned int child_at_origin[2][2][2] = {
                    {{0, 0},   // RefinementCase<dim>::cut_x, face_flip=false,
                               // face_rotation=false and true
                     {1, 1}},  // RefinementCase<dim>::cut_x, face_flip=true,
                               // face_rotation=false and true
                    {{0, 1},   // RefinementCase<dim>::cut_y, face_flip=false,
                               // face_rotation=false and true
                     {1, 0}}}; // RefinementCase<dim>::cut_y, face_flip=true,
                               // face_rotation=false and true

                  //-------------------------------------
                  //
                  // in the following we will do the same thing for
                  // each refinement case: create a new vertex (if
                  // needed), create new interior lines (if needed),
                  // create new interior quads and afterwards build
                  // the children hexes out of these and the existing
                  // subfaces of the outer quads (which have been
                  // created above). However, even if the steps are
                  // quite similar, the actual work strongly depends
                  // on the actual refinement case. therefore, we use
                  // separate blocks of code for each of these cases,
                  // which hopefully increases the readability to some
                  // extend.

                  switch (ref_case)
                    {
                      case RefinementCase<dim>::cut_x:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_x
                          //
                          // the refined cube will look
                          // like this:
                          //
                          //        *----*----*
                          //       /    /    /|
                          //      /    /    / |
                          //     /    /    /  |
                          //    *----*----*   |
                          //    |    |    |   |
                          //    |    |    |   *
                          //    |    |    |  /
                          //    |    |    | /
                          //    |    |    |/
                          //    *----*----*
                          //
                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering:

                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   |   |
                          //   |   0   |
                          //   |   |   |
                          //   *---*---*
                          //       m0
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //       m1
                          //   *---*---*
                          //   |   |   |
                          //   |   1   |
                          //   |   |   |
                          //   *---*---*
                          // face 4: bottom plane
                          //       *---*---*
                          //      /   /   /
                          //     /   2   /
                          //    /   /   /
                          //   *---*---*
                          //       m0
                          // face 5: top plane
                          //           m1
                          //       *---*---*
                          //      /   /   /
                          //     /   3   /
                          //    /   /   /
                          //   *---*---*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<dim, spacedim>::
                            raw_line_iterator lines[4] = {
                              hex->face(2)->child(0)->line(
                                (hex->face(2)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 0
                              hex->face(3)->child(0)->line(
                                (hex->face(3)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 1
                              hex->face(4)->child(0)->line(
                                (hex->face(4)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 2
                              hex->face(5)->child(0)->line(
                                (hex->face(5)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3) // 3
                            };

                          unsigned int line_indices[4];
                          for (unsigned int i = 0; i < 4; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[4];

                          // the middle vertex marked as m0 above is the
                          // start vertex for lines 0 and 2 in standard
                          // orientation, whereas m1 is the end vertex of
                          // lines 1 and 3 in standard orientation
                          const unsigned int middle_vertices[2] = {
                            hex->line(2)->child(0)->vertex_index(1),
                            hex->line(7)->child(0)->vertex_index(1)};

                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(i % 2) ==
                                middle_vertices[i % 2])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other
                                // way round then
                                Assert(lines[i]->vertex_index((i + 1) % 2) ==
                                         middle_vertices[i % 2],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // set up the new quad, line numbering is as
                          // indicated above
                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[0],
                             line_indices[1],
                             line_indices[2],
                             line_indices[3]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[0]);
                          new_quads[0]->set_line_orientation(
                            1, line_orientation[1]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[2]);
                          new_quads[0]->set_line_orientation(
                            3, line_orientation[3]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    / | x
                          //   /  | *-------*      *---------*
                          //  *   | |       |     /         /
                          //  | 0 | |       |    /         /
                          //  |   * |       |   /         /
                          //  |  /  *-------*y *---------*x
                          //  | /
                          //  |/
                          //  *
                          //
                          // children of the faces of the old hex
                          //
                          //      *---*---*        *---*---*
                          //     /|   |   |       /   /   /|
                          //    / |   |   |      / 9 / 10/ |
                          //   /  | 5 | 6 |     /   /   /  |
                          //  *   |   |   |    *---*---*   |
                          //  | 1 *---*---*    |   |   | 2 *
                          //  |  /   /   /     |   |   |  /
                          //  | / 7 / 8 /      | 3 | 4 | /
                          //  |/   /   /       |   |   |/
                          //  *---*---*        *---*---*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[11] = {
                            new_quads[0]->index(), // 0

                            hex->face(0)->index(), // 1

                            hex->face(1)->index(), // 2

                            hex->face(2)->child_index(
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]), // 3
                            hex->face(2)->child_index(
                              1 -
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]),

                            hex->face(3)->child_index(
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]), // 5
                            hex->face(3)->child_index(
                              1 -
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]),

                            hex->face(4)->child_index(
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]), // 7
                            hex->face(4)->child_index(
                              1 -
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]),

                            hex->face(5)->child_index(
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]]), // 9
                            hex->face(5)->child_index(
                              1 -
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]])

                          };

                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[1],
                             quad_indices[0],
                             quad_indices[3],
                             quad_indices[5],
                             quad_indices[7],
                             quad_indices[9]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[0],
                             quad_indices[2],
                             quad_indices[4],
                             quad_indices[6],
                             quad_indices[8],
                             quad_indices[10]});
                          break;
                        }

                      case RefinementCase<dim>::cut_y:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_y
                          //
                          // the refined cube will look like this:
                          //
                          //        *---------*
                          //       /         /|
                          //      *---------* |
                          //     /         /| |
                          //    *---------* | |
                          //    |         | | |
                          //    |         | | *
                          //    |         | |/
                          //    |         | *
                          //    |         |/
                          //    *---------*
                          //
                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering:

                          // face 0: left plane
                          //       *
                          //      /|
                          //     * |
                          //    /| |
                          //   * | |
                          //   | 0 |
                          //   | | *
                          //   | |/
                          //   | *m0
                          //   |/
                          //   *
                          // face 1: right plane
                          //       *
                          //      /|
                          //   m1* |
                          //    /| |
                          //   * | |
                          //   | 1 |
                          //   | | *
                          //   | |/
                          //   | *
                          //   |/
                          //   *
                          // face 4: bottom plane
                          //       *-------*
                          //      /       /
                          //   m0*---2---*
                          //    /       /
                          //   *-------*
                          // face 5: top plane
                          //       *-------*
                          //      /       /
                          //     *---3---*m1
                          //    /       /
                          //   *-------*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<dim, spacedim>::
                            raw_line_iterator lines[4] = {
                              hex->face(0)->child(0)->line(
                                (hex->face(0)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 0
                              hex->face(1)->child(0)->line(
                                (hex->face(1)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 1
                              hex->face(4)->child(0)->line(
                                (hex->face(4)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 2
                              hex->face(5)->child(0)->line(
                                (hex->face(5)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3) // 3
                            };

                          unsigned int line_indices[4];
                          for (unsigned int i = 0; i < 4; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[4];

                          // the middle vertex marked as m0 above is the
                          // start vertex for lines 0 and 2 in standard
                          // orientation, whereas m1 is the end vertex of
                          // lines 1 and 3 in standard orientation
                          const unsigned int middle_vertices[2] = {
                            hex->line(0)->child(0)->vertex_index(1),
                            hex->line(5)->child(0)->vertex_index(1)};

                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(i % 2) ==
                                middle_vertices[i % 2])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way round then
                                Assert(lines[i]->vertex_index((i + 1) % 2) ==
                                         middle_vertices[i % 2],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // set up the new quad, line numbering is as
                          // indicated above
                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[2],
                             line_indices[3],
                             line_indices[0],
                             line_indices[1]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[2]);
                          new_quads[0]->set_line_orientation(
                            1, line_orientation[3]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[0]);
                          new_quads[0]->set_line_orientation(
                            3, line_orientation[1]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    / | x
                          //   /  | *-------*      *---------*
                          //  *   | |       |     /         /
                          //  |   | |   0   |    /         /
                          //  |   * |       |   /         /
                          //  |  /  *-------*y *---------*x
                          //  | /
                          //  |/
                          //  *
                          //
                          // children of the faces of the old hex
                          //
                          //      *-------*        *-------*
                          //     /|       |       /   10  /|
                          //    * |       |      *-------* |
                          //   /| |   6   |     /   9   /| |
                          //  * |2|       |    *-------* |4|
                          //  | | *-------*    |       | | *
                          //  |1|/   8   /     |       |3|/
                          //  | *-------*      |   5   | *
                          //  |/   7   /       |       |/
                          //  *-------*        *-------*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[11] = {
                            new_quads[0]->index(), // 0

                            hex->face(0)->child_index(
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]), // 1
                            hex->face(0)->child_index(
                              1 -
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]),

                            hex->face(1)->child_index(
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]), // 3
                            hex->face(1)->child_index(
                              1 -
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]),

                            hex->face(2)->index(), // 5

                            hex->face(3)->index(), // 6

                            hex->face(4)->child_index(
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]), // 7
                            hex->face(4)->child_index(
                              1 -
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]),

                            hex->face(5)->child_index(
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]]), // 9
                            hex->face(5)->child_index(
                              1 -
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]])

                          };

                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[1],
                             quad_indices[3],
                             quad_indices[5],
                             quad_indices[0],
                             quad_indices[7],
                             quad_indices[9]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[2],
                             quad_indices[4],
                             quad_indices[0],
                             quad_indices[6],
                             quad_indices[8],
                             quad_indices[10]});
                          break;
                        }

                      case RefinementCase<dim>::cut_z:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_z
                          //
                          // the refined cube will look like this:
                          //
                          //        *---------*
                          //       /         /|
                          //      /         / |
                          //     /         /  *
                          //    *---------*  /|
                          //    |         | / |
                          //    |         |/  *
                          //    *---------*  /
                          //    |         | /
                          //    |         |/
                          //    *---------*
                          //
                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering:

                          // face 0: left plane
                          //       *
                          //      /|
                          //     / |
                          //    /  *
                          //   *  /|
                          //   | 0 |
                          //   |/  *
                          // m0*  /
                          //   | /
                          //   |/
                          //   *
                          // face 1: right plane
                          //       *
                          //      /|
                          //     / |
                          //    /  *m1
                          //   *  /|
                          //   | 1 |
                          //   |/  *
                          //   *  /
                          //   | /
                          //   |/
                          //   *
                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *-------*
                          //   |       |
                          // m0*---2---*
                          //   |       |
                          //   *-------*
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //   *-------*
                          //   |       |
                          //   *---3---*m1
                          //   |       |
                          //   *-------*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<dim, spacedim>::
                            raw_line_iterator lines[4] = {
                              hex->face(0)->child(0)->line(
                                (hex->face(0)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 0
                              hex->face(1)->child(0)->line(
                                (hex->face(1)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 1
                              hex->face(2)->child(0)->line(
                                (hex->face(2)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3), // 2
                              hex->face(3)->child(0)->line(
                                (hex->face(3)->refinement_case() ==
                                 RefinementCase<2>::cut_x) ?
                                  1 :
                                  3) // 3
                            };

                          unsigned int line_indices[4];
                          for (unsigned int i = 0; i < 4; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[4];

                          // the middle vertex marked as m0 above is the
                          // start vertex for lines 0 and 2 in standard
                          // orientation, whereas m1 is the end vertex of
                          // lines 1 and 3 in standard orientation
                          const unsigned int middle_vertices[2] = {
                            middle_vertex_index<dim, spacedim>(hex->line(8)),
                            middle_vertex_index<dim, spacedim>(hex->line(11))};

                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(i % 2) ==
                                middle_vertices[i % 2])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way round then
                                Assert(lines[i]->vertex_index((i + 1) % 2) ==
                                         middle_vertices[i % 2],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // set up the new quad, line numbering is as
                          // indicated above
                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[0],
                             line_indices[1],
                             line_indices[2],
                             line_indices[3]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[0]);
                          new_quads[0]->set_line_orientation(
                            1, line_orientation[1]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[2]);
                          new_quads[0]->set_line_orientation(
                            3, line_orientation[3]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    / | x
                          //   /  | *-------*      *---------*
                          //  *   | |       |     /         /
                          //  |   | |       |    /    0    /
                          //  |   * |       |   /         /
                          //  |  /  *-------*y *---------*x
                          //  | /
                          //  |/
                          //  *
                          //
                          // children of the faces of the old hex
                          //
                          //      *---*---*        *-------*
                          //     /|   8   |       /       /|
                          //    / |       |      /   10  / |
                          //   /  *-------*     /       /  *
                          //  * 2/|       |    *-------* 4/|
                          //  | / |   7   |    |   6   | / |
                          //  |/1 *-------*    |       |/3 *
                          //  *  /       /     *-------*  /
                          //  | /   9   /      |       | /
                          //  |/       /       |   5   |/
                          //  *-------*        *---*---*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[11] = {
                            new_quads[0]->index(), // 0

                            hex->face(0)->child_index(
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]), // 1
                            hex->face(0)->child_index(
                              1 -
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]),

                            hex->face(1)->child_index(
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]), // 3
                            hex->face(1)->child_index(
                              1 -
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]),

                            hex->face(2)->child_index(
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]), // 5
                            hex->face(2)->child_index(
                              1 -
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]),

                            hex->face(3)->child_index(
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]), // 7
                            hex->face(3)->child_index(
                              1 -
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]),

                            hex->face(4)->index(), // 9

                            hex->face(5)->index() // 10
                          };

                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[1],
                             quad_indices[3],
                             quad_indices[5],
                             quad_indices[7],
                             quad_indices[9],
                             quad_indices[0]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[2],
                             quad_indices[4],
                             quad_indices[6],
                             quad_indices[8],
                             quad_indices[0],
                             quad_indices[10]});
                          break;
                        }

                      case RefinementCase<dim>::cut_xy:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_xy
                          //
                          // the refined cube will look like this:
                          //
                          //        *----*----*
                          //       /    /    /|
                          //      *----*----* |
                          //     /    /    /| |
                          //    *----*----* | |
                          //    |    |    | | |
                          //    |    |    | | *
                          //    |    |    | |/
                          //    |    |    | *
                          //    |    |    |/
                          //    *----*----*
                          //

                          // first, create the new internal line
                          new_lines[0]->set_bounding_object_indices(
                            {middle_vertex_index<dim, spacedim>(hex->face(4)),
                             middle_vertex_index<dim, spacedim>(hex->face(5))});

                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering:

                          // face 0: left plane
                          //       *
                          //      /|
                          //     * |
                          //    /| |
                          //   * | |
                          //   | 0 |
                          //   | | *
                          //   | |/
                          //   | *
                          //   |/
                          //   *
                          // face 1: right plane
                          //       *
                          //      /|
                          //     * |
                          //    /| |
                          //   * | |
                          //   | 1 |
                          //   | | *
                          //   | |/
                          //   | *
                          //   |/
                          //   *
                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   |   |
                          //   |   2   |
                          //   |   |   |
                          //   *-------*
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   |   |
                          //   |   3   |
                          //   |   |   |
                          //   *---*---*
                          // face 4: bottom plane
                          //       *---*---*
                          //      /   5   /
                          //     *-6-*-7-*
                          //    /   4   /
                          //   *---*---*
                          // face 5: top plane
                          //       *---*---*
                          //      /   9   /
                          //     *10-*-11*
                          //    /   8   /
                          //   *---*---*
                          // middle planes
                          //     *-------*   *---*---*
                          //    /       /    |   |   |
                          //   /       /     |   12  |
                          //  /       /      |   |   |
                          // *-------*       *---*---*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<
                            dim,
                            spacedim>::raw_line_iterator lines[13] = {
                            hex->face(0)->child(0)->line(
                              (hex->face(0)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 0
                            hex->face(1)->child(0)->line(
                              (hex->face(1)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 1
                            hex->face(2)->child(0)->line(
                              (hex->face(2)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 2
                            hex->face(3)->child(0)->line(
                              (hex->face(3)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 3

                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[4], f_fl[4], f_ro[4])), // 4
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[4], f_fl[4], f_ro[4])), // 5
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[4], f_fl[4], f_ro[4])), // 6
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[4], f_fl[4], f_ro[4])), // 7

                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[5], f_fl[5], f_ro[5])), // 8
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[5], f_fl[5], f_ro[5])), // 9
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[5], f_fl[5], f_ro[5])), // 10
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[5], f_fl[5], f_ro[5])), // 11

                            new_lines[0] // 12
                          };

                          unsigned int line_indices[13];
                          for (unsigned int i = 0; i < 13; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[13];

                          // the middle vertices of the lines of our
                          // bottom face
                          const unsigned int middle_vertices[4] = {
                            hex->line(0)->child(0)->vertex_index(1),
                            hex->line(1)->child(0)->vertex_index(1),
                            hex->line(2)->child(0)->vertex_index(1),
                            hex->line(3)->child(0)->vertex_index(1),
                          };

                          // note: for lines 0 to 3 the orientation of the
                          // line is 'true', if vertex 0 is on the bottom
                          // face
                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(0) == middle_vertices[i])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way round then
                                Assert(lines[i]->vertex_index(1) ==
                                         middle_vertices[i],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // note: for lines 4 to 11 (inner lines of the
                          // outer quads) the following holds: the second
                          // vertex of the even lines in standard
                          // orientation is the vertex in the middle of
                          // the quad, whereas for odd lines the first
                          // vertex is the same middle vertex.
                          for (unsigned int i = 4; i < 12; ++i)
                            if (lines[i]->vertex_index((i + 1) % 2) ==
                                middle_vertex_index<dim, spacedim>(
                                  hex->face(3 + i / 4)))
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way
                                // round then
                                Assert(lines[i]->vertex_index(i % 2) ==
                                         (middle_vertex_index<dim, spacedim>(
                                           hex->face(3 + i / 4))),
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }
                          // for the last line the line orientation is
                          // always true, since it was just constructed
                          // that way
                          line_orientation[12] = true;

                          // set up the 4 quads, numbered as follows (left
                          // quad numbering, right line numbering
                          // extracted from above)
                          //
                          //      *          *
                          //     /|         9|
                          //    * |        * |
                          //  y/| |       8| 3
                          //  * |1|      * | |
                          //  | | |x     | 12|
                          //  |0| *      | | *
                          //  | |/       2 |5
                          //  | *        | *
                          //  |/         |4
                          //  *          *
                          //
                          //  x
                          //  *---*---*      *10-*-11*
                          //  |   |   |      |   |   |
                          //  | 2 | 3 |      0   12  1
                          //  |   |   |      |   |   |
                          //  *---*---*y     *-6-*-7-*

                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[2],
                             line_indices[12],
                             line_indices[4],
                             line_indices[8]});
                          new_quads[1]->set_bounding_object_indices(
                            {line_indices[12],
                             line_indices[3],
                             line_indices[5],
                             line_indices[9]});
                          new_quads[2]->set_bounding_object_indices(
                            {line_indices[6],
                             line_indices[10],
                             line_indices[0],
                             line_indices[12]});
                          new_quads[3]->set_bounding_object_indices(
                            {line_indices[7],
                             line_indices[11],
                             line_indices[12],
                             line_indices[1]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[2]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[4]);
                          new_quads[0]->set_line_orientation(
                            3, line_orientation[8]);

                          new_quads[1]->set_line_orientation(
                            1, line_orientation[3]);
                          new_quads[1]->set_line_orientation(
                            2, line_orientation[5]);
                          new_quads[1]->set_line_orientation(
                            3, line_orientation[9]);

                          new_quads[2]->set_line_orientation(
                            0, line_orientation[6]);
                          new_quads[2]->set_line_orientation(
                            1, line_orientation[10]);
                          new_quads[2]->set_line_orientation(
                            2, line_orientation[0]);

                          new_quads[3]->set_line_orientation(
                            0, line_orientation[7]);
                          new_quads[3]->set_line_orientation(
                            1, line_orientation[11]);
                          new_quads[3]->set_line_orientation(
                            3, line_orientation[1]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    * | x
                          //   /| | *---*---*      *---------*
                          //  * |1| |   |   |     /         /
                          //  | | | | 2 | 3 |    /         /
                          //  |0| * |   |   |   /         /
                          //  | |/  *---*---*y *---------*x
                          //  | *
                          //  |/
                          //  *
                          //
                          // children of the faces of the old hex
                          //
                          //      *---*---*        *---*---*
                          //     /|   |   |       /18 / 19/|
                          //    * |10 | 11|      /---/---* |
                          //   /| |   |   |     /16 / 17/| |
                          //  * |5|   |   |    *---*---* |7|
                          //  | | *---*---*    |   |   | | *
                          //  |4|/14 / 15/     |   |   |6|/
                          //  | *---/---/      | 8 | 9 | *
                          //  |/12 / 13/       |   |   |/
                          //  *---*---*        *---*---*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[20] = {
                            new_quads[0]->index(), // 0
                            new_quads[1]->index(),
                            new_quads[2]->index(),
                            new_quads[3]->index(),

                            hex->face(0)->child_index(
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]), // 4
                            hex->face(0)->child_index(
                              1 -
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]),

                            hex->face(1)->child_index(
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]), // 6
                            hex->face(1)->child_index(
                              1 -
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]),

                            hex->face(2)->child_index(
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]), // 8
                            hex->face(2)->child_index(
                              1 -
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]),

                            hex->face(3)->child_index(
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]), // 10
                            hex->face(3)->child_index(
                              1 -
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]),

                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[4], f_fl[4], f_ro[4])), // 12
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[4], f_fl[4], f_ro[4])),
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[4], f_fl[4], f_ro[4])),
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[4], f_fl[4], f_ro[4])),

                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[5], f_fl[5], f_ro[5])), // 16
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[5], f_fl[5], f_ro[5])),
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[5], f_fl[5], f_ro[5])),
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[5], f_fl[5], f_ro[5]))};

                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[4],
                             quad_indices[0],
                             quad_indices[8],
                             quad_indices[2],
                             quad_indices[12],
                             quad_indices[16]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[0],
                             quad_indices[6],
                             quad_indices[9],
                             quad_indices[3],
                             quad_indices[13],
                             quad_indices[17]});
                          new_hexes[2]->set_bounding_object_indices(
                            {quad_indices[5],
                             quad_indices[1],
                             quad_indices[2],
                             quad_indices[10],
                             quad_indices[14],
                             quad_indices[18]});
                          new_hexes[3]->set_bounding_object_indices(
                            {quad_indices[1],
                             quad_indices[7],
                             quad_indices[3],
                             quad_indices[11],
                             quad_indices[15],
                             quad_indices[19]});
                          break;
                        }

                      case RefinementCase<dim>::cut_xz:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_xz
                          //
                          // the refined cube will look like this:
                          //
                          //        *----*----*
                          //       /    /    /|
                          //      /    /    / |
                          //     /    /    /  *
                          //    *----*----*  /|
                          //    |    |    | / |
                          //    |    |    |/  *
                          //    *----*----*  /
                          //    |    |    | /
                          //    |    |    |/
                          //    *----*----*
                          //

                          // first, create the new internal line
                          new_lines[0]->set_bounding_object_indices(
                            {middle_vertex_index<dim, spacedim>(hex->face(2)),
                             middle_vertex_index<dim, spacedim>(hex->face(3))});

                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering:

                          // face 0: left plane
                          //       *
                          //      /|
                          //     / |
                          //    /  *
                          //   *  /|
                          //   | 0 |
                          //   |/  *
                          //   *  /
                          //   | /
                          //   |/
                          //   *
                          // face 1: right plane
                          //       *
                          //      /|
                          //     / |
                          //    /  *
                          //   *  /|
                          //   | 1 |
                          //   |/  *
                          //   *  /
                          //   | /
                          //   |/
                          //   *
                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   5   |
                          //   *-6-*-7-*
                          //   |   4   |
                          //   *---*---*
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   9   |
                          //   *10-*-11*
                          //   |   8   |
                          //   *---*---*
                          // face 4: bottom plane
                          //       *---*---*
                          //      /   /   /
                          //     /   2   /
                          //    /   /   /
                          //   *---*---*
                          // face 5: top plane
                          //       *---*---*
                          //      /   /   /
                          //     /   3   /
                          //    /   /   /
                          //   *---*---*
                          // middle planes
                          //     *---*---*   *-------*
                          //    /   /   /    |       |
                          //   /   12  /     |       |
                          //  /   /   /      |       |
                          // *---*---*       *-------*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<
                            dim,
                            spacedim>::raw_line_iterator lines[13] = {
                            hex->face(0)->child(0)->line(
                              (hex->face(0)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 0
                            hex->face(1)->child(0)->line(
                              (hex->face(1)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 1
                            hex->face(4)->child(0)->line(
                              (hex->face(4)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 2
                            hex->face(5)->child(0)->line(
                              (hex->face(5)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 3

                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[2], f_fl[2], f_ro[2])), // 4
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[2], f_fl[2], f_ro[2])), // 5
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[2], f_fl[2], f_ro[2])), // 6
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[2], f_fl[2], f_ro[2])), // 7

                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[3], f_fl[3], f_ro[3])), // 8
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[3], f_fl[3], f_ro[3])), // 9
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[3], f_fl[3], f_ro[3])), // 10
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[3], f_fl[3], f_ro[3])), // 11

                            new_lines[0] // 12
                          };

                          unsigned int line_indices[13];
                          for (unsigned int i = 0; i < 13; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[13];

                          // the middle vertices of the
                          // lines of our front face
                          const unsigned int middle_vertices[4] = {
                            hex->line(8)->child(0)->vertex_index(1),
                            hex->line(9)->child(0)->vertex_index(1),
                            hex->line(2)->child(0)->vertex_index(1),
                            hex->line(6)->child(0)->vertex_index(1),
                          };

                          // note: for lines 0 to 3 the orientation of the
                          // line is 'true', if vertex 0 is on the front
                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(0) == middle_vertices[i])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way round then
                                Assert(lines[i]->vertex_index(1) ==
                                         middle_vertices[i],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // note: for lines 4 to 11 (inner lines of the
                          // outer quads) the following holds: the second
                          // vertex of the even lines in standard
                          // orientation is the vertex in the middle of
                          // the quad, whereas for odd lines the first
                          // vertex is the same middle vertex.
                          for (unsigned int i = 4; i < 12; ++i)
                            if (lines[i]->vertex_index((i + 1) % 2) ==
                                middle_vertex_index<dim, spacedim>(
                                  hex->face(1 + i / 4)))
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way
                                // round then
                                Assert(lines[i]->vertex_index(i % 2) ==
                                         (middle_vertex_index<dim, spacedim>(
                                           hex->face(1 + i / 4))),
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }
                          // for the last line the line orientation is
                          // always true, since it was just constructed
                          // that way
                          line_orientation[12] = true;

                          // set up the 4 quads, numbered as follows (left
                          // quad numbering, right line numbering
                          // extracted from above), the drawings denote
                          // middle planes
                          //
                          //      *          *
                          //     /|         /|
                          //    / |        3 9
                          //  y/  *       /  *
                          //  * 3/|      *  /|
                          //  | / |x     5 12|8
                          //  |/  *      |/  *
                          //  * 2/       *  /
                          //  | /        4 2
                          //  |/         |/
                          //  *          *
                          //
                          //       y
                          //      *----*----*      *-10-*-11-*
                          //     /    /    /      /    /    /
                          //    / 0  /  1 /      0    12   1
                          //   /    /    /      /    /    /
                          //  *----*----*x     *--6-*--7-*

                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[0],
                             line_indices[12],
                             line_indices[6],
                             line_indices[10]});
                          new_quads[1]->set_bounding_object_indices(
                            {line_indices[12],
                             line_indices[1],
                             line_indices[7],
                             line_indices[11]});
                          new_quads[2]->set_bounding_object_indices(
                            {line_indices[4],
                             line_indices[8],
                             line_indices[2],
                             line_indices[12]});
                          new_quads[3]->set_bounding_object_indices(
                            {line_indices[5],
                             line_indices[9],
                             line_indices[12],
                             line_indices[3]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[0]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[6]);
                          new_quads[0]->set_line_orientation(
                            3, line_orientation[10]);

                          new_quads[1]->set_line_orientation(
                            1, line_orientation[1]);
                          new_quads[1]->set_line_orientation(
                            2, line_orientation[7]);
                          new_quads[1]->set_line_orientation(
                            3, line_orientation[11]);

                          new_quads[2]->set_line_orientation(
                            0, line_orientation[4]);
                          new_quads[2]->set_line_orientation(
                            1, line_orientation[8]);
                          new_quads[2]->set_line_orientation(
                            2, line_orientation[2]);

                          new_quads[3]->set_line_orientation(
                            0, line_orientation[5]);
                          new_quads[3]->set_line_orientation(
                            1, line_orientation[9]);
                          new_quads[3]->set_line_orientation(
                            3, line_orientation[3]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    / | x
                          //   /3 * *-------*      *----*----*
                          //  *  /| |       |     /    /    /
                          //  | / | |       |    /  0 /  1 /
                          //  |/  * |       |   /    /    /
                          //  * 2/  *-------*y *----*----*x
                          //  | /
                          //  |/
                          //  *
                          //
                          // children of the faces
                          // of the old hex
                          //      *---*---*        *---*---*
                          //     /|13 | 15|       /   /   /|
                          //    / |   |   |      /18 / 19/ |
                          //   /  *---*---*     /   /   /  *
                          //  * 5/|   |   |    *---*---* 7/|
                          //  | / |12 | 14|    | 9 | 11| / |
                          //  |/4 *---*---*    |   |   |/6 *
                          //  *  /   /   /     *---*---*  /
                          //  | /16 / 17/      |   |   | /
                          //  |/   /   /       | 8 | 10|/
                          //  *---*---*        *---*---*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[20] = {
                            new_quads[0]->index(), // 0
                            new_quads[1]->index(),
                            new_quads[2]->index(),
                            new_quads[3]->index(),

                            hex->face(0)->child_index(
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]), // 4
                            hex->face(0)->child_index(
                              1 -
                              child_at_origin[hex->face(0)->refinement_case() -
                                              1][f_fl[0]][f_ro[0]]),

                            hex->face(1)->child_index(
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]), // 6
                            hex->face(1)->child_index(
                              1 -
                              child_at_origin[hex->face(1)->refinement_case() -
                                              1][f_fl[1]][f_ro[1]]),

                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[2], f_fl[2], f_ro[2])), // 8
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[2], f_fl[2], f_ro[2])),
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[2], f_fl[2], f_ro[2])),
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[2], f_fl[2], f_ro[2])),

                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[3], f_fl[3], f_ro[3])), // 12
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[3], f_fl[3], f_ro[3])),
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[3], f_fl[3], f_ro[3])),
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[3], f_fl[3], f_ro[3])),

                            hex->face(4)->child_index(
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]), // 16
                            hex->face(4)->child_index(
                              1 -
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]),

                            hex->face(5)->child_index(
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]]), // 18
                            hex->face(5)->child_index(
                              1 -
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]])};

                          // due to the exchange of x and y for the front
                          // and back face, we order the children
                          // according to
                          //
                          // *---*---*
                          // | 1 | 3 |
                          // *---*---*
                          // | 0 | 2 |
                          // *---*---*
                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[4],
                             quad_indices[2],
                             quad_indices[8],
                             quad_indices[12],
                             quad_indices[16],
                             quad_indices[0]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[5],
                             quad_indices[3],
                             quad_indices[9],
                             quad_indices[13],
                             quad_indices[0],
                             quad_indices[18]});
                          new_hexes[2]->set_bounding_object_indices(
                            {quad_indices[2],
                             quad_indices[6],
                             quad_indices[10],
                             quad_indices[14],
                             quad_indices[17],
                             quad_indices[1]});
                          new_hexes[3]->set_bounding_object_indices(
                            {quad_indices[3],
                             quad_indices[7],
                             quad_indices[11],
                             quad_indices[15],
                             quad_indices[1],
                             quad_indices[19]});
                          break;
                        }

                      case RefinementCase<dim>::cut_yz:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_yz
                          //
                          // the refined cube will look like this:
                          //
                          //        *---------*
                          //       /         /|
                          //      *---------* |
                          //     /         /| |
                          //    *---------* |/|
                          //    |         | * |
                          //    |         |/| *
                          //    *---------* |/
                          //    |         | *
                          //    |         |/
                          //    *---------*
                          //

                          // first, create the new
                          // internal line
                          new_lines[0]->set_bounding_object_indices(

                            {middle_vertex_index<dim, spacedim>(hex->face(0)),
                             middle_vertex_index<dim, spacedim>(hex->face(1))});

                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering: (note that face 0 and 1 each are
                          // shown twice for better readability)

                          // face 0: left plane
                          //       *            *
                          //      /|           /|
                          //     * |          * |
                          //    /| *         /| *
                          //   * 5/|        * |7|
                          //   | * |        | * |
                          //   |/| *        |6| *
                          //   * 4/         * |/
                          //   | *          | *
                          //   |/           |/
                          //   *            *
                          // face 1: right plane
                          //       *            *
                          //      /|           /|
                          //     * |          * |
                          //    /| *         /| *
                          //   * 9/|        * |11
                          //   | * |        | * |
                          //   |/| *        |10 *
                          //   * 8/         * |/
                          //   | *          | *
                          //   |/           |/
                          //   *            *
                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *-------*
                          //   |       |
                          //   *---0---*
                          //   |       |
                          //   *-------*
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //   *-------*
                          //   |       |
                          //   *---1---*
                          //   |       |
                          //   *-------*
                          // face 4: bottom plane
                          //       *-------*
                          //      /       /
                          //     *---2---*
                          //    /       /
                          //   *-------*
                          // face 5: top plane
                          //       *-------*
                          //      /       /
                          //     *---3---*
                          //    /       /
                          //   *-------*
                          // middle planes
                          //     *-------*   *-------*
                          //    /       /    |       |
                          //   *---12--*     |       |
                          //  /       /      |       |
                          // *-------*       *-------*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<
                            dim,
                            spacedim>::raw_line_iterator lines[13] = {
                            hex->face(2)->child(0)->line(
                              (hex->face(2)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 0
                            hex->face(3)->child(0)->line(
                              (hex->face(3)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 1
                            hex->face(4)->child(0)->line(
                              (hex->face(4)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 2
                            hex->face(5)->child(0)->line(
                              (hex->face(5)->refinement_case() ==
                               RefinementCase<2>::cut_x) ?
                                1 :
                                3), // 3

                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[0], f_fl[0], f_ro[0])), // 4
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[0], f_fl[0], f_ro[0])), // 5
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[0], f_fl[0], f_ro[0])), // 6
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[0], f_fl[0], f_ro[0])), // 7

                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[1], f_fl[1], f_ro[1])), // 8
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[1], f_fl[1], f_ro[1])), // 9
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[1], f_fl[1], f_ro[1])), // 10
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[1], f_fl[1], f_ro[1])), // 11

                            new_lines[0] // 12
                          };

                          unsigned int line_indices[13];

                          for (unsigned int i = 0; i < 13; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[13];

                          // the middle vertices of the lines of our front
                          // face
                          const unsigned int middle_vertices[4] = {
                            hex->line(8)->child(0)->vertex_index(1),
                            hex->line(10)->child(0)->vertex_index(1),
                            hex->line(0)->child(0)->vertex_index(1),
                            hex->line(4)->child(0)->vertex_index(1),
                          };

                          // note: for lines 0 to 3 the orientation of the
                          // line is 'true', if vertex 0 is on the front
                          for (unsigned int i = 0; i < 4; ++i)
                            if (lines[i]->vertex_index(0) == middle_vertices[i])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way round then
                                Assert(lines[i]->vertex_index(1) ==
                                         middle_vertices[i],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }

                          // note: for lines 4 to 11 (inner lines of the
                          // outer quads) the following holds: the second
                          // vertex of the even lines in standard
                          // orientation is the vertex in the middle of
                          // the quad, whereas for odd lines the first
                          // vertex is the same middle vertex.
                          for (unsigned int i = 4; i < 12; ++i)
                            if (lines[i]->vertex_index((i + 1) % 2) ==
                                middle_vertex_index<dim, spacedim>(
                                  hex->face(i / 4 - 1)))
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way
                                // round then
                                Assert(lines[i]->vertex_index(i % 2) ==
                                         (middle_vertex_index<dim, spacedim>(
                                           hex->face(i / 4 - 1))),
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }
                          // for the last line the line orientation is
                          // always true, since it was just constructed
                          // that way
                          line_orientation[12] = true;

                          // set up the 4 quads, numbered as follows (left
                          // quad numbering, right line numbering
                          // extracted from above)
                          //
                          //  x
                          //  *-------*      *---3---*
                          //  |   3   |      5       9
                          //  *-------*      *---12--*
                          //  |   2   |      4       8
                          //  *-------*y     *---2---*
                          //
                          //       y
                          //      *---------*      *----1----*
                          //     /    1    /      7         11
                          //    *---------*      *----12---*
                          //   /    0    /      6         10
                          //  *---------*x     *----0----*

                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[6],
                             line_indices[10],
                             line_indices[0],
                             line_indices[12]});
                          new_quads[1]->set_bounding_object_indices(
                            {line_indices[7],
                             line_indices[11],
                             line_indices[12],
                             line_indices[1]});
                          new_quads[2]->set_bounding_object_indices(
                            {line_indices[2],
                             line_indices[12],
                             line_indices[4],
                             line_indices[8]});
                          new_quads[3]->set_bounding_object_indices(
                            {line_indices[12],
                             line_indices[3],
                             line_indices[5],
                             line_indices[9]});

                          new_quads[0]->set_line_orientation(
                            0, line_orientation[6]);
                          new_quads[0]->set_line_orientation(
                            1, line_orientation[10]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[0]);

                          new_quads[1]->set_line_orientation(
                            0, line_orientation[7]);
                          new_quads[1]->set_line_orientation(
                            1, line_orientation[11]);
                          new_quads[1]->set_line_orientation(
                            3, line_orientation[1]);

                          new_quads[2]->set_line_orientation(
                            0, line_orientation[2]);
                          new_quads[2]->set_line_orientation(
                            2, line_orientation[4]);
                          new_quads[2]->set_line_orientation(
                            3, line_orientation[8]);

                          new_quads[3]->set_line_orientation(
                            1, line_orientation[3]);
                          new_quads[3]->set_line_orientation(
                            2, line_orientation[5]);
                          new_quads[3]->set_line_orientation(
                            3, line_orientation[9]);

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    / | x
                          //   /  | *-------*      *---------*
                          //  *   | |   3   |     /    1    /
                          //  |   | *-------*    *---------*
                          //  |   * |   2   |   /    0    /
                          //  |  /  *-------*y *---------*x
                          //  | /
                          //  |/
                          //  *
                          //
                          // children of the faces
                          // of the old hex
                          //      *-------*        *-------*
                          //     /|       |       /  19   /|
                          //    * |  15   |      *-------* |
                          //   /|7*-------*     /  18   /|11
                          //  * |/|       |    *-------* |/|
                          //  |6* |  14   |    |       10* |
                          //  |/|5*-------*    |  13   |/|9*
                          //  * |/  17   /     *-------* |/
                          //  |4*-------*      |       |8*
                          //  |/  16   /       |  12   |/
                          //  *-------*        *-------*
                          //
                          // note that we have to take care of the
                          // orientation of faces.
                          const int quad_indices[20] = {
                            new_quads[0]->index(), // 0
                            new_quads[1]->index(),
                            new_quads[2]->index(),
                            new_quads[3]->index(),

                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[0], f_fl[0], f_ro[0])), // 4
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[0], f_fl[0], f_ro[0])),
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[0], f_fl[0], f_ro[0])),
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[0], f_fl[0], f_ro[0])),

                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[1], f_fl[1], f_ro[1])), // 8
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[1], f_fl[1], f_ro[1])),
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[1], f_fl[1], f_ro[1])),
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[1], f_fl[1], f_ro[1])),

                            hex->face(2)->child_index(
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]), // 12
                            hex->face(2)->child_index(
                              1 -
                              child_at_origin[hex->face(2)->refinement_case() -
                                              1][f_fl[2]][f_ro[2]]),

                            hex->face(3)->child_index(
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]), // 14
                            hex->face(3)->child_index(
                              1 -
                              child_at_origin[hex->face(3)->refinement_case() -
                                              1][f_fl[3]][f_ro[3]]),

                            hex->face(4)->child_index(
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]), // 16
                            hex->face(4)->child_index(
                              1 -
                              child_at_origin[hex->face(4)->refinement_case() -
                                              1][f_fl[4]][f_ro[4]]),

                            hex->face(5)->child_index(
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]]), // 18
                            hex->face(5)->child_index(
                              1 -
                              child_at_origin[hex->face(5)->refinement_case() -
                                              1][f_fl[5]][f_ro[5]])};

                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[4],
                             quad_indices[8],
                             quad_indices[12],
                             quad_indices[2],
                             quad_indices[16],
                             quad_indices[0]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[5],
                             quad_indices[9],
                             quad_indices[2],
                             quad_indices[14],
                             quad_indices[17],
                             quad_indices[1]});
                          new_hexes[2]->set_bounding_object_indices(
                            {quad_indices[6],
                             quad_indices[10],
                             quad_indices[13],
                             quad_indices[3],
                             quad_indices[0],
                             quad_indices[18]});
                          new_hexes[3]->set_bounding_object_indices(
                            {quad_indices[7],
                             quad_indices[11],
                             quad_indices[3],
                             quad_indices[15],
                             quad_indices[1],
                             quad_indices[19]});
                          break;
                        }

                      case RefinementCase<dim>::cut_xyz:
                        {
                          //----------------------------
                          //
                          //     RefinementCase<dim>::cut_xyz
                          //     isotropic refinement
                          //
                          // the refined cube will look
                          // like this:
                          //
                          //        *----*----*
                          //       /    /    /|
                          //      *----*----* |
                          //     /    /    /| *
                          //    *----*----* |/|
                          //    |    |    | * |
                          //    |    |    |/| *
                          //    *----*----* |/
                          //    |    |    | *
                          //    |    |    |/
                          //    *----*----*
                          //

                          // find the next unused vertex and set it
                          // appropriately
                          while (
                            triangulation.vertices_used[next_unused_vertex] ==
                            true)
                            ++next_unused_vertex;
                          Assert(
                            next_unused_vertex < triangulation.vertices.size(),
                            ExcMessage(
                              "Internal error: During refinement, the triangulation wants to access an element of the 'vertices' array but it turns out that the array is not large enough."));
                          triangulation.vertices_used[next_unused_vertex] =
                            true;

                          // the new vertex is definitely in the interior,
                          // so we need not worry about the
                          // boundary. However we need to worry about
                          // Manifolds. Let the cell compute its own
                          // center, by querying the underlying manifold
                          // object.
                          triangulation.vertices[next_unused_vertex] =
                            hex->center(true, true);

                          // set the data of the six lines.  first collect
                          // the indices of the seven vertices (consider
                          // the two planes to be crossed to form the
                          // planes cutting the hex in two vertically and
                          // horizontally)
                          //
                          //     *--3--*   *--5--*
                          //    /  /  /    |  |  |
                          //   0--6--1     0--6--1
                          //  /  /  /      |  |  |
                          // *--2--*       *--4--*
                          // the lines are numbered
                          // as follows:
                          //     *--*--*   *--*--*
                          //    /  1  /    |  5  |
                          //   *2-*-3*     *2-*-3*
                          //  /  0  /      |  4  |
                          // *--*--*       *--*--*
                          //
                          const unsigned int vertex_indices[7] = {
                            middle_vertex_index<dim, spacedim>(hex->face(0)),
                            middle_vertex_index<dim, spacedim>(hex->face(1)),
                            middle_vertex_index<dim, spacedim>(hex->face(2)),
                            middle_vertex_index<dim, spacedim>(hex->face(3)),
                            middle_vertex_index<dim, spacedim>(hex->face(4)),
                            middle_vertex_index<dim, spacedim>(hex->face(5)),
                            next_unused_vertex};

                          new_lines[0]->set_bounding_object_indices(
                            {vertex_indices[2], vertex_indices[6]});
                          new_lines[1]->set_bounding_object_indices(
                            {vertex_indices[6], vertex_indices[3]});
                          new_lines[2]->set_bounding_object_indices(
                            {vertex_indices[0], vertex_indices[6]});
                          new_lines[3]->set_bounding_object_indices(
                            {vertex_indices[6], vertex_indices[1]});
                          new_lines[4]->set_bounding_object_indices(
                            {vertex_indices[4], vertex_indices[6]});
                          new_lines[5]->set_bounding_object_indices(
                            {vertex_indices[6], vertex_indices[5]});

                          // again, first collect some data about the
                          // indices of the lines, with the following
                          // numbering: (note that face 0 and 1 each are
                          // shown twice for better readability)

                          // face 0: left plane
                          //       *            *
                          //      /|           /|
                          //     * |          * |
                          //    /| *         /| *
                          //   * 1/|        * |3|
                          //   | * |        | * |
                          //   |/| *        |2| *
                          //   * 0/         * |/
                          //   | *          | *
                          //   |/           |/
                          //   *            *
                          // face 1: right plane
                          //       *            *
                          //      /|           /|
                          //     * |          * |
                          //    /| *         /| *
                          //   * 5/|        * |7|
                          //   | * |        | * |
                          //   |/| *        |6| *
                          //   * 4/         * |/
                          //   | *          | *
                          //   |/           |/
                          //   *            *
                          // face 2: front plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   11  |
                          //   *-8-*-9-*
                          //   |   10  |
                          //   *---*---*
                          // face 3: back plane
                          //   (note: x,y exchanged)
                          //   *---*---*
                          //   |   15  |
                          //   *12-*-13*
                          //   |   14  |
                          //   *---*---*
                          // face 4: bottom plane
                          //       *---*---*
                          //      /  17   /
                          //     *18-*-19*
                          //    /   16  /
                          //   *---*---*
                          // face 5: top plane
                          //       *---*---*
                          //      /  21   /
                          //     *22-*-23*
                          //    /   20  /
                          //   *---*---*
                          // middle planes
                          //     *---*---*   *---*---*
                          //    /  25   /    |   29  |
                          //   *26-*-27*     *26-*-27*
                          //  /   24  /      |   28  |
                          // *---*---*       *---*---*

                          // set up a list of line iterators first. from
                          // this, construct lists of line_indices and
                          // line orientations later on
                          const typename Triangulation<
                            dim,
                            spacedim>::raw_line_iterator lines[30] = {
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[0], f_fl[0], f_ro[0])), // 0
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[0], f_fl[0], f_ro[0])), // 1
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[0], f_fl[0], f_ro[0])), // 2
                            hex->face(0)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[0], f_fl[0], f_ro[0]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[0], f_fl[0], f_ro[0])), // 3

                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[1], f_fl[1], f_ro[1])), // 4
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[1], f_fl[1], f_ro[1])), // 5
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[1], f_fl[1], f_ro[1])), // 6
                            hex->face(1)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[1], f_fl[1], f_ro[1]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[1], f_fl[1], f_ro[1])), // 7

                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[2], f_fl[2], f_ro[2])), // 8
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[2], f_fl[2], f_ro[2])), // 9
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[2], f_fl[2], f_ro[2])), // 10
                            hex->face(2)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[2], f_fl[2], f_ro[2]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[2], f_fl[2], f_ro[2])), // 11

                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[3], f_fl[3], f_ro[3])), // 12
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[3], f_fl[3], f_ro[3])), // 13
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[3], f_fl[3], f_ro[3])), // 14
                            hex->face(3)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[3], f_fl[3], f_ro[3]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[3], f_fl[3], f_ro[3])), // 15

                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[4], f_fl[4], f_ro[4])), // 16
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[4], f_fl[4], f_ro[4])), // 17
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[4], f_fl[4], f_ro[4])), // 18
                            hex->face(4)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[4], f_fl[4], f_ro[4]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[4], f_fl[4], f_ro[4])), // 19

                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  1, f_or[5], f_fl[5], f_ro[5])), // 20
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  0, f_or[5], f_fl[5], f_ro[5])), // 21
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  0, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  3, f_or[5], f_fl[5], f_ro[5])), // 22
                            hex->face(5)
                              ->isotropic_child(
                                GeometryInfo<dim>::standard_to_real_face_vertex(
                                  3, f_or[5], f_fl[5], f_ro[5]))
                              ->line(
                                GeometryInfo<dim>::standard_to_real_face_line(
                                  2, f_or[5], f_fl[5], f_ro[5])), // 23

                            new_lines[0], // 24
                            new_lines[1], // 25
                            new_lines[2], // 26
                            new_lines[3], // 27
                            new_lines[4], // 28
                            new_lines[5]  // 29
                          };

                          unsigned int line_indices[30];
                          for (unsigned int i = 0; i < 30; ++i)
                            line_indices[i] = lines[i]->index();

                          // the orientation of lines for the inner quads
                          // is quite tricky. as these lines are newly
                          // created ones and thus have no parents, they
                          // cannot inherit this property. set up an array
                          // and fill it with the respective values
                          bool line_orientation[30];

                          // note: for the first 24 lines (inner lines of
                          // the outer quads) the following holds: the
                          // second vertex of the even lines in standard
                          // orientation is the vertex in the middle of
                          // the quad, whereas for odd lines the first
                          // vertex is the same middle vertex.
                          for (unsigned int i = 0; i < 24; ++i)
                            if (lines[i]->vertex_index((i + 1) % 2) ==
                                vertex_indices[i / 4])
                              line_orientation[i] = true;
                            else
                              {
                                // it must be the other way
                                // round then
                                Assert(lines[i]->vertex_index(i % 2) ==
                                         vertex_indices[i / 4],
                                       ExcInternalError());
                                line_orientation[i] = false;
                              }
                          // for the last 6 lines the line orientation is
                          // always true, since they were just constructed
                          // that way
                          for (unsigned int i = 24; i < 30; ++i)
                            line_orientation[i] = true;

                          // set up the 12 quads, numbered as follows
                          // (left quad numbering, right line numbering
                          // extracted from above)
                          //
                          //      *          *
                          //     /|        21|
                          //    * |        * 15
                          //  y/|3*      20| *
                          //  * |/|      * |/|
                          //  |2* |x    11 * 14
                          //  |/|1*      |/| *
                          //  * |/       * |17
                          //  |0*       10 *
                          //  |/         |16
                          //  *          *
                          //
                          //  x
                          //  *---*---*      *22-*-23*
                          //  | 5 | 7 |      1  29   5
                          //  *---*---*      *26-*-27*
                          //  | 4 | 6 |      0  28   4
                          //  *---*---*y     *18-*-19*
                          //
                          //       y
                          //      *----*----*      *-12-*-13-*
                          //     / 10 / 11 /      3   25    7
                          //    *----*----*      *-26-*-27-*
                          //   / 8  / 9  /      2   24    6
                          //  *----*----*x     *--8-*--9-*

                          new_quads[0]->set_bounding_object_indices(
                            {line_indices[10],
                             line_indices[28],
                             line_indices[16],
                             line_indices[24]});
                          new_quads[1]->set_bounding_object_indices(
                            {line_indices[28],
                             line_indices[14],
                             line_indices[17],
                             line_indices[25]});
                          new_quads[2]->set_bounding_object_indices(
                            {line_indices[11],
                             line_indices[29],
                             line_indices[24],
                             line_indices[20]});
                          new_quads[3]->set_bounding_object_indices(
                            {line_indices[29],
                             line_indices[15],
                             line_indices[25],
                             line_indices[21]});
                          new_quads[4]->set_bounding_object_indices(
                            {line_indices[18],
                             line_indices[26],
                             line_indices[0],
                             line_indices[28]});
                          new_quads[5]->set_bounding_object_indices(
                            {line_indices[26],
                             line_indices[22],
                             line_indices[1],
                             line_indices[29]});
                          new_quads[6]->set_bounding_object_indices(
                            {line_indices[19],
                             line_indices[27],
                             line_indices[28],
                             line_indices[4]});
                          new_quads[7]->set_bounding_object_indices(
                            {line_indices[27],
                             line_indices[23],
                             line_indices[29],
                             line_indices[5]});
                          new_quads[8]->set_bounding_object_indices(
                            {line_indices[2],
                             line_indices[24],
                             line_indices[8],
                             line_indices[26]});
                          new_quads[9]->set_bounding_object_indices(
                            {line_indices[24],
                             line_indices[6],
                             line_indices[9],
                             line_indices[27]});
                          new_quads[10]->set_bounding_object_indices(
                            {line_indices[3],
                             line_indices[25],
                             line_indices[26],
                             line_indices[12]});
                          new_quads[11]->set_bounding_object_indices(
                            {line_indices[25],
                             line_indices[7],
                             line_indices[27],
                             line_indices[13]});

                          // now reset the line_orientation flags of outer
                          // lines as they cannot be set in a loop (at
                          // least not easily)
                          new_quads[0]->set_line_orientation(
                            0, line_orientation[10]);
                          new_quads[0]->set_line_orientation(
                            2, line_orientation[16]);

                          new_quads[1]->set_line_orientation(
                            1, line_orientation[14]);
                          new_quads[1]->set_line_orientation(
                            2, line_orientation[17]);

                          new_quads[2]->set_line_orientation(
                            0, line_orientation[11]);
                          new_quads[2]->set_line_orientation(
                            3, line_orientation[20]);

                          new_quads[3]->set_line_orientation(
                            1, line_orientation[15]);
                          new_quads[3]->set_line_orientation(
                            3, line_orientation[21]);

                          new_quads[4]->set_line_orientation(
                            0, line_orientation[18]);
                          new_quads[4]->set_line_orientation(
                            2, line_orientation[0]);

                          new_quads[5]->set_line_orientation(
                            1, line_orientation[22]);
                          new_quads[5]->set_line_orientation(
                            2, line_orientation[1]);

                          new_quads[6]->set_line_orientation(
                            0, line_orientation[19]);
                          new_quads[6]->set_line_orientation(
                            3, line_orientation[4]);

                          new_quads[7]->set_line_orientation(
                            1, line_orientation[23]);
                          new_quads[7]->set_line_orientation(
                            3, line_orientation[5]);

                          new_quads[8]->set_line_orientation(
                            0, line_orientation[2]);
                          new_quads[8]->set_line_orientation(
                            2, line_orientation[8]);

                          new_quads[9]->set_line_orientation(
                            1, line_orientation[6]);
                          new_quads[9]->set_line_orientation(
                            2, line_orientation[9]);

                          new_quads[10]->set_line_orientation(
                            0, line_orientation[3]);
                          new_quads[10]->set_line_orientation(
                            3, line_orientation[12]);

                          new_quads[11]->set_line_orientation(
                            1, line_orientation[7]);
                          new_quads[11]->set_line_orientation(
                            3, line_orientation[13]);

                          //-------------------------------
                          // create the eight new hexes
                          //
                          // again first collect some data.  here, we need
                          // the indices of a whole lotta quads.

                          // the quads are numbered as follows:
                          //
                          // planes in the interior of the old hex:
                          //
                          //      *
                          //     /|
                          //    * |
                          //   /|3*  *---*---*      *----*----*
                          //  * |/|  | 5 | 7 |     / 10 / 11 /
                          //  |2* |  *---*---*    *----*----*
                          //  |/|1*  | 4 | 6 |   / 8  / 9  /
                          //  * |/   *---*---*y *----*----*x
                          //  |0*
                          //  |/
                          //  *
                          //
                          // children of the faces
                          // of the old hex
                          //      *-------*        *-------*
                          //     /|25   27|       /34   35/|
                          //    15|       |      /       /19
                          //   /  |       |     /32   33/  |
                          //  *   |24   26|    *-------*18 |
                          //  1413*-------*    |21   23| 17*
                          //  |  /30   31/     |       |  /
                          //  12/       /      |       |16
                          //  |/28   29/       |20   22|/
                          //  *-------*        *-------*
                          //
                          // note that we have to
                          // take care of the
                          // orientation of
                          // faces.
                          const int quad_indices[36] = {
                            new_quads[0]->index(), // 0
                            new_quads[1]->index(),
                            new_quads[2]->index(),
                            new_quads[3]->index(),
                            new_quads[4]->index(),
                            new_quads[5]->index(),
                            new_quads[6]->index(),
                            new_quads[7]->index(),
                            new_quads[8]->index(),
                            new_quads[9]->index(),
                            new_quads[10]->index(),
                            new_quads[11]->index(), // 11

                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[0], f_fl[0], f_ro[0])), // 12
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[0], f_fl[0], f_ro[0])),
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[0], f_fl[0], f_ro[0])),
                            hex->face(0)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[0], f_fl[0], f_ro[0])),

                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[1], f_fl[1], f_ro[1])), // 16
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[1], f_fl[1], f_ro[1])),
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[1], f_fl[1], f_ro[1])),
                            hex->face(1)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[1], f_fl[1], f_ro[1])),

                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[2], f_fl[2], f_ro[2])), // 20
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[2], f_fl[2], f_ro[2])),
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[2], f_fl[2], f_ro[2])),
                            hex->face(2)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[2], f_fl[2], f_ro[2])),

                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[3], f_fl[3], f_ro[3])), // 24
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[3], f_fl[3], f_ro[3])),
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[3], f_fl[3], f_ro[3])),
                            hex->face(3)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[3], f_fl[3], f_ro[3])),

                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[4], f_fl[4], f_ro[4])), // 28
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[4], f_fl[4], f_ro[4])),
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[4], f_fl[4], f_ro[4])),
                            hex->face(4)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[4], f_fl[4], f_ro[4])),

                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                0, f_or[5], f_fl[5], f_ro[5])), // 32
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                1, f_or[5], f_fl[5], f_ro[5])),
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                2, f_or[5], f_fl[5], f_ro[5])),
                            hex->face(5)->isotropic_child_index(
                              GeometryInfo<dim>::standard_to_real_face_vertex(
                                3, f_or[5], f_fl[5], f_ro[5]))};

                          // bottom children
                          new_hexes[0]->set_bounding_object_indices(
                            {quad_indices[12],
                             quad_indices[0],
                             quad_indices[20],
                             quad_indices[4],
                             quad_indices[28],
                             quad_indices[8]});
                          new_hexes[1]->set_bounding_object_indices(
                            {quad_indices[0],
                             quad_indices[16],
                             quad_indices[22],
                             quad_indices[6],
                             quad_indices[29],
                             quad_indices[9]});
                          new_hexes[2]->set_bounding_object_indices(
                            {quad_indices[13],
                             quad_indices[1],
                             quad_indices[4],
                             quad_indices[24],
                             quad_indices[30],
                             quad_indices[10]});
                          new_hexes[3]->set_bounding_object_indices(
                            {quad_indices[1],
                             quad_indices[17],
                             quad_indices[6],
                             quad_indices[26],
                             quad_indices[31],
                             quad_indices[11]});

                          // top children
                          new_hexes[4]->set_bounding_object_indices(
                            {quad_indices[14],
                             quad_indices[2],
                             quad_indices[21],
                             quad_indices[5],
                             quad_indices[8],
                             quad_indices[32]});
                          new_hexes[5]->set_bounding_object_indices(
                            {quad_indices[2],
                             quad_indices[18],
                             quad_indices[23],
                             quad_indices[7],
                             quad_indices[9],
                             quad_indices[33]});
                          new_hexes[6]->set_bounding_object_indices(
                            {quad_indices[15],
                             quad_indices[3],
                             quad_indices[5],
                             quad_indices[25],
                             quad_indices[10],
                             quad_indices[34]});
                          new_hexes[7]->set_bounding_object_indices(
                            {quad_indices[3],
                             quad_indices[19],
                             quad_indices[7],
                             quad_indices[27],
                             quad_indices[11],
                             quad_indices[35]});
                          break;
                        }
                      default:
                        // all refinement cases have been treated, there
                        // only remains
                        // RefinementCase<dim>::no_refinement as
                        // untreated enumeration value. However, in that
                        // case we should have aborted much
                        // earlier. thus we should never get here
                        Assert(false, ExcInternalError());
                        break;
                    } // switch (ref_case)

                  // and set face orientation flags. note that new
                  // faces in the interior of the mother cell always
                  // have a correctly oriented face, but the ones on
                  // the outer faces will inherit this flag
                  //
                  // the flag have been set to true for all faces
                  // initially, now go the other way round and reset
                  // faces that are at the boundary of the mother cube
                  //
                  // the same is true for the face_flip and
                  // face_rotation flags. however, the latter two are
                  // set to false by default as this is the standard
                  // value

                  // loop over all faces and all (relevant) subfaces
                  // of that in order to set the correct values for
                  // face_orientation, face_flip and face_rotation,
                  // which are inherited from the corresponding face
                  // of the mother cube
                  for (const unsigned int f : GeometryInfo<dim>::face_indices())
                    for (unsigned int s = 0;
                         s < std::max(GeometryInfo<dim - 1>::n_children(
                                        GeometryInfo<dim>::face_refinement_case(
                                          ref_case, f)),
                                      1U);
                         ++s)
                      {
                        const unsigned int current_child =
                          GeometryInfo<dim>::child_cell_on_face(
                            ref_case,
                            f,
                            s,
                            f_or[f],
                            f_fl[f],
                            f_ro[f],
                            GeometryInfo<dim>::face_refinement_case(
                              ref_case, f, f_or[f], f_fl[f], f_ro[f]));
                        new_hexes[current_child]->set_face_orientation(f,
                                                                       f_or[f]);
                        new_hexes[current_child]->set_face_flip(f, f_fl[f]);
                        new_hexes[current_child]->set_face_rotation(f, f_ro[f]);
                      }

                  // now see if we have created cells that are
                  // distorted and if so add them to our list
                  if (check_for_distorted_cells &&
                      has_distorted_children<dim, spacedim>(hex))
                    cells_with_distorted_children.distorted_cells.push_back(
                      hex);

                  // note that the refinement flag was already cleared
                  // at the beginning of this loop

                  // inform all listeners that cell refinement is done
                  triangulation.signals.post_refinement_on_cell(hex);
                }
          }

        // clear user data on quads. we used some of this data to
        // indicate anisotropic refinemnt cases on faces. all data
        // should be cleared by now, but the information whether we
        // used indices or pointers is still present. reset it now to
        // enable the user to use whichever they like later on.
        triangulation.faces->quads.clear_user_data();

        // return the list with distorted children
        return cells_with_distorted_children;
      }


      /**
       * At the boundary of the domain, the new point on the face may
       * be far inside the current cell, if the boundary has a strong
       * curvature. If we allow anisotropic refinement here, the
       * resulting cell may be strongly distorted. To prevent this,
       * this function flags such cells for isotropic refinement. It
       * is called automatically from
       * prepare_coarsening_and_refinement().
       *
       * This function does nothing in 1d (therefore the
       * specialization).
       */
      template <int spacedim>
      static void
      prevent_distorted_boundary_cells(Triangulation<1, spacedim> &)
      {}



      template <int dim, int spacedim>
      static void
      prevent_distorted_boundary_cells(
        Triangulation<dim, spacedim> &triangulation)
      {
        // If the codimension is one, we cannot perform this check
        // yet.
        if (spacedim > dim)
          return;

        for (const auto &cell : triangulation.cell_iterators())
          if (cell->at_boundary() && cell->refine_flag_set() &&
              cell->refine_flag_set() !=
                RefinementCase<dim>::isotropic_refinement)
            {
              // The cell is at the boundary and it is flagged for
              // anisotropic refinement. Therefore, we have a closer
              // look
              const RefinementCase<dim> ref_case = cell->refine_flag_set();
              for (const unsigned int face_no :
                   GeometryInfo<dim>::face_indices())
                if (cell->face(face_no)->at_boundary())
                  {
                    // this is the critical face at the boundary.
                    if (GeometryInfo<dim>::face_refinement_case(ref_case,
                                                                face_no) !=
                        RefinementCase<dim - 1>::isotropic_refinement)
                      {
                        // up to now, we do not want to refine this
                        // cell along the face under consideration
                        // here.
                        const typename Triangulation<dim,
                                                     spacedim>::face_iterator
                          face = cell->face(face_no);
                        // the new point on the boundary would be this
                        // one.
                        const Point<spacedim> new_bound = face->center(true);
                        // to check it, transform to the unit cell
                        // with a linear mapping
                        const Point<dim> new_unit =
                          cell->reference_cell()
                            .template get_default_linear_mapping<dim,
                                                                 spacedim>()
                            .transform_real_to_unit_cell(cell, new_bound);

                        // Now, we have to calculate the distance from
                        // the face in the unit cell.

                        // take the correct coordinate direction (0
                        // for faces 0 and 1, 1 for faces 2 and 3, 2
                        // for faces 4 and 5) and subtract the correct
                        // boundary value of the face (0 for faces 0,
                        // 2, and 4; 1 for faces 1, 3 and 5)
                        const double dist =
                          std::fabs(new_unit[face_no / 2] - face_no % 2);

                        // compare this with the empirical value
                        // allowed. if it is too big, flag the face
                        // for isotropic refinement
                        const double allowed = 0.25;

                        if (dist > allowed)
                          cell->flag_for_face_refinement(face_no);
                      } // if flagged for anistropic refinement
                  }     // if (cell->face(face)->at_boundary())
            }           // for all cells
      }


      /**
       * Some dimension dependent stuff for mesh smoothing.
       *
       * At present, this function does nothing in 1d and 2D, but
       * makes sure no two cells with a level difference greater than
       * one share one line in 3D. This is a requirement needed for
       * the interpolation of hanging nodes, since otherwise two steps
       * of interpolation would be necessary. This would make the
       * processes implemented in the @p AffineConstraints class much
       * more complex, since these two steps of interpolation do not
       * commute.
       */
      template <int dim, int spacedim>
      static void
      prepare_refinement_dim_dependent(const Triangulation<dim, spacedim> &)
      {
        Assert(dim < 3,
               ExcMessage("Wrong function called -- there should "
                          "be a specialization."));
      }


      template <int spacedim>
      static void
      prepare_refinement_dim_dependent(
        Triangulation<3, spacedim> &triangulation)
      {
        const unsigned int dim = 3;

        // first clear flags on lines, since we need them to determine
        // which lines will be refined
        triangulation.clear_user_flags_line();

        // also clear flags on hexes, since we need them to mark those
        // cells which are to be coarsened
        triangulation.clear_user_flags_hex();

        // variable to store whether the mesh was changed in the
        // present loop and in the whole process
        bool mesh_changed = false;

        do
          {
            mesh_changed = false;

            // for this following, we need to know which cells are
            // going to be coarsened, if we had to make a
            // decision. the following function sets these flags:
            triangulation.fix_coarsen_flags();


            // flag those lines that are refined and will not be
            // coarsened and those that will be refined
            for (const auto &cell : triangulation.cell_iterators())
              if (cell->refine_flag_set())
                {
                  for (unsigned int line = 0; line < cell->n_lines(); ++line)
                    if (GeometryInfo<dim>::line_refinement_case(
                          cell->refine_flag_set(), line) ==
                        RefinementCase<1>::cut_x)
                      // flag a line, that will be
                      // refined
                      cell->line(line)->set_user_flag();
                }
              else if (cell->has_children() &&
                       !cell->child(0)->coarsen_flag_set())
                {
                  for (unsigned int line = 0; line < cell->n_lines(); ++line)
                    if (GeometryInfo<dim>::line_refinement_case(
                          cell->refinement_case(), line) ==
                        RefinementCase<1>::cut_x)
                      // flag a line, that is refined
                      // and will stay so
                      cell->line(line)->set_user_flag();
                }
              else if (cell->has_children() &&
                       cell->child(0)->coarsen_flag_set())
                cell->set_user_flag();


            // now check whether there are cells with lines that are
            // more than once refined or that will be more than once
            // refined. The first thing should never be the case, in
            // the second case we flag the cell for refinement
            for (typename Triangulation<dim, spacedim>::active_cell_iterator
                   cell = triangulation.last_active();
                 cell != triangulation.end();
                 --cell)
              for (unsigned int line = 0; line < cell->n_lines(); ++line)
                {
                  if (cell->line(line)->has_children())
                    {
                      // if this line is refined, its children should
                      // not have further children
                      //
                      // however, if any of the children is flagged
                      // for further refinement, we need to refine
                      // this cell also (at least, if the cell is not
                      // already flagged)
                      bool offending_line_found = false;

                      for (unsigned int c = 0; c < 2; ++c)
                        {
                          Assert(cell->line(line)->child(c)->has_children() ==
                                   false,
                                 ExcInternalError());

                          if (cell->line(line)->child(c)->user_flag_set() &&
                              (GeometryInfo<dim>::line_refinement_case(
                                 cell->refine_flag_set(), line) ==
                               RefinementCase<1>::no_refinement))
                            {
                              // tag this cell for refinement
                              cell->clear_coarsen_flag();
                              // if anisotropic coarsening is allowed:
                              // extend the refine_flag in the needed
                              // direction, else set refine_flag
                              // (isotropic)
                              if (triangulation.smooth_grid &
                                  Triangulation<dim, spacedim>::
                                    allow_anisotropic_smoothing)
                                cell->flag_for_line_refinement(line);
                              else
                                cell->set_refine_flag();

                              for (unsigned int l = 0; l < cell->n_lines(); ++l)
                                if (GeometryInfo<dim>::line_refinement_case(
                                      cell->refine_flag_set(), line) ==
                                    RefinementCase<1>::cut_x)
                                  // flag a line, that will be refined
                                  cell->line(l)->set_user_flag();

                              // note that we have changed the grid
                              offending_line_found = true;

                              // it may save us several loop
                              // iterations if we flag all lines of
                              // this cell now (and not at the outset
                              // of the next iteration) for refinement
                              for (unsigned int l = 0; l < cell->n_lines(); ++l)
                                if (!cell->line(l)->has_children() &&
                                    (GeometryInfo<dim>::line_refinement_case(
                                       cell->refine_flag_set(), l) !=
                                     RefinementCase<1>::no_refinement))
                                  cell->line(l)->set_user_flag();

                              break;
                            }
                        }

                      if (offending_line_found)
                        {
                          mesh_changed = true;
                          break;
                        }
                    }
                }


            // there is another thing here: if any of the lines will
            // be refined, then we may not coarsen the present cell
            // similarly, if any of the lines *is* already refined, we
            // may not coarsen the current cell. however, there's a
            // catch: if the line is refined, but the cell behind it
            // is going to be coarsened, then the situation
            // changes. if we forget this second condition, the
            // refine_and_coarsen_3d test will start to fail. note
            // that to know which cells are going to be coarsened, the
            // call for fix_coarsen_flags above is necessary
            for (typename Triangulation<dim, spacedim>::cell_iterator cell =
                   triangulation.last();
                 cell != triangulation.end();
                 --cell)
              {
                if (cell->user_flag_set())
                  for (unsigned int line = 0; line < cell->n_lines(); ++line)
                    if (cell->line(line)->has_children() &&
                        (cell->line(line)->child(0)->user_flag_set() ||
                         cell->line(line)->child(1)->user_flag_set()))
                      {
                        for (unsigned int c = 0; c < cell->n_children(); ++c)
                          cell->child(c)->clear_coarsen_flag();
                        cell->clear_user_flag();
                        for (unsigned int l = 0; l < cell->n_lines(); ++l)
                          if (GeometryInfo<dim>::line_refinement_case(
                                cell->refinement_case(), l) ==
                              RefinementCase<1>::cut_x)
                            // flag a line, that is refined
                            // and will stay so
                            cell->line(l)->set_user_flag();
                        mesh_changed = true;
                        break;
                      }
              }
          }
        while (mesh_changed == true);
      }



      /**
       * Helper function for @p fix_coarsen_flags. Return whether
       * coarsening of this cell is allowed.  Coarsening can be
       * forbidden if the neighboring cells are or will be refined
       * twice along the common face.
       */
      template <int dim, int spacedim>
      static bool
      coarsening_allowed(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      {
        // in 1d, coarsening is always allowed since we don't enforce
        // the 2:1 constraint there
        if (dim == 1)
          return true;

        const RefinementCase<dim> ref_case = cell->refinement_case();
        for (unsigned int n : GeometryInfo<dim>::face_indices())
          {
            // if the cell is not refined along that face, coarsening
            // will not change anything, so do nothing. the same
            // applies, if the face is at the boandary
            const RefinementCase<dim - 1> face_ref_case =
              GeometryInfo<dim>::face_refinement_case(cell->refinement_case(),
                                                      n);

            const unsigned int n_subfaces =
              GeometryInfo<dim - 1>::n_children(face_ref_case);

            if (n_subfaces == 0 || cell->at_boundary(n))
              continue;
            for (unsigned int c = 0; c < n_subfaces; ++c)
              {
                const typename Triangulation<dim, spacedim>::cell_iterator
                  child = cell->child(
                    GeometryInfo<dim>::child_cell_on_face(ref_case, n, c));

                const typename Triangulation<dim, spacedim>::cell_iterator
                  child_neighbor = child->neighbor(n);
                if (!child->neighbor_is_coarser(n))
                  // in 2d, if the child's neighbor is coarser, then
                  // it has no children. however, in 3d it might be
                  // otherwise. consider for example, that our face
                  // might be refined with cut_x, but the neighbor is
                  // refined with cut_xy at that face. then the
                  // neighbor pointers of the children of our cell
                  // will point to the common neighbor cell, not to
                  // its children. what we really want to know in the
                  // following is, whether the neighbor cell is
                  // refined twice with reference to our cell.  that
                  // only has to be asked, if the child's neighbor is
                  // not a coarser one.
                  if ((child_neighbor->has_children() &&
                       !child_neighbor->user_flag_set()) ||
                      // neighbor has children, which are further
                      // refined along the face, otherwise something
                      // went wrong in the construction of neighbor
                      // pointers.  then only allow coarsening if this
                      // neighbor will be coarsened as well
                      // (user_pointer is set).  the same applies, if
                      // the neighbors children are not refined but
                      // will be after refinement
                      child_neighbor->refine_flag_set())
                    return false;
              }
          }
        return true;
      }
    };


    /**
     * Same as above but for mixed meshes (and simplex meshes).
     */
    struct ImplementationMixedMesh
    {
      template <int spacedim>
      static void
      update_neighbors(Triangulation<1, spacedim> &)
      {}

      template <int dim, int spacedim>
      void static update_neighbors(Triangulation<dim, spacedim> &triangulation)
      {
        std::vector<std::pair<unsigned int, unsigned int>> adjacent_cells(
          2 * triangulation.n_raw_faces(),
          {numbers::invalid_unsigned_int, numbers::invalid_unsigned_int});

        const auto set_entry = [&](const auto &face_index, const auto &cell) {
          const std::pair<unsigned int, unsigned int> cell_pair = {
            cell->level(), cell->index()};
          unsigned int index;

          if (adjacent_cells[2 * face_index].first ==
                numbers::invalid_unsigned_int &&
              adjacent_cells[2 * face_index].second ==
                numbers::invalid_unsigned_int)
            {
              index = 2 * face_index + 0;
            }
          else
            {
              Assert(((adjacent_cells[2 * face_index + 1].first ==
                       numbers::invalid_unsigned_int) &&
                      (adjacent_cells[2 * face_index + 1].second ==
                       numbers::invalid_unsigned_int)),
                     ExcNotImplemented());
              index = 2 * face_index + 1;
            }

          adjacent_cells[index] = cell_pair;
        };

        const auto get_entry =
          [&](const auto &face_index,
              const auto &cell) -> TriaIterator<CellAccessor<dim, spacedim>> {
          auto test = adjacent_cells[2 * face_index];

          if (test == std::pair<unsigned int, unsigned int>(cell->level(),
                                                            cell->index()))
            test = adjacent_cells[2 * face_index + 1];

          if ((test.first != numbers::invalid_unsigned_int) &&
              (test.second != numbers::invalid_unsigned_int))
            return TriaIterator<CellAccessor<dim, spacedim>>(&triangulation,
                                                             test.first,
                                                             test.second);
          else
            return typename Triangulation<dim, spacedim>::cell_iterator();
        };

        for (const auto &cell : triangulation.cell_iterators())
          for (const auto &face : cell->face_iterators())
            {
              set_entry(face->index(), cell);

              if (cell->is_active() && face->has_children())
                for (unsigned int c = 0; c < face->n_children(); ++c)
                  set_entry(face->child(c)->index(), cell);
            }

        for (const auto &cell : triangulation.cell_iterators())
          for (auto f : cell->face_indices())
            cell->set_neighbor(f, get_entry(cell->face(f)->index(), cell));
      }

      template <int dim, int spacedim>
      static void
      delete_children(
        Triangulation<dim, spacedim> &                        triangulation,
        typename Triangulation<dim, spacedim>::cell_iterator &cell,
        std::vector<unsigned int> &                           line_cell_count,
        std::vector<unsigned int> &                           quad_cell_count)
      {
        AssertThrow(false, ExcNotImplemented());
        (void)triangulation;
        (void)cell;
        (void)line_cell_count;
        (void)quad_cell_count;
      }

      template <int dim, int spacedim>
      static typename Triangulation<dim, spacedim>::DistortedCellList
      execute_refinement(Triangulation<dim, spacedim> &triangulation,
                         const bool check_for_distorted_cells)
      {
        return Implementation::execute_refinement_isotropic(
          triangulation, check_for_distorted_cells);
      }

      template <int dim, int spacedim>
      static void
      prevent_distorted_boundary_cells(
        Triangulation<dim, spacedim> &triangulation)
      {
        // nothing to do since anisotropy is not supported
        (void)triangulation;
      }

      template <int dim, int spacedim>
      static void
      prepare_refinement_dim_dependent(
        Triangulation<dim, spacedim> &triangulation)
      {
        Implementation::prepare_refinement_dim_dependent(triangulation);
      }

      template <int dim, int spacedim>
      static bool
      coarsening_allowed(
        const typename Triangulation<dim, spacedim>::cell_iterator &cell)
      {
        AssertThrow(false, ExcNotImplemented());
        (void)cell;

        return false;
      }
    };


    template <int dim, int spacedim>
    const Manifold<dim, spacedim> &
    get_default_flat_manifold()
    {
      static const FlatManifold<dim, spacedim> flat_manifold;
      return flat_manifold;
    }
  } // namespace TriangulationImplementation
} // namespace internal



template <int dim, int spacedim>
const unsigned int Triangulation<dim, spacedim>::dimension;



template <int dim, int spacedim>
Triangulation<dim, spacedim>::Triangulation(
  const MeshSmoothing smooth_grid,
  const bool          check_for_distorted_cells)
  : smooth_grid(smooth_grid)
  , anisotropic_refinement(false)
  , check_for_distorted_cells(check_for_distorted_cells)
{
  if (dim == 1)
    {
      vertex_to_boundary_id_map_1d =
        std::make_unique<std::map<unsigned int, types::boundary_id>>();
      vertex_to_manifold_id_map_1d =
        std::make_unique<std::map<unsigned int, types::manifold_id>>();
    }

  // connect the any_change signal to the other top level signals
  signals.create.connect(signals.any_change);
  signals.post_refinement.connect(signals.any_change);
  signals.clear.connect(signals.any_change);
  signals.mesh_movement.connect(signals.any_change);
}



template <int dim, int spacedim>
Triangulation<dim, spacedim>::Triangulation(
  Triangulation<dim, spacedim> &&tria) noexcept
  : Subscriptor(std::move(tria))
  , smooth_grid(tria.smooth_grid)
  , reference_cells(std::move(tria.reference_cells))
  , periodic_face_pairs_level_0(std::move(tria.periodic_face_pairs_level_0))
  , periodic_face_map(std::move(tria.periodic_face_map))
  , levels(std::move(tria.levels))
  , faces(std::move(tria.faces))
  , vertices(std::move(tria.vertices))
  , vertices_used(std::move(tria.vertices_used))
  , manifolds(std::move(tria.manifolds))
  , anisotropic_refinement(tria.anisotropic_refinement)
  , check_for_distorted_cells(tria.check_for_distorted_cells)
  , number_cache(std::move(tria.number_cache))
  , vertex_to_boundary_id_map_1d(std::move(tria.vertex_to_boundary_id_map_1d))
  , vertex_to_manifold_id_map_1d(std::move(tria.vertex_to_manifold_id_map_1d))
{
  tria.number_cache = internal::TriangulationImplementation::NumberCache<dim>();

  if (tria.policy)
    this->policy = tria.policy->clone();
}


template <int dim, int spacedim>
Triangulation<dim, spacedim> &
Triangulation<dim, spacedim>::operator=(
  Triangulation<dim, spacedim> &&tria) noexcept
{
  Subscriptor::operator=(std::move(tria));

  smooth_grid                  = tria.smooth_grid;
  reference_cells              = std::move(tria.reference_cells);
  periodic_face_pairs_level_0  = std::move(tria.periodic_face_pairs_level_0);
  periodic_face_map            = std::move(tria.periodic_face_map);
  levels                       = std::move(tria.levels);
  faces                        = std::move(tria.faces);
  vertices                     = std::move(tria.vertices);
  vertices_used                = std::move(tria.vertices_used);
  manifolds                    = std::move(tria.manifolds);
  anisotropic_refinement       = tria.anisotropic_refinement;
  number_cache                 = tria.number_cache;
  vertex_to_boundary_id_map_1d = std::move(tria.vertex_to_boundary_id_map_1d);
  vertex_to_manifold_id_map_1d = std::move(tria.vertex_to_manifold_id_map_1d);

  tria.number_cache = internal::TriangulationImplementation::NumberCache<dim>();

  if (tria.policy)
    this->policy = tria.policy->clone();

  return *this;
}



template <int dim, int spacedim>
Triangulation<dim, spacedim>::~Triangulation()
{
  // notify listeners that the triangulation is going down...
  try
    {
      signals.clear();
    }
  catch (...)
    {}

  levels.clear();

  // the vertex_to_boundary_id_map_1d field should be unused except in
  // 1d. double check this here, as destruction is a good place to
  // ensure that what we've done over the course of the lifetime of
  // this object makes sense
  AssertNothrow((dim == 1) || (vertex_to_boundary_id_map_1d == nullptr),
                ExcInternalError());

  // the vertex_to_manifold_id_map_1d field should be also unused
  // except in 1d. check this as well
  AssertNothrow((dim == 1) || (vertex_to_manifold_id_map_1d == nullptr),
                ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear()
{
  // notify listeners that the triangulation is going down...
  signals.clear();

  // ...and then actually clear all content of it
  clear_despite_subscriptions();
  periodic_face_pairs_level_0.clear();
  periodic_face_map.clear();
  reference_cells.clear();
}


template <int dim, int spacedim>
MPI_Comm
Triangulation<dim, spacedim>::get_communicator() const
{
  return MPI_COMM_SELF;
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_mesh_smoothing(
  const MeshSmoothing mesh_smoothing)
{
  Assert(n_levels() == 0,
         ExcTriangulationNotEmpty(vertices.size(), levels.size()));
  smooth_grid = mesh_smoothing;
}



template <int dim, int spacedim>
const typename Triangulation<dim, spacedim>::MeshSmoothing &
Triangulation<dim, spacedim>::get_mesh_smoothing() const
{
  return smooth_grid;
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_manifold(
  const types::manifold_id       m_number,
  const Manifold<dim, spacedim> &manifold_object)
{
  AssertIndexRange(m_number, numbers::flat_manifold_id);

  manifolds[m_number] = manifold_object.clone();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_manifold(const types::manifold_id m_number)
{
  AssertIndexRange(m_number, numbers::flat_manifold_id);

  // delete the entry located at number.
  manifolds.erase(m_number);
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_all_manifolds()
{
  manifolds.clear();
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_all_manifold_ids(
  const types::manifold_id m_number)
{
  Assert(
    n_cells() > 0,
    ExcMessage(
      "Error: set_all_manifold_ids() can not be called on an empty Triangulation."));

  for (const auto &cell : this->active_cell_iterators())
    cell->set_all_manifold_ids(m_number);
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_all_manifold_ids_on_boundary(
  const types::manifold_id m_number)
{
  Assert(
    n_cells() > 0,
    ExcMessage(
      "Error: set_all_manifold_ids_on_boundary() can not be called on an empty Triangulation."));

  for (const auto &cell : this->active_cell_iterators())
    for (auto f : GeometryInfo<dim>::face_indices())
      if (cell->face(f)->at_boundary())
        cell->face(f)->set_all_manifold_ids(m_number);
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_all_manifold_ids_on_boundary(
  const types::boundary_id b_id,
  const types::manifold_id m_number)
{
  Assert(
    n_cells() > 0,
    ExcMessage(
      "Error: set_all_manifold_ids_on_boundary() can not be called on an empty Triangulation."));

  bool boundary_found = false;

  for (const auto &cell : this->active_cell_iterators())
    {
      // loop on faces
      for (auto f : GeometryInfo<dim>::face_indices())
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == b_id)
          {
            boundary_found = true;
            cell->face(f)->set_manifold_id(m_number);
          }

      // loop on edges if dim >= 3
      if (dim >= 3)
        for (unsigned int e = 0; e < GeometryInfo<dim>::lines_per_cell; ++e)
          if (cell->line(e)->at_boundary() &&
              cell->line(e)->boundary_id() == b_id)
            {
              boundary_found = true;
              cell->line(e)->set_manifold_id(m_number);
            }
    }

  (void)boundary_found;
  Assert(boundary_found, ExcBoundaryIdNotFound(b_id));
}



template <int dim, int spacedim>
const Manifold<dim, spacedim> &
Triangulation<dim, spacedim>::get_manifold(
  const types::manifold_id m_number) const
{
  // look, if there is a manifold stored at
  // manifold_id number.
  const auto it = manifolds.find(m_number);

  if (it != manifolds.end())
    {
      // if we have found an entry, return it
      return *(it->second);
    }

  // if we have not found an entry connected with number, we return
  // the default (flat) manifold
  return internal::TriangulationImplementation::
    get_default_flat_manifold<dim, spacedim>();
}



template <int dim, int spacedim>
std::vector<types::boundary_id>
Triangulation<dim, spacedim>::get_boundary_ids() const
{
  // in 1d, we store a map of all used boundary indicators. use it for
  // our purposes
  if (dim == 1)
    {
      std::vector<types::boundary_id> boundary_ids;
      for (std::map<unsigned int, types::boundary_id>::const_iterator p =
             vertex_to_boundary_id_map_1d->begin();
           p != vertex_to_boundary_id_map_1d->end();
           ++p)
        boundary_ids.push_back(p->second);

      return boundary_ids;
    }
  else
    {
      std::set<types::boundary_id> b_ids;
      for (auto cell : active_cell_iterators())
        if (cell->is_locally_owned())
          for (const unsigned int face : cell->face_indices())
            if (cell->at_boundary(face))
              b_ids.insert(cell->face(face)->boundary_id());
      std::vector<types::boundary_id> boundary_ids(b_ids.begin(), b_ids.end());
      return boundary_ids;
    }
}



template <int dim, int spacedim>
std::vector<types::manifold_id>
Triangulation<dim, spacedim>::get_manifold_ids() const
{
  std::set<types::manifold_id> m_ids;
  for (auto cell : active_cell_iterators())
    if (cell->is_locally_owned())
      {
        m_ids.insert(cell->manifold_id());
        for (const auto &face : cell->face_iterators())
          m_ids.insert(face->manifold_id());
        if (dim == 3)
          for (const unsigned int l : cell->line_indices())
            m_ids.insert(cell->line(l)->manifold_id());
      }
  return {m_ids.begin(), m_ids.end()};
}

/*-----------------------------------------------------------------*/


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::copy_triangulation(
  const Triangulation<dim, spacedim> &other_tria)
{
  Assert((vertices.size() == 0) && (levels.size() == 0) && (faces == nullptr),
         ExcTriangulationNotEmpty(vertices.size(), levels.size()));
  Assert((other_tria.levels.size() != 0) && (other_tria.vertices.size() != 0) &&
           (dim == 1 || other_tria.faces != nullptr),
         ExcMessage(
           "When calling Triangulation::copy_triangulation(), "
           "the target triangulation must be empty but the source "
           "triangulation (the argument to this function) must contain "
           "something. Here, it seems like the source does not "
           "contain anything at all."));


  // copy normal elements
  vertices               = other_tria.vertices;
  vertices_used          = other_tria.vertices_used;
  anisotropic_refinement = other_tria.anisotropic_refinement;
  smooth_grid            = other_tria.smooth_grid;
  reference_cells        = other_tria.reference_cells;

  if (dim > 1)
    faces = std::make_unique<internal::TriangulationImplementation::TriaFaces>(
      *other_tria.faces);

  for (const auto &p : other_tria.manifolds)
    set_manifold(p.first, *p.second);


  levels.reserve(other_tria.levels.size());
  for (unsigned int level = 0; level < other_tria.levels.size(); ++level)
    levels.push_back(
      std::make_unique<internal::TriangulationImplementation::TriaLevel>(
        *other_tria.levels[level]));

  number_cache = other_tria.number_cache;

  if (dim == 1)
    {
      vertex_to_boundary_id_map_1d =
        std::make_unique<std::map<unsigned int, types::boundary_id>>(
          *other_tria.vertex_to_boundary_id_map_1d);

      vertex_to_manifold_id_map_1d =
        std::make_unique<std::map<unsigned int, types::manifold_id>>(
          *other_tria.vertex_to_manifold_id_map_1d);
    }

  if (other_tria.policy)
    this->policy = other_tria.policy->clone();

  // inform those who are listening on other_tria of the copy operation
  other_tria.signals.copy(*this);
  // also inform all listeners of the current triangulation that the
  // triangulation has been created
  signals.create();

  // note that we need not copy the
  // subscriptor!
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::create_triangulation_compatibility(
  const std::vector<Point<spacedim>> &v,
  const std::vector<CellData<dim>> &  cells,
  const SubCellData &                 subcelldata)
{
  std::vector<CellData<dim>> reordered_cells(cells);             // NOLINT
  SubCellData                reordered_subcelldata(subcelldata); // NOLINT

  // in-place reordering of data
  reorder_compatibility(reordered_cells, reordered_subcelldata);

  // now create triangulation from
  // reordered data
  create_triangulation(v, reordered_cells, reordered_subcelldata);
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_policy()
{
  this->update_reference_cells();

  if (this->all_reference_cells_are_hyper_cube())
    {
      this->policy =
        std::make_unique<internal::TriangulationImplementation::PolicyWrapper<
          dim,
          spacedim,
          internal::TriangulationImplementation::Implementation>>();
    }
  else
    {
      this->policy =
        std::make_unique<internal::TriangulationImplementation::PolicyWrapper<
          dim,
          spacedim,
          internal::TriangulationImplementation::ImplementationMixedMesh>>();
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::create_triangulation(
  const std::vector<Point<spacedim>> &v,
  const std::vector<CellData<dim>> &  cells,
  const SubCellData &                 subcelldata)
{
  Assert((vertices.size() == 0) && (levels.size() == 0) && (faces == nullptr),
         ExcTriangulationNotEmpty(vertices.size(), levels.size()));
  // check that no forbidden arrays
  // are used
  Assert(subcelldata.check_consistency(dim), ExcInternalError());

  // try to create a triangulation; if this fails, we still want to
  // throw an exception but if we just do so we'll get into trouble
  // because sometimes other objects are already attached to it:
  try
    {
      internal::TriangulationImplementation::Implementation::
        create_triangulation(v, cells, subcelldata, *this);
    }
  catch (...)
    {
      clear_despite_subscriptions();
      throw;
    }

  reset_policy();

  // update our counts of the various elements of a triangulation, and set
  // active_cell_indices of all cells
  reset_cell_vertex_indices_cache();
  internal::TriangulationImplementation::Implementation::compute_number_cache(
    *this, levels.size(), number_cache);
  reset_active_cell_indices();
  reset_global_cell_indices();

  // now verify that there are indeed no distorted cells. as per the
  // documentation of this class, we first collect all distorted cells
  // and then throw an exception if there are any
  if (check_for_distorted_cells)
    {
      DistortedCellList distorted_cells = collect_distorted_coarse_cells(*this);
      // throw the array (and fill the various location fields) if
      // there are distorted cells. otherwise, just fall off the end
      // of the function
      AssertThrow(distorted_cells.distorted_cells.size() == 0, distorted_cells);
    }


  /*
      When the triangulation is a manifold (dim < spacedim), the normal field
      provided from the map class depends on the order of the vertices.
      It may happen that this normal field is discontinuous.
      The following code takes care that this is not the case by setting the
      cell direction flag on those cell that produce the wrong orientation.

      To determine if 2 neighbours have the same or opposite orientation
      we use a table of truth.
      Its entries are indexes by the local indices of the common face.
      For example if two elements share a face, and this face is
      face 0 for element 0 and face 1 for element 1, then
      table(0,1) will tell whether the orientation are the same (true) or
      opposite (false).

      Even though there may be a combinatorial/graph theory argument to get
      this table in any dimension, I tested by hand all the different possible
      cases in 1D and 2D to generate the table.

      Assuming that a surface respects the standard orientation for 2d meshes,
      the tables of truth are symmetric and their true values are the following
      1D curves:  (0,1)
      2D surface: (0,1),(0,2),(1,3),(2,3)

      We store this data using an n_faces x n_faces full matrix, which is
     actually much bigger than the minimal data required, but it makes the code
     more readable.

    */
  if (dim < spacedim)
    {
      Table<2, bool> correct(GeometryInfo<dim>::faces_per_cell,
                             GeometryInfo<dim>::faces_per_cell);
      switch (dim)
        {
          case 1:
            {
              bool values[][2] = {{false, true}, {true, false}};
              for (const unsigned int i : GeometryInfo<dim>::face_indices())
                for (const unsigned int j : GeometryInfo<dim>::face_indices())
                  correct(i, j) = (values[i][j]);
              break;
            }
          case 2:
            {
              bool values[][4] = {{false, true, true, false},
                                  {true, false, false, true},
                                  {true, false, false, true},
                                  {false, true, true, false}};
              for (const unsigned int i : GeometryInfo<dim>::face_indices())
                for (const unsigned int j : GeometryInfo<dim>::face_indices())
                  correct(i, j) = (values[i][j]);
              break;
            }
          default:
            Assert(false, ExcNotImplemented());
        }


      std::list<active_cell_iterator> this_round, next_round;
      active_cell_iterator            neighbor;

      this_round.push_back(begin_active());
      begin_active()->set_direction_flag(true);
      begin_active()->set_user_flag();

      while (this_round.size() > 0)
        {
          for (typename std::list<active_cell_iterator>::iterator cell =
                 this_round.begin();
               cell != this_round.end();
               ++cell)
            {
              for (const unsigned int i : (*cell)->face_indices())
                {
                  if (!((*cell)->face(i)->at_boundary()))
                    {
                      neighbor = (*cell)->neighbor(i);

                      unsigned int cf = (*cell)->face_index(i);
                      unsigned int j  = 0;
                      while (neighbor->face_index(j) != cf)
                        {
                          ++j;
                        }


                      // If we already saw this guy, check that everything is
                      // fine
                      if (neighbor->user_flag_set())
                        {
                          // If we have visited this guy, then the ordering and
                          // the orientation should agree
                          Assert(!(correct(i, j) ^
                                   (neighbor->direction_flag() ==
                                    (*cell)->direction_flag())),
                                 ExcNonOrientableTriangulation());
                        }
                      else
                        {
                          next_round.push_back(neighbor);
                          neighbor->set_user_flag();
                          if ((correct(i, j) ^ (neighbor->direction_flag() ==
                                                (*cell)->direction_flag())))
                            neighbor->set_direction_flag(
                              !neighbor->direction_flag());
                        }
                    }
                }
            }

          // Before we quit let's check
          // that if the triangulation
          // is disconnected that we
          // still get all cells
          if (next_round.size() == 0)
            for (const auto &cell : this->active_cell_iterators())
              if (cell->user_flag_set() == false)
                {
                  next_round.push_back(cell);
                  cell->set_direction_flag(true);
                  cell->set_user_flag();
                  break;
                }

          this_round = next_round;
          next_round.clear();
        }
    }

  // inform all listeners that the triangulation has been created
  signals.create();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::create_triangulation(
  const TriangulationDescription::Description<dim, spacedim> &construction_data)
{
  // 1) create coarse grid
  create_triangulation(construction_data.coarse_cell_vertices,
                       construction_data.coarse_cells,
                       SubCellData());

  // create a copy of cell_infos such that we can sort them
  auto cell_infos = construction_data.cell_infos;

  // sort cell_infos on each level separately
  for (auto &cell_info : cell_infos)
    std::sort(
      cell_info.begin(),
      cell_info.end(),
      [&](TriangulationDescription::CellData<dim> a,
          TriangulationDescription::CellData<dim> b) {
        const CellId a_id(a.id);
        const CellId b_id(b.id);

        const auto a_coarse_cell_index =
          this->coarse_cell_id_to_coarse_cell_index(a_id.get_coarse_cell_id());
        const auto b_coarse_cell_index =
          this->coarse_cell_id_to_coarse_cell_index(b_id.get_coarse_cell_id());

        // according to their coarse-cell index and if that is
        // same according to their cell id (the result is that
        // cells on each level are sorted according to their
        // index on that level - what we need in the following
        // operations)
        if (a_coarse_cell_index != b_coarse_cell_index)
          return a_coarse_cell_index < b_coarse_cell_index;
        else
          return a_id < b_id;
      });

  // 2) create all levels via a sequence of refinements. note that
  //    we must make sure that we actually have cells on this level,
  //    which is not clear in a parallel context for some processes
  for (unsigned int level = 0;
       level < cell_infos.size() && !cell_infos[level].empty();
       ++level)
    {
      // a) set manifold ids here (because new vertices have to be
      //    positioned correctly during each refinement step)
      {
        auto cell      = this->begin(level);
        auto cell_info = cell_infos[level].begin();
        for (; cell_info != cell_infos[level].end(); ++cell_info)
          {
            while (cell_info->id != cell->id().template to_binary<dim>())
              ++cell;
            if (dim == 3)
              for (const auto quad : cell->face_indices())
                cell->quad(quad)->set_manifold_id(
                  cell_info->manifold_quad_ids[quad]);

            if (dim >= 2)
              for (const auto line : cell->line_indices())
                cell->line(line)->set_manifold_id(
                  cell_info->manifold_line_ids[line]);

            cell->set_manifold_id(cell_info->manifold_id);
          }
      }

      // b) perform refinement on all levels but on the finest
      if (level + 1 != cell_infos.size())
        {
          // find cells that should have children and mark them for
          // refinement
          auto coarse_cell    = this->begin(level);
          auto fine_cell_info = cell_infos[level + 1].begin();

          // loop over all cells on the next level
          for (; fine_cell_info != cell_infos[level + 1].end();
               ++fine_cell_info)
            {
              // find the parent of that cell
              while (
                !coarse_cell->id().is_parent_of(CellId(fine_cell_info->id)))
                ++coarse_cell;

              // set parent for refinement
              coarse_cell->set_refine_flag();
            }

          // execute refinement
          dealii::Triangulation<dim,
                                spacedim>::execute_coarsening_and_refinement();
        }
    }

  // 3) set boundary ids
  for (unsigned int level = 0;
       level < cell_infos.size() && !cell_infos[level].empty();
       ++level)
    {
      auto cell      = this->begin(level);
      auto cell_info = cell_infos[level].begin();
      for (; cell_info != cell_infos[level].end(); ++cell_info)
        {
          // find cell that has the correct cell
          while (cell_info->id != cell->id().template to_binary<dim>())
            ++cell;

          // boundary ids
          for (auto pair : cell_info->boundary_ids)
            if (cell->face(pair.first)->at_boundary())
              cell->face(pair.first)->set_boundary_id(pair.second);
        }
    }
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::flip_all_direction_flags()
{
  AssertThrow(dim + 1 == spacedim,
              ExcMessage("Only works for dim == spacedim-1"));
  for (const auto &cell : this->active_cell_iterators())
    cell->set_direction_flag(!cell->direction_flag());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::set_all_refine_flags()
{
  Assert(n_cells() > 0,
         ExcMessage("Error: An empty Triangulation can not be refined."));

  for (const auto &cell : this->active_cell_iterators())
    {
      cell->clear_coarsen_flag();
      cell->set_refine_flag();
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::refine_global(const unsigned int times)
{
  for (unsigned int i = 0; i < times; ++i)
    {
      set_all_refine_flags();
      execute_coarsening_and_refinement();
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::coarsen_global(const unsigned int times)
{
  for (unsigned int i = 0; i < times; ++i)
    {
      for (const auto &cell : this->active_cell_iterators())
        {
          cell->clear_refine_flag();
          cell->set_coarsen_flag();
        }
      execute_coarsening_and_refinement();
    }
}


/*-------------------- refine/coarsen flags -------------------------*/



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_refine_flags(std::vector<bool> &v) const
{
  v.resize(dim * n_active_cells(), false);
  std::vector<bool>::iterator i = v.begin();

  for (const auto &cell : this->active_cell_iterators())
    for (unsigned int j = 0; j < dim; ++j, ++i)
      if (cell->refine_flag_set() & (1 << j))
        *i = true;

  Assert(i == v.end(), ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_refine_flags(std::ostream &out) const
{
  std::vector<bool> v;
  save_refine_flags(v);
  write_bool_vector(mn_tria_refine_flags_begin,
                    v,
                    mn_tria_refine_flags_end,
                    out);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_refine_flags(std::istream &in)
{
  std::vector<bool> v;
  read_bool_vector(mn_tria_refine_flags_begin, v, mn_tria_refine_flags_end, in);
  load_refine_flags(v);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_refine_flags(const std::vector<bool> &v)
{
  AssertThrow(v.size() == dim * n_active_cells(), ExcGridReadError());

  std::vector<bool>::const_iterator i = v.begin();
  for (const auto &cell : this->active_cell_iterators())
    {
      unsigned int ref_case = 0;

      for (unsigned int j = 0; j < dim; ++j, ++i)
        if (*i == true)
          ref_case += 1 << j;
      Assert(ref_case < RefinementCase<dim>::isotropic_refinement + 1,
             ExcGridReadError());
      if (ref_case > 0)
        cell->set_refine_flag(RefinementCase<dim>(ref_case));
      else
        cell->clear_refine_flag();
    }

  Assert(i == v.end(), ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_coarsen_flags(std::vector<bool> &v) const
{
  v.resize(n_active_cells(), false);
  std::vector<bool>::iterator i = v.begin();
  for (const auto &cell : this->active_cell_iterators())
    {
      *i = cell->coarsen_flag_set();
      ++i;
    }

  Assert(i == v.end(), ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_coarsen_flags(std::ostream &out) const
{
  std::vector<bool> v;
  save_coarsen_flags(v);
  write_bool_vector(mn_tria_coarsen_flags_begin,
                    v,
                    mn_tria_coarsen_flags_end,
                    out);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_coarsen_flags(std::istream &in)
{
  std::vector<bool> v;
  read_bool_vector(mn_tria_coarsen_flags_begin,
                   v,
                   mn_tria_coarsen_flags_end,
                   in);
  load_coarsen_flags(v);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_coarsen_flags(const std::vector<bool> &v)
{
  Assert(v.size() == n_active_cells(), ExcGridReadError());

  std::vector<bool>::const_iterator i = v.begin();
  for (const auto &cell : this->active_cell_iterators())
    {
      if (*i == true)
        cell->set_coarsen_flag();
      else
        cell->clear_coarsen_flag();
      ++i;
    }

  Assert(i == v.end(), ExcInternalError());
}


template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::get_anisotropic_refinement_flag() const
{
  return anisotropic_refinement;
}



/*-------------------- user data/flags -------------------------*/


namespace
{
  // clear user data of cells
  void
  clear_user_data(std::vector<std::unique_ptr<
                    internal::TriangulationImplementation::TriaLevel>> &levels)
  {
    for (auto &level : levels)
      level->cells.clear_user_data();
  }


  // clear user data of faces
  void
  clear_user_data(internal::TriangulationImplementation::TriaFaces *faces)
  {
    if (faces->dim == 2)
      {
        faces->lines.clear_user_data();
      }


    if (faces->dim == 3)
      {
        faces->lines.clear_user_data();
        faces->quads.clear_user_data();
      }
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_user_data()
{
  // let functions in anonymous namespace do their work
  dealii::clear_user_data(levels);
  if (dim > 1)
    dealii::clear_user_data(faces.get());
}



namespace
{
  void
  clear_user_flags_line(
    unsigned int dim,
    std::vector<
      std::unique_ptr<internal::TriangulationImplementation::TriaLevel>>
      &                                               levels,
    internal::TriangulationImplementation::TriaFaces *faces)
  {
    if (dim == 1)
      {
        for (const auto &level : levels)
          level->cells.clear_user_flags();
      }
    else if (dim == 2 || dim == 3)
      {
        faces->lines.clear_user_flags();
      }
    else
      {
        Assert(false, ExcNotImplemented())
      }
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_user_flags_line()
{
  dealii::clear_user_flags_line(dim, levels, faces.get());
}



namespace
{
  void
  clear_user_flags_quad(
    unsigned int dim,
    std::vector<
      std::unique_ptr<internal::TriangulationImplementation::TriaLevel>>
      &                                               levels,
    internal::TriangulationImplementation::TriaFaces *faces)
  {
    if (dim == 1)
      {
        // nothing to do in 1d
      }
    else if (dim == 2)
      {
        for (const auto &level : levels)
          level->cells.clear_user_flags();
      }
    else if (dim == 3)
      {
        faces->quads.clear_user_flags();
      }
    else
      {
        Assert(false, ExcNotImplemented())
      }
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_user_flags_quad()
{
  dealii::clear_user_flags_quad(dim, levels, faces.get());
}



namespace
{
  void
  clear_user_flags_hex(
    unsigned int dim,
    std::vector<
      std::unique_ptr<internal::TriangulationImplementation::TriaLevel>>
      &levels,
    internal::TriangulationImplementation::TriaFaces *)
  {
    if (dim == 1)
      {
        // nothing to do in 1d
      }
    else if (dim == 2)
      {
        // nothing to do in 2d
      }
    else if (dim == 3)
      {
        for (const auto &level : levels)
          level->cells.clear_user_flags();
      }
    else
      {
        Assert(false, ExcNotImplemented())
      }
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_user_flags_hex()
{
  dealii::clear_user_flags_hex(dim, levels, faces.get());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_user_flags()
{
  clear_user_flags_line();
  clear_user_flags_quad();
  clear_user_flags_hex();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags(std::ostream &out) const
{
  save_user_flags_line(out);

  if (dim >= 2)
    save_user_flags_quad(out);

  if (dim >= 3)
    save_user_flags_hex(out);

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags(std::vector<bool> &v) const
{
  // clear vector and append
  // all the stuff later on
  v.clear();

  std::vector<bool> tmp;

  save_user_flags_line(tmp);
  v.insert(v.end(), tmp.begin(), tmp.end());

  if (dim >= 2)
    {
      save_user_flags_quad(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 3)
    {
      save_user_flags_hex(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags(std::istream &in)
{
  load_user_flags_line(in);

  if (dim >= 2)
    load_user_flags_quad(in);

  if (dim >= 3)
    load_user_flags_hex(in);

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags(const std::vector<bool> &v)
{
  Assert(v.size() == n_lines() + n_quads() + n_hexs(), ExcInternalError());
  std::vector<bool> tmp;

  // first extract the flags
  // belonging to lines
  tmp.insert(tmp.end(), v.begin(), v.begin() + n_lines());
  // and set the lines
  load_user_flags_line(tmp);

  if (dim >= 2)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines(),
                 v.begin() + n_lines() + n_quads());
      load_user_flags_quad(tmp);
    }

  if (dim >= 3)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines() + n_quads(),
                 v.begin() + n_lines() + n_quads() + n_hexs());
      load_user_flags_hex(tmp);
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_line(std::vector<bool> &v) const
{
  v.resize(n_lines(), false);
  std::vector<bool>::iterator i    = v.begin();
  line_iterator               line = begin_line(), endl = end_line();
  for (; line != endl; ++line, ++i)
    *i = line->user_flag_set();

  Assert(i == v.end(), ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_line(std::ostream &out) const
{
  std::vector<bool> v;
  save_user_flags_line(v);
  write_bool_vector(mn_tria_line_user_flags_begin,
                    v,
                    mn_tria_line_user_flags_end,
                    out);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_line(std::istream &in)
{
  std::vector<bool> v;
  read_bool_vector(mn_tria_line_user_flags_begin,
                   v,
                   mn_tria_line_user_flags_end,
                   in);
  load_user_flags_line(v);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_line(const std::vector<bool> &v)
{
  Assert(v.size() == n_lines(), ExcGridReadError());

  line_iterator                     line = begin_line(), endl = end_line();
  std::vector<bool>::const_iterator i = v.begin();
  for (; line != endl; ++line, ++i)
    if (*i == true)
      line->set_user_flag();
    else
      line->clear_user_flag();

  Assert(i == v.end(), ExcInternalError());
}


namespace
{
  template <typename Iterator>
  bool
  get_user_flag(const Iterator &i)
  {
    return i->user_flag_set();
  }



  template <int structdim, int dim, int spacedim>
  bool
  get_user_flag(const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &)
  {
    Assert(false, ExcInternalError());
    return false;
  }



  template <typename Iterator>
  void
  set_user_flag(const Iterator &i)
  {
    i->set_user_flag();
  }



  template <int structdim, int dim, int spacedim>
  void
  set_user_flag(const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &)
  {
    Assert(false, ExcInternalError());
  }



  template <typename Iterator>
  void
  clear_user_flag(const Iterator &i)
  {
    i->clear_user_flag();
  }



  template <int structdim, int dim, int spacedim>
  void
  clear_user_flag(
    const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &)
  {
    Assert(false, ExcInternalError());
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_quad(std::vector<bool> &v) const
{
  v.resize(n_quads(), false);

  if (dim >= 2)
    {
      std::vector<bool>::iterator i    = v.begin();
      quad_iterator               quad = begin_quad(), endq = end_quad();
      for (; quad != endq; ++quad, ++i)
        *i = get_user_flag(quad);

      Assert(i == v.end(), ExcInternalError());
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_quad(std::ostream &out) const
{
  std::vector<bool> v;
  save_user_flags_quad(v);
  write_bool_vector(mn_tria_quad_user_flags_begin,
                    v,
                    mn_tria_quad_user_flags_end,
                    out);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_quad(std::istream &in)
{
  std::vector<bool> v;
  read_bool_vector(mn_tria_quad_user_flags_begin,
                   v,
                   mn_tria_quad_user_flags_end,
                   in);
  load_user_flags_quad(v);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_quad(const std::vector<bool> &v)
{
  Assert(v.size() == n_quads(), ExcGridReadError());

  if (dim >= 2)
    {
      quad_iterator                     quad = begin_quad(), endq = end_quad();
      std::vector<bool>::const_iterator i = v.begin();
      for (; quad != endq; ++quad, ++i)
        if (*i == true)
          set_user_flag(quad);
        else
          clear_user_flag(quad);

      Assert(i == v.end(), ExcInternalError());
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_hex(std::vector<bool> &v) const
{
  v.resize(n_hexs(), false);

  if (dim >= 3)
    {
      std::vector<bool>::iterator i   = v.begin();
      hex_iterator                hex = begin_hex(), endh = end_hex();
      for (; hex != endh; ++hex, ++i)
        *i = get_user_flag(hex);

      Assert(i == v.end(), ExcInternalError());
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_flags_hex(std::ostream &out) const
{
  std::vector<bool> v;
  save_user_flags_hex(v);
  write_bool_vector(mn_tria_hex_user_flags_begin,
                    v,
                    mn_tria_hex_user_flags_end,
                    out);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_hex(std::istream &in)
{
  std::vector<bool> v;
  read_bool_vector(mn_tria_hex_user_flags_begin,
                   v,
                   mn_tria_hex_user_flags_end,
                   in);
  load_user_flags_hex(v);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_flags_hex(const std::vector<bool> &v)
{
  Assert(v.size() == n_hexs(), ExcGridReadError());

  if (dim >= 3)
    {
      hex_iterator                      hex = begin_hex(), endh = end_hex();
      std::vector<bool>::const_iterator i = v.begin();
      for (; hex != endh; ++hex, ++i)
        if (*i == true)
          set_user_flag(hex);
        else
          clear_user_flag(hex);

      Assert(i == v.end(), ExcInternalError());
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_indices(
  std::vector<unsigned int> &v) const
{
  // clear vector and append all the
  // stuff later on
  v.clear();

  std::vector<unsigned int> tmp;

  save_user_indices_line(tmp);
  v.insert(v.end(), tmp.begin(), tmp.end());

  if (dim >= 2)
    {
      save_user_indices_quad(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 3)
    {
      save_user_indices_hex(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_indices(
  const std::vector<unsigned int> &v)
{
  Assert(v.size() == n_lines() + n_quads() + n_hexs(), ExcInternalError());
  std::vector<unsigned int> tmp;

  // first extract the indices
  // belonging to lines
  tmp.insert(tmp.end(), v.begin(), v.begin() + n_lines());
  // and set the lines
  load_user_indices_line(tmp);

  if (dim >= 2)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines(),
                 v.begin() + n_lines() + n_quads());
      load_user_indices_quad(tmp);
    }

  if (dim >= 3)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines() + n_quads(),
                 v.begin() + n_lines() + n_quads() + n_hexs());
      load_user_indices_hex(tmp);
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



namespace
{
  template <typename Iterator>
  unsigned int
  get_user_index(const Iterator &i)
  {
    return i->user_index();
  }



  template <int structdim, int dim, int spacedim>
  unsigned int
  get_user_index(
    const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &)
  {
    Assert(false, ExcInternalError());
    return numbers::invalid_unsigned_int;
  }



  template <typename Iterator>
  void
  set_user_index(const Iterator &i, const unsigned int x)
  {
    i->set_user_index(x);
  }



  template <int structdim, int dim, int spacedim>
  void
  set_user_index(
    const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &,
    const unsigned int)
  {
    Assert(false, ExcInternalError());
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_indices_line(
  std::vector<unsigned int> &v) const
{
  v.resize(n_lines(), 0);
  std::vector<unsigned int>::iterator i    = v.begin();
  line_iterator                       line = begin_line(), endl = end_line();
  for (; line != endl; ++line, ++i)
    *i = line->user_index();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_indices_line(
  const std::vector<unsigned int> &v)
{
  Assert(v.size() == n_lines(), ExcGridReadError());

  line_iterator line = begin_line(), endl = end_line();
  std::vector<unsigned int>::const_iterator i = v.begin();
  for (; line != endl; ++line, ++i)
    line->set_user_index(*i);
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_indices_quad(
  std::vector<unsigned int> &v) const
{
  v.resize(n_quads(), 0);

  if (dim >= 2)
    {
      std::vector<unsigned int>::iterator i = v.begin();
      quad_iterator quad = begin_quad(), endq = end_quad();
      for (; quad != endq; ++quad, ++i)
        *i = get_user_index(quad);
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_indices_quad(
  const std::vector<unsigned int> &v)
{
  Assert(v.size() == n_quads(), ExcGridReadError());

  if (dim >= 2)
    {
      quad_iterator quad = begin_quad(), endq = end_quad();
      std::vector<unsigned int>::const_iterator i = v.begin();
      for (; quad != endq; ++quad, ++i)
        set_user_index(quad, *i);
    }
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_indices_hex(
  std::vector<unsigned int> &v) const
{
  v.resize(n_hexs(), 0);

  if (dim >= 3)
    {
      std::vector<unsigned int>::iterator i   = v.begin();
      hex_iterator                        hex = begin_hex(), endh = end_hex();
      for (; hex != endh; ++hex, ++i)
        *i = get_user_index(hex);
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_indices_hex(
  const std::vector<unsigned int> &v)
{
  Assert(v.size() == n_hexs(), ExcGridReadError());

  if (dim >= 3)
    {
      hex_iterator hex = begin_hex(), endh = end_hex();
      std::vector<unsigned int>::const_iterator i = v.begin();
      for (; hex != endh; ++hex, ++i)
        set_user_index(hex, *i);
    }
}



//---------------- user pointers ----------------------------------------//


namespace
{
  template <typename Iterator>
  void *
  get_user_pointer(const Iterator &i)
  {
    return i->user_pointer();
  }



  template <int structdim, int dim, int spacedim>
  void *
  get_user_pointer(
    const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &)
  {
    Assert(false, ExcInternalError());
    return nullptr;
  }



  template <typename Iterator>
  void
  set_user_pointer(const Iterator &i, void *x)
  {
    i->set_user_pointer(x);
  }



  template <int structdim, int dim, int spacedim>
  void
  set_user_pointer(
    const TriaIterator<InvalidAccessor<structdim, dim, spacedim>> &,
    void *)
  {
    Assert(false, ExcInternalError());
  }
} // namespace


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_pointers(std::vector<void *> &v) const
{
  // clear vector and append all the
  // stuff later on
  v.clear();

  std::vector<void *> tmp;

  save_user_pointers_line(tmp);
  v.insert(v.end(), tmp.begin(), tmp.end());

  if (dim >= 2)
    {
      save_user_pointers_quad(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 3)
    {
      save_user_pointers_hex(tmp);
      v.insert(v.end(), tmp.begin(), tmp.end());
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_pointers(const std::vector<void *> &v)
{
  Assert(v.size() == n_lines() + n_quads() + n_hexs(), ExcInternalError());
  std::vector<void *> tmp;

  // first extract the pointers
  // belonging to lines
  tmp.insert(tmp.end(), v.begin(), v.begin() + n_lines());
  // and set the lines
  load_user_pointers_line(tmp);

  if (dim >= 2)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines(),
                 v.begin() + n_lines() + n_quads());
      load_user_pointers_quad(tmp);
    }

  if (dim >= 3)
    {
      tmp.clear();
      tmp.insert(tmp.end(),
                 v.begin() + n_lines() + n_quads(),
                 v.begin() + n_lines() + n_quads() + n_hexs());
      load_user_pointers_hex(tmp);
    }

  if (dim >= 4)
    Assert(false, ExcNotImplemented());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_pointers_line(
  std::vector<void *> &v) const
{
  v.resize(n_lines(), nullptr);
  std::vector<void *>::iterator i    = v.begin();
  line_iterator                 line = begin_line(), endl = end_line();
  for (; line != endl; ++line, ++i)
    *i = line->user_pointer();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_pointers_line(
  const std::vector<void *> &v)
{
  Assert(v.size() == n_lines(), ExcGridReadError());

  line_iterator                       line = begin_line(), endl = end_line();
  std::vector<void *>::const_iterator i = v.begin();
  for (; line != endl; ++line, ++i)
    line->set_user_pointer(*i);
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_pointers_quad(
  std::vector<void *> &v) const
{
  v.resize(n_quads(), nullptr);

  if (dim >= 2)
    {
      std::vector<void *>::iterator i    = v.begin();
      quad_iterator                 quad = begin_quad(), endq = end_quad();
      for (; quad != endq; ++quad, ++i)
        *i = get_user_pointer(quad);
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_pointers_quad(
  const std::vector<void *> &v)
{
  Assert(v.size() == n_quads(), ExcGridReadError());

  if (dim >= 2)
    {
      quad_iterator quad = begin_quad(), endq = end_quad();
      std::vector<void *>::const_iterator i = v.begin();
      for (; quad != endq; ++quad, ++i)
        set_user_pointer(quad, *i);
    }
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::save_user_pointers_hex(
  std::vector<void *> &v) const
{
  v.resize(n_hexs(), nullptr);

  if (dim >= 3)
    {
      std::vector<void *>::iterator i   = v.begin();
      hex_iterator                  hex = begin_hex(), endh = end_hex();
      for (; hex != endh; ++hex, ++i)
        *i = get_user_pointer(hex);
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::load_user_pointers_hex(
  const std::vector<void *> &v)
{
  Assert(v.size() == n_hexs(), ExcGridReadError());

  if (dim >= 3)
    {
      hex_iterator                        hex = begin_hex(), endh = end_hex();
      std::vector<void *>::const_iterator i = v.begin();
      for (; hex != endh; ++hex, ++i)
        set_user_pointer(hex, *i);
    }
}



/*------------------------ Cell iterator functions ------------------------*/


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::raw_cell_iterator
Triangulation<dim, spacedim>::begin_raw(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return begin_raw_line(level);
      case 2:
        return begin_raw_quad(level);
      case 3:
        return begin_raw_hex(level);
      default:
        Assert(false, ExcNotImplemented());
        return raw_cell_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::cell_iterator
Triangulation<dim, spacedim>::begin(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return begin_line(level);
      case 2:
        return begin_quad(level);
      case 3:
        return begin_hex(level);
      default:
        Assert(false, ExcImpossibleInDim(dim));
        return cell_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_cell_iterator
Triangulation<dim, spacedim>::begin_active(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return begin_active_line(level);
      case 2:
        return begin_active_quad(level);
      case 3:
        return begin_active_hex(level);
      default:
        Assert(false, ExcNotImplemented());
        return active_cell_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::cell_iterator
Triangulation<dim, spacedim>::last() const
{
  const unsigned int level = levels.size() - 1;
  if (levels[level]->cells.n_objects() == 0)
    return end(level);

  // find the last raw iterator on
  // this level
  raw_cell_iterator ri(const_cast<Triangulation<dim, spacedim> *>(this),
                       level,
                       levels[level]->cells.n_objects() - 1);

  // then move to the last used one
  if (ri->used() == true)
    return ri;
  while ((--ri).state() == IteratorState::valid)
    if (ri->used() == true)
      return ri;
  return ri;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_cell_iterator
Triangulation<dim, spacedim>::last_active() const
{
  // get the last used cell
  cell_iterator cell = last();

  if (cell != end())
    {
      // then move to the last active one
      if (cell->is_active() == true)
        return cell;
      while ((--cell).state() == IteratorState::valid)
        if (cell->is_active() == true)
          return cell;
    }
  return cell;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::cell_iterator
Triangulation<dim, spacedim>::create_cell_iterator(const CellId &cell_id) const
{
  cell_iterator cell(
    this, 0, coarse_cell_id_to_coarse_cell_index(cell_id.get_coarse_cell_id()));

  for (const auto &child_index : cell_id.get_child_indices())
    {
      Assert(
        cell->has_children(),
        ExcMessage(
          "CellId is invalid for this triangulation.\n"
          "Either the provided CellId does not correspond to a cell in this "
          "triangulation object, or, in case you are using a parallel "
          "triangulation, may correspond to an artificial cell that is less "
          "refined on this processor."));
      cell = cell->child(static_cast<unsigned int>(child_index));
    }

  return cell;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::cell_iterator
Triangulation<dim, spacedim>::end() const
{
  return cell_iterator(const_cast<Triangulation<dim, spacedim> *>(this),
                       -1,
                       -1);
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::raw_cell_iterator
Triangulation<dim, spacedim>::end_raw(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just return the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end();
    }

  // Query whether the given level is valid for the local portion of the
  // triangulation.
  Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));
  if (level < levels.size() - 1)
    return begin_raw(level + 1);
  else
    return end();
}


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::cell_iterator
Triangulation<dim, spacedim>::end(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just retrn the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end();
    }

  // Query whether the given level is valid for the local portion of the
  // triangulation.
  Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));
  if (level < levels.size() - 1)
    return begin(level + 1);
  else
    return end();
}


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_cell_iterator
Triangulation<dim, spacedim>::end_active(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just return the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end();
    }

  // Query whether the given level is valid for the local portion of the
  // triangulation.
  Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));
  return (level >= levels.size() - 1 ? active_cell_iterator(end()) :
                                       begin_active(level + 1));
}



template <int dim, int spacedim>
IteratorRange<typename Triangulation<dim, spacedim>::cell_iterator>
Triangulation<dim, spacedim>::cell_iterators() const
{
  return IteratorRange<typename Triangulation<dim, spacedim>::cell_iterator>(
    begin(), end());
}


template <int dim, int spacedim>
IteratorRange<typename Triangulation<dim, spacedim>::active_cell_iterator>
Triangulation<dim, spacedim>::active_cell_iterators() const
{
  return IteratorRange<
    typename Triangulation<dim, spacedim>::active_cell_iterator>(begin_active(),
                                                                 end());
}



template <int dim, int spacedim>
IteratorRange<typename Triangulation<dim, spacedim>::cell_iterator>
Triangulation<dim, spacedim>::cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<typename Triangulation<dim, spacedim>::cell_iterator>(
    begin(level), end(level));
}



template <int dim, int spacedim>
IteratorRange<typename Triangulation<dim, spacedim>::active_cell_iterator>
Triangulation<dim, spacedim>::active_cell_iterators_on_level(
  const unsigned int level) const
{
  return IteratorRange<
    typename Triangulation<dim, spacedim>::active_cell_iterator>(
    begin_active(level), end_active(level));
}


/*------------------------ Face iterator functions ------------------------*/


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::face_iterator
Triangulation<dim, spacedim>::begin_face() const
{
  switch (dim)
    {
      case 1:
        Assert(false, ExcImpossibleInDim(1));
        return raw_face_iterator();
      case 2:
        return begin_line();
      case 3:
        return begin_quad();
      default:
        Assert(false, ExcNotImplemented());
        return face_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_face_iterator
Triangulation<dim, spacedim>::begin_active_face() const
{
  switch (dim)
    {
      case 1:
        Assert(false, ExcImpossibleInDim(1));
        return raw_face_iterator();
      case 2:
        return begin_active_line();
      case 3:
        return begin_active_quad();
      default:
        Assert(false, ExcNotImplemented());
        return active_face_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::face_iterator
Triangulation<dim, spacedim>::end_face() const
{
  switch (dim)
    {
      case 1:
        Assert(false, ExcImpossibleInDim(1));
        return raw_face_iterator();
      case 2:
        return end_line();
      case 3:
        return end_quad();
      default:
        Assert(false, ExcNotImplemented());
        return raw_face_iterator();
    }
}



template <int dim, int spacedim>
IteratorRange<typename Triangulation<dim, spacedim>::active_face_iterator>
Triangulation<dim, spacedim>::active_face_iterators() const
{
  return IteratorRange<
    typename Triangulation<dim, spacedim>::active_face_iterator>(
    begin_active_face(), end_face());
}

/*------------------------ Vertex iterator functions ------------------------*/


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::vertex_iterator
Triangulation<dim, spacedim>::begin_vertex() const
{
  vertex_iterator i =
    raw_vertex_iterator(const_cast<Triangulation<dim, spacedim> *>(this), 0, 0);
  if (i.state() != IteratorState::valid)
    return i;
  // This loop will end because every triangulation has used vertices.
  while (i->used() == false)
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_vertex_iterator
Triangulation<dim, spacedim>::begin_active_vertex() const
{
  // every vertex is active
  return begin_vertex();
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::vertex_iterator
Triangulation<dim, spacedim>::end_vertex() const
{
  return raw_vertex_iterator(const_cast<Triangulation<dim, spacedim> *>(this),
                             -1,
                             numbers::invalid_unsigned_int);
}



/*------------------------ Line iterator functions ------------------------*/



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::raw_line_iterator
Triangulation<dim, spacedim>::begin_raw_line(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just return the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end_line();
    }

  switch (dim)
    {
      case 1:
        // Query whether the given level is valid for the local portion of the
        // triangulation.
        Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));

        if (level >= levels.size() || levels[level]->cells.n_objects() == 0)
          return end_line();

        return raw_line_iterator(
          const_cast<Triangulation<dim, spacedim> *>(this), level, 0);

      default:
        Assert(level == 0, ExcFacesHaveNoLevel());
        return raw_line_iterator(
          const_cast<Triangulation<dim, spacedim> *>(this), 0, 0);
    }
}


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::line_iterator
Triangulation<dim, spacedim>::begin_line(const unsigned int level) const
{
  // level is checked in begin_raw
  raw_line_iterator ri = begin_raw_line(level);
  if (ri.state() != IteratorState::valid)
    return ri;
  while (ri->used() == false)
    if ((++ri).state() != IteratorState::valid)
      return ri;
  return ri;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_line_iterator
Triangulation<dim, spacedim>::begin_active_line(const unsigned int level) const
{
  // level is checked in begin_raw
  line_iterator i = begin_line(level);
  if (i.state() != IteratorState::valid)
    return i;
  while (i->has_children())
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::line_iterator
Triangulation<dim, spacedim>::end_line() const
{
  return raw_line_iterator(const_cast<Triangulation<dim, spacedim> *>(this),
                           -1,
                           -1);
}



/*------------------------ Quad iterator functions ------------------------*/


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::raw_quad_iterator
Triangulation<dim, spacedim>::begin_raw_quad(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just return the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end_quad();
    }

  switch (dim)
    {
      case 1:
        Assert(false, ExcImpossibleInDim(1));
        return raw_hex_iterator();
      case 2:
        {
          // Query whether the given level is valid for the local portion of the
          // triangulation.
          Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));

          if (level >= levels.size() || levels[level]->cells.n_objects() == 0)
            return end_quad();

          return raw_quad_iterator(
            const_cast<Triangulation<dim, spacedim> *>(this), level, 0);
        }

      case 3:
        {
          Assert(level == 0, ExcFacesHaveNoLevel());

          return raw_quad_iterator(
            const_cast<Triangulation<dim, spacedim> *>(this), 0, 0);
        }


      default:
        Assert(false, ExcNotImplemented());
        return raw_hex_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::quad_iterator
Triangulation<dim, spacedim>::begin_quad(const unsigned int level) const
{
  // level is checked in begin_raw
  raw_quad_iterator ri = begin_raw_quad(level);
  if (ri.state() != IteratorState::valid)
    return ri;
  while (ri->used() == false)
    if ((++ri).state() != IteratorState::valid)
      return ri;
  return ri;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_quad_iterator
Triangulation<dim, spacedim>::begin_active_quad(const unsigned int level) const
{
  // level is checked in begin_raw
  quad_iterator i = begin_quad(level);
  if (i.state() != IteratorState::valid)
    return i;
  while (i->has_children())
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::quad_iterator
Triangulation<dim, spacedim>::end_quad() const
{
  return raw_quad_iterator(const_cast<Triangulation<dim, spacedim> *>(this),
                           -1,
                           -1);
}


/*------------------------ Hex iterator functions ------------------------*/


template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::raw_hex_iterator
Triangulation<dim, spacedim>::begin_raw_hex(const unsigned int level) const
{
  // This function may be called on parallel triangulations on levels
  // that exist globally, but not on the local portion of the
  // triangulation. In that case, just return the end iterator.
  //
  // We need to use levels.size() instead of n_levels() because the
  // latter function uses the cache, but we need to be able to call
  // this function at a time when the cache is not currently up to
  // date.
  if (level >= levels.size())
    {
      Assert(level < n_global_levels(),
             ExcInvalidLevel(level, n_global_levels()));
      return end_hex();
    }

  switch (dim)
    {
      case 1:
      case 2:
        Assert(false, ExcImpossibleInDim(1));
        return raw_hex_iterator();
      case 3:
        {
          // Query whether the given level is valid for the local portion of the
          // triangulation.
          Assert(level < levels.size(), ExcInvalidLevel(level, levels.size()));

          if (level >= levels.size() || levels[level]->cells.n_objects() == 0)
            return end_hex();

          return raw_hex_iterator(
            const_cast<Triangulation<dim, spacedim> *>(this), level, 0);
        }

      default:
        Assert(false, ExcNotImplemented());
        return raw_hex_iterator();
    }
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::hex_iterator
Triangulation<dim, spacedim>::begin_hex(const unsigned int level) const
{
  // level is checked in begin_raw
  raw_hex_iterator ri = begin_raw_hex(level);
  if (ri.state() != IteratorState::valid)
    return ri;
  while (ri->used() == false)
    if ((++ri).state() != IteratorState::valid)
      return ri;
  return ri;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::active_hex_iterator
Triangulation<dim, spacedim>::begin_active_hex(const unsigned int level) const
{
  // level is checked in begin_raw
  hex_iterator i = begin_hex(level);
  if (i.state() != IteratorState::valid)
    return i;
  while (i->has_children())
    if ((++i).state() != IteratorState::valid)
      return i;
  return i;
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::hex_iterator
Triangulation<dim, spacedim>::end_hex() const
{
  return raw_hex_iterator(const_cast<Triangulation<dim, spacedim> *>(this),
                          -1,
                          -1);
}



// -------------------------------- number of cells etc ---------------


namespace internal
{
  namespace TriangulationImplementation
  {
    inline unsigned int
    n_cells(const internal::TriangulationImplementation::NumberCache<1> &c)
    {
      return c.n_lines;
    }


    inline unsigned int
    n_active_cells(
      const internal::TriangulationImplementation::NumberCache<1> &c)
    {
      return c.n_active_lines;
    }


    inline unsigned int
    n_cells(const internal::TriangulationImplementation::NumberCache<2> &c)
    {
      return c.n_quads;
    }


    inline unsigned int
    n_active_cells(
      const internal::TriangulationImplementation::NumberCache<2> &c)
    {
      return c.n_active_quads;
    }


    inline unsigned int
    n_cells(const internal::TriangulationImplementation::NumberCache<3> &c)
    {
      return c.n_hexes;
    }


    inline unsigned int
    n_active_cells(
      const internal::TriangulationImplementation::NumberCache<3> &c)
    {
      return c.n_active_hexes;
    }
  } // namespace TriangulationImplementation
} // namespace internal



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_cells() const
{
  return internal::TriangulationImplementation::n_cells(number_cache);
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_cells() const
{
  return internal::TriangulationImplementation::n_active_cells(number_cache);
}

template <int dim, int spacedim>
types::global_cell_index
Triangulation<dim, spacedim>::n_global_active_cells() const
{
  return n_active_cells();
}

template <int dim, int spacedim>
types::coarse_cell_id
Triangulation<dim, spacedim>::n_global_coarse_cells() const
{
  return n_cells(0);
}

template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_faces() const
{
  switch (dim)
    {
      case 1:
        return n_used_vertices();
      case 2:
        return n_lines();
      case 3:
        return n_quads();
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_faces() const
{
  switch (dim)
    {
      case 1:
        return n_vertices();
      case 2:
        return n_raw_lines();
      case 3:
        return n_raw_quads();
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_faces() const
{
  switch (dim)
    {
      case 1:
        return n_used_vertices();
      case 2:
        return n_active_lines();
      case 3:
        return n_active_quads();
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_cells(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return n_raw_lines(level);
      case 2:
        return n_raw_quads(level);
      case 3:
        return n_raw_hexs(level);
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_cells(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return n_lines(level);
      case 2:
        return n_quads(level);
      case 3:
        return n_hexs(level);
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_cells(const unsigned int level) const
{
  switch (dim)
    {
      case 1:
        return n_active_lines(level);
      case 2:
        return n_active_quads(level);
      case 3:
        return n_active_hexs(level);
      default:
        Assert(false, ExcNotImplemented());
    }
  return 0;
}


template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::has_hanging_nodes() const
{
  for (unsigned int lvl = 0; lvl < n_global_levels() - 1; ++lvl)
    if (n_active_cells(lvl) != 0)
      return true;

  return false;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_lines() const
{
  return number_cache.n_lines;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_lines(const unsigned int level) const
{
  if (dim == 1)
    {
      AssertIndexRange(level, n_levels());
      return levels[level]->cells.n_objects();
    }

  Assert(false, ExcFacesHaveNoLevel());
  return 0;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_lines() const
{
  if (dim == 1)
    {
      Assert(false, ExcNotImplemented());
      return 0;
    }

  return faces->lines.n_objects();
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_lines(const unsigned int level) const
{
  AssertIndexRange(level, number_cache.n_lines_level.size());
  Assert(dim == 1, ExcFacesHaveNoLevel());
  return number_cache.n_lines_level[level];
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_lines() const
{
  return number_cache.n_active_lines;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_lines(const unsigned int level) const
{
  AssertIndexRange(level, number_cache.n_lines_level.size());
  Assert(dim == 1, ExcFacesHaveNoLevel());

  return number_cache.n_active_lines_level[level];
}


template <>
unsigned int
Triangulation<1, 1>::n_quads() const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 1>::n_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 1>::n_raw_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 1>::n_raw_hexs(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 1>::n_active_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 1>::n_active_quads() const
{
  return 0;
}



template <>
unsigned int
Triangulation<1, 2>::n_quads() const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 2>::n_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 2>::n_raw_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 2>::n_raw_hexs(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 2>::n_active_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 2>::n_active_quads() const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_quads() const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_raw_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_raw_hexs(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_active_quads(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<1, 3>::n_active_quads() const
{
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_quads() const
{
  return number_cache.n_quads;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_quads(const unsigned int level) const
{
  Assert(dim == 2, ExcFacesHaveNoLevel());
  AssertIndexRange(level, number_cache.n_quads_level.size());
  return number_cache.n_quads_level[level];
}



template <>
unsigned int
Triangulation<2, 2>::n_raw_quads(const unsigned int level) const
{
  AssertIndexRange(level, n_levels());
  return levels[level]->cells.n_objects();
}



template <>
unsigned int
Triangulation<2, 3>::n_raw_quads(const unsigned int level) const
{
  AssertIndexRange(level, n_levels());
  return levels[level]->cells.n_objects();
}


template <>
unsigned int
Triangulation<3, 3>::n_raw_quads(const unsigned int) const
{
  Assert(false, ExcFacesHaveNoLevel());
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_quads() const
{
  Assert(false, ExcNotImplemented());
  return 0;
}



template <>
unsigned int
Triangulation<3, 3>::n_raw_quads() const
{
  return faces->quads.n_objects();
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_quads() const
{
  return number_cache.n_active_quads;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_quads(const unsigned int level) const
{
  AssertIndexRange(level, number_cache.n_quads_level.size());
  Assert(dim == 2, ExcFacesHaveNoLevel());

  return number_cache.n_active_quads_level[level];
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_hexs() const
{
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_hexs(const unsigned int) const
{
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_raw_hexs(const unsigned int) const
{
  return 0;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_hexs() const
{
  return 0;
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_active_hexs(const unsigned int) const
{
  return 0;
}


template <>
unsigned int
Triangulation<3, 3>::n_hexs() const
{
  return number_cache.n_hexes;
}



template <>
unsigned int
Triangulation<3, 3>::n_hexs(const unsigned int level) const
{
  AssertIndexRange(level, number_cache.n_hexes_level.size());

  return number_cache.n_hexes_level[level];
}



template <>
unsigned int
Triangulation<3, 3>::n_raw_hexs(const unsigned int level) const
{
  AssertIndexRange(level, n_levels());
  return levels[level]->cells.n_objects();
}


template <>
unsigned int
Triangulation<3, 3>::n_active_hexs() const
{
  return number_cache.n_active_hexes;
}



template <>
unsigned int
Triangulation<3, 3>::n_active_hexs(const unsigned int level) const
{
  AssertIndexRange(level, number_cache.n_hexes_level.size());

  return number_cache.n_active_hexes_level[level];
}



template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::n_used_vertices() const
{
  return std::count(vertices_used.begin(), vertices_used.end(), true);
}



template <int dim, int spacedim>
const std::vector<bool> &
Triangulation<dim, spacedim>::get_used_vertices() const
{
  return vertices_used;
}



template <>
unsigned int
Triangulation<1, 1>::max_adjacent_cells() const
{
  return 2;
}



template <>
unsigned int
Triangulation<1, 2>::max_adjacent_cells() const
{
  return 2;
}


template <>
unsigned int
Triangulation<1, 3>::max_adjacent_cells() const
{
  return 2;
}


template <int dim, int spacedim>
unsigned int
Triangulation<dim, spacedim>::max_adjacent_cells() const
{
  cell_iterator cell = begin(0),
                endc = (n_levels() > 1 ? begin(1) : cell_iterator(end()));
  // store the largest index of the
  // vertices used on level 0
  unsigned int max_vertex_index = 0;
  for (; cell != endc; ++cell)
    for (const unsigned int vertex : GeometryInfo<dim>::vertex_indices())
      if (cell->vertex_index(vertex) > max_vertex_index)
        max_vertex_index = cell->vertex_index(vertex);

  // store the number of times a cell
  // touches a vertex. An unsigned
  // int should suffice, even for
  // larger dimensions
  std::vector<unsigned short int> usage_count(max_vertex_index + 1, 0);
  // touch a vertex's usage count
  // every time we find an adjacent
  // element
  for (cell = begin(); cell != endc; ++cell)
    for (const unsigned int vertex : GeometryInfo<dim>::vertex_indices())
      ++usage_count[cell->vertex_index(vertex)];

  return std::max(GeometryInfo<dim>::vertices_per_cell,
                  static_cast<unsigned int>(
                    *std::max_element(usage_count.begin(), usage_count.end())));
}



template <int dim, int spacedim>
types::subdomain_id
Triangulation<dim, spacedim>::locally_owned_subdomain() const
{
  return numbers::invalid_subdomain_id;
}



template <int dim, int spacedim>
Triangulation<dim, spacedim> &
Triangulation<dim, spacedim>::get_triangulation()
{
  return *this;
}



template <int dim, int spacedim>
const Triangulation<dim, spacedim> &
Triangulation<dim, spacedim>::get_triangulation() const
{
  return *this;
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::add_periodicity(
  const std::vector<GridTools::PeriodicFacePair<cell_iterator>>
    &periodicity_vector)
{
  periodic_face_pairs_level_0.insert(periodic_face_pairs_level_0.end(),
                                     periodicity_vector.begin(),
                                     periodicity_vector.end());

  // Now initialize periodic_face_map
  update_periodic_face_map();
}



template <int dim, int spacedim>
const typename std::map<
  std::pair<typename Triangulation<dim, spacedim>::cell_iterator, unsigned int>,
  std::pair<std::pair<typename Triangulation<dim, spacedim>::cell_iterator,
                      unsigned int>,
            std::bitset<3>>> &
Triangulation<dim, spacedim>::get_periodic_face_map() const
{
  return periodic_face_map;
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::execute_coarsening_and_refinement()
{
  // Call our version of prepare_coarsening_and_refinement() even if a derived
  // class like parallel::distributed::Triangulation overrides it. Their
  // function will be called in their execute_coarsening_and_refinement()
  // function. Even in a distributed computation our job here is to reconstruct
  // the local part of the mesh and as such checking our flags is enough.
  Triangulation<dim, spacedim>::prepare_coarsening_and_refinement();

  // verify a case with which we have had
  // some difficulty in the past (see the
  // deal.II/coarsening_* tests)
  if (smooth_grid & limit_level_difference_at_vertices)
    Assert(satisfies_level1_at_vertex_rule(*this) == true, ExcInternalError());

  // Inform all listeners about beginning of refinement.
  signals.pre_refinement();

  execute_coarsening();

  const DistortedCellList cells_with_distorted_children = execute_refinement();

  reset_cell_vertex_indices_cache();

  // verify a case with which we have had
  // some difficulty in the past (see the
  // deal.II/coarsening_* tests)
  if (smooth_grid & limit_level_difference_at_vertices)
    Assert(satisfies_level1_at_vertex_rule(*this) == true, ExcInternalError());

  // finally build up neighbor connectivity information, and set
  // active cell indices
  this->policy->update_neighbors(*this);
  reset_active_cell_indices();

  reset_global_cell_indices(); // TODO: better place?

  // Inform all listeners about end of refinement.
  signals.post_refinement();

  AssertThrow(cells_with_distorted_children.distorted_cells.size() == 0,
              cells_with_distorted_children);

  update_periodic_face_map();
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_active_cell_indices()
{
  unsigned int active_cell_index = 0;
  for (raw_cell_iterator cell = begin_raw(); cell != end(); ++cell)
    if ((cell->used() == false) || cell->has_children())
      cell->set_active_cell_index(numbers::invalid_unsigned_int);
    else
      {
        cell->set_active_cell_index(active_cell_index);
        ++active_cell_index;
      }

  Assert(active_cell_index == n_active_cells(), ExcInternalError());
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_global_cell_indices()
{
  {
    types::global_cell_index cell_index = 0;
    for (const auto &cell : active_cell_iterators())
      cell->set_global_active_cell_index(cell_index++);
  }

  for (unsigned int l = 0; l < levels.size(); ++l)
    {
      types::global_cell_index cell_index = 0;
      for (const auto &cell : cell_iterators_on_level(l))
        cell->set_global_level_cell_index(cell_index++);
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::reset_cell_vertex_indices_cache()
{
  for (unsigned int l = 0; l < levels.size(); ++l)
    {
      constexpr unsigned int     max_vertices_per_cell = 1 << dim;
      std::vector<unsigned int> &cache = levels[l]->cell_vertex_indices_cache;
      cache.clear();
      cache.resize(levels[l]->refine_flags.size() * max_vertices_per_cell,
                   numbers::invalid_unsigned_int);
      for (const auto &cell : cell_iterators_on_level(l))
        {
          const unsigned int my_index = cell->index() * max_vertices_per_cell;
          for (const unsigned int i : cell->vertex_indices())
            cache[my_index + i] = internal::TriaAccessorImplementation::
              Implementation::vertex_index(*cell, i);
        }
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::update_periodic_face_map()
{
  // first empty the currently stored objects
  periodic_face_map.clear();

  typename std::vector<
    GridTools::PeriodicFacePair<cell_iterator>>::const_iterator it;
  for (it = periodic_face_pairs_level_0.begin();
       it != periodic_face_pairs_level_0.end();
       ++it)
    {
      update_periodic_face_map_recursively<dim, spacedim>(it->cell[0],
                                                          it->cell[1],
                                                          it->face_idx[0],
                                                          it->face_idx[1],
                                                          it->orientation,
                                                          periodic_face_map);

      // for the other way, we need to invert the orientation
      std::bitset<3> inverted_orientation;
      {
        bool orientation, flip, rotation;
        orientation = it->orientation[0];
        rotation    = it->orientation[2];
        flip = orientation ? rotation ^ it->orientation[1] : it->orientation[1];
        inverted_orientation[0] = orientation;
        inverted_orientation[1] = flip;
        inverted_orientation[2] = rotation;
      }
      update_periodic_face_map_recursively<dim, spacedim>(it->cell[1],
                                                          it->cell[0],
                                                          it->face_idx[1],
                                                          it->face_idx[0],
                                                          inverted_orientation,
                                                          periodic_face_map);
    }

  // check consistency
  typename std::map<std::pair<cell_iterator, unsigned int>,
                    std::pair<std::pair<cell_iterator, unsigned int>,
                              std::bitset<3>>>::const_iterator it_test;
  for (it_test = periodic_face_map.begin(); it_test != periodic_face_map.end();
       ++it_test)
    {
      const Triangulation<dim, spacedim>::cell_iterator cell_1 =
        it_test->first.first;
      const Triangulation<dim, spacedim>::cell_iterator cell_2 =
        it_test->second.first.first;
      if (cell_1->level() == cell_2->level())
        {
          // if both cells have the same neighbor, then the same pair
          // order swapped has to be in the map
          Assert(periodic_face_map[it_test->second.first].first ==
                   it_test->first,
                 ExcInternalError());
        }
    }
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::update_reference_cells()
{
  std::set<ReferenceCell> reference_cells_set;
  for (auto cell : active_cell_iterators())
    if (cell->is_locally_owned())
      reference_cells_set.insert(cell->reference_cell());

  std::vector<ReferenceCell> reference_cells(reference_cells_set.begin(),
                                             reference_cells_set.end());

  this->reference_cells = reference_cells;
}



template <int dim, int spacedim>
const std::vector<ReferenceCell> &
Triangulation<dim, spacedim>::get_reference_cells() const
{
  return this->reference_cells;
}



template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::all_reference_cells_are_hyper_cube() const
{
  Assert(this->reference_cells.size() > 0,
         ExcMessage("You can't ask about the kinds of reference "
                    "cells used by this triangulation if the "
                    "triangulation doesn't yet have any cells in it."));
  return (this->reference_cells.size() == 1 &&
          this->reference_cells[0].is_hyper_cube());
}



template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::all_reference_cells_are_simplex() const
{
  Assert(this->reference_cells.size() > 0,
         ExcMessage("You can't ask about the kinds of reference "
                    "cells used by this triangulation if the "
                    "triangulation doesn't yet have any cells in it."));
  return (this->reference_cells.size() == 1 &&
          this->reference_cells[0].is_simplex());
}



template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::is_mixed_mesh() const
{
  Assert(this->reference_cells.size() > 0,
         ExcMessage("You can't ask about the kinds of reference "
                    "cells used by this triangulation if the "
                    "triangulation doesn't yet have any cells in it."));
  return reference_cells.size() > 1 ||
         ((reference_cells[0].is_hyper_cube() == false) &&
          (reference_cells[0].is_simplex() == false));
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::clear_despite_subscriptions()
{
  levels.clear();
  faces.reset();

  vertices.clear();
  vertices_used.clear();

  manifolds.clear();

  number_cache = internal::TriangulationImplementation::NumberCache<dim>();
}



template <int dim, int spacedim>
typename Triangulation<dim, spacedim>::DistortedCellList
Triangulation<dim, spacedim>::execute_refinement()
{
  const DistortedCellList cells_with_distorted_children =
    this->policy->execute_refinement(*this, check_for_distorted_cells);



  // re-compute number of lines
  internal::TriangulationImplementation::Implementation::compute_number_cache(
    *this, levels.size(), number_cache);

#ifdef DEBUG
  for (const auto &level : levels)
    monitor_memory(level->cells, dim);

  // check whether really all refinement flags are reset (also of
  // previously non-active cells which we may not have touched. If the
  // refinement flag of a non-active cell is set, something went wrong
  // since the cell-accessors should have caught this)
  for (const auto &cell : this->cell_iterators())
    Assert(!cell->refine_flag_set(), ExcInternalError());
#endif

  return cells_with_distorted_children;
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::execute_coarsening()
{
  // create a vector counting for each line how many cells contain
  // this line. in 3D, this is used later on to decide which lines can
  // be deleted after coarsening a cell. in other dimensions it will
  // be ignored
  std::vector<unsigned int> line_cell_count =
    count_cells_bounded_by_line(*this);
  std::vector<unsigned int> quad_cell_count =
    count_cells_bounded_by_quad(*this);

  // loop over all cells. Flag all cells of which all children are
  // flagged for coarsening and delete the childrens' flags. In
  // effect, only those cells are flagged of which originally all
  // children were flagged and for which all children are on the same
  // refinement level. For flagging, the user flags are used, to avoid
  // confusion and because non-active cells can't be flagged for
  // coarsening. Note that because of the effects of
  // @p{fix_coarsen_flags}, of a cell either all or no children must
  // be flagged for coarsening, so it is ok to only check the first
  // child
  clear_user_flags();

  for (const auto &cell : this->cell_iterators())
    if (!cell->is_active())
      if (cell->child(0)->coarsen_flag_set())
        {
          cell->set_user_flag();
          for (unsigned int child = 0; child < cell->n_children(); ++child)
            {
              Assert(cell->child(child)->coarsen_flag_set(),
                     ExcInternalError());
              cell->child(child)->clear_coarsen_flag();
            }
        }


  // now do the actual coarsening step. Since the loop goes over used
  // cells we only need not worry about deleting some cells since the
  // ++operator will then just hop over them if we should hit one. Do
  // the loop in the reverse way since we may only delete some cells
  // if their neighbors have already been deleted (if the latter are
  // on a higher level for example)
  //
  // since we delete the *children* of cells, we can ignore cells
  // on the highest level, i.e., level must be less than or equal
  // to n_levels()-2.
  cell_iterator cell = begin(), endc = end();
  if (levels.size() >= 2)
    for (cell = last(); cell != endc; --cell)
      if (cell->level() <= static_cast<int>(levels.size() - 2) &&
          cell->user_flag_set())
        {
          // inform all listeners that cell coarsening is going to happen
          signals.pre_coarsening_on_cell(cell);
          // use a separate function, since this is dimension specific
          this->policy->delete_children(*this,
                                        cell,
                                        line_cell_count,
                                        quad_cell_count);
        }

  // re-compute number of lines and quads
  internal::TriangulationImplementation::Implementation::compute_number_cache(
    *this, levels.size(), number_cache);

  // in principle no user flags should be set any more at this point
#if DEBUG
  for (cell = begin(); cell != endc; ++cell)
    Assert(cell->user_flag_set() == false, ExcInternalError());
#endif
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::fix_coarsen_flags()
{
  // copy a piece of code from prepare_coarsening_and_refinement that
  // ensures that the level difference at vertices is limited if so
  // desired. we need this code here since at least in 1d we don't
  // call the dimension-independent version of
  // prepare_coarsening_and_refinement function. in 2d and 3d, having
  // this hunk here makes our lives a bit easier as well as it takes
  // care of these cases earlier than it would otherwise happen.
  //
  // the main difference to the code in p_c_and_r is that here we
  // absolutely have to make sure that we get things right, i.e. that
  // in particular we set flags right if
  // limit_level_difference_at_vertices is set. to do so we iterate
  // until the flags don't change any more
  std::vector<bool> previous_coarsen_flags(n_active_cells());
  save_coarsen_flags(previous_coarsen_flags);

  std::vector<int> vertex_level(vertices.size(), 0);

  bool continue_iterating = true;

  do
    {
      if (smooth_grid & limit_level_difference_at_vertices)
        {
          Assert(!anisotropic_refinement,
                 ExcMessage("In case of anisotropic refinement the "
                            "limit_level_difference_at_vertices flag for "
                            "mesh smoothing must not be set!"));

          // store highest level one of the cells adjacent to a vertex
          // belongs to
          std::fill(vertex_level.begin(), vertex_level.end(), 0);
          for (const auto &cell : this->active_cell_iterators())
            {
              if (cell->refine_flag_set())
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  vertex_level[cell->vertex_index(vertex)] =
                    std::max(vertex_level[cell->vertex_index(vertex)],
                             cell->level() + 1);
              else if (!cell->coarsen_flag_set())
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  vertex_level[cell->vertex_index(vertex)] =
                    std::max(vertex_level[cell->vertex_index(vertex)],
                             cell->level());
              else
                {
                  // if coarsen flag is set then tentatively assume
                  // that the cell will be coarsened. this isn't
                  // always true (the coarsen flag could be removed
                  // again) and so we may make an error here. we try
                  // to correct this by iterating over the entire
                  // process until we are converged
                  Assert(cell->coarsen_flag_set(), ExcInternalError());
                  for (const unsigned int vertex :
                       GeometryInfo<dim>::vertex_indices())
                    vertex_level[cell->vertex_index(vertex)] =
                      std::max(vertex_level[cell->vertex_index(vertex)],
                               cell->level() - 1);
                }
            }


          // loop over all cells in reverse order. do so because we
          // can then update the vertex levels on the adjacent
          // vertices and maybe already flag additional cells in this
          // loop
          //
          // note that not only may we have to add additional
          // refinement flags, but we will also have to remove
          // coarsening flags on cells adjacent to vertices that will
          // see refinement
          active_cell_iterator cell = begin_active(), endc = end();
          for (cell = last_active(); cell != endc; --cell)
            if (cell->refine_flag_set() == false)
              {
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  if (vertex_level[cell->vertex_index(vertex)] >=
                      cell->level() + 1)
                    {
                      // remove coarsen flag...
                      cell->clear_coarsen_flag();

                      // ...and if necessary also refine the current
                      // cell, at the same time updating the level
                      // information about vertices
                      if (vertex_level[cell->vertex_index(vertex)] >
                          cell->level() + 1)
                        {
                          cell->set_refine_flag();

                          for (const unsigned int v :
                               GeometryInfo<dim>::vertex_indices())
                            vertex_level[cell->vertex_index(v)] =
                              std::max(vertex_level[cell->vertex_index(v)],
                                       cell->level() + 1);
                        }

                      // continue and see whether we may, for example,
                      // go into the inner 'if' above based on a
                      // different vertex
                    }
              }
        }

      // loop over all cells. Flag all cells of which all children are
      // flagged for coarsening and delete the childrens' flags. Also
      // delete all flags of cells for which not all children of a
      // cell are flagged. In effect, only those cells are flagged of
      // which originally all children were flagged and for which all
      // children are on the same refinement level. For flagging, the
      // user flags are used, to avoid confusion and because
      // non-active cells can't be flagged for coarsening
      //
      // In effect, all coarsen flags are turned into user flags of
      // the mother cell if coarsening is possible or deleted
      // otherwise.
      clear_user_flags();
      // Coarsen flags of cells with no mother cell, i.e. on the
      // coarsest level are deleted explicitly.
      for (const auto &acell : this->active_cell_iterators_on_level(0))
        acell->clear_coarsen_flag();

      for (const auto &cell : this->cell_iterators())
        {
          // nothing to do if we are already on the finest level
          if (cell->is_active())
            continue;

          const unsigned int n_children       = cell->n_children();
          unsigned int       flagged_children = 0;
          for (unsigned int child = 0; child < n_children; ++child)
            if (cell->child(child)->is_active() &&
                cell->child(child)->coarsen_flag_set())
              {
                ++flagged_children;
                // clear flag since we don't need it anymore
                cell->child(child)->clear_coarsen_flag();
              }

          // flag this cell for coarsening if all children were
          // flagged
          if (flagged_children == n_children)
            cell->set_user_flag();
        }

        // in principle no coarsen flags should be set any more at this
        // point
#if DEBUG
      for (auto &cell : this->cell_iterators())
        Assert(cell->coarsen_flag_set() == false, ExcInternalError());
#endif

      // now loop over all cells which have the user flag set. their
      // children were flagged for coarsening. set the coarsen flag
      // again if we are sure that none of the neighbors of these
      // children are refined, or will be refined, since then we would
      // get a two-level jump in refinement. on the other hand, if one
      // of the children's neighbors has their user flag set, then we
      // know that its children will go away by coarsening, and we
      // will be ok.
      //
      // note on the other hand that we do allow level-2 jumps in
      // refinement between neighbors in 1d, so this whole procedure
      // is only necessary if we are not in 1d
      //
      // since we remove some coarsening/user flags in the process, we
      // have to work from the finest level to the coarsest one, since
      // we occasionally inspect user flags of cells on finer levels
      // and need to be sure that these flags are final
      cell_iterator cell = begin(), endc = end();
      for (cell = last(); cell != endc; --cell)
        if (cell->user_flag_set())
          // if allowed: flag the
          // children for coarsening
          if (this->policy->coarsening_allowed(cell))
            for (unsigned int c = 0; c < cell->n_children(); ++c)
              {
                Assert(cell->child(c)->refine_flag_set() == false,
                       ExcInternalError());

                cell->child(c)->set_coarsen_flag();
              }

      // clear all user flags again, now that we don't need them any
      // more
      clear_user_flags();


      // now see if anything has changed in the last iteration of this
      // function
      std::vector<bool> current_coarsen_flags(n_active_cells());
      save_coarsen_flags(current_coarsen_flags);

      continue_iterating = (current_coarsen_flags != previous_coarsen_flags);
      previous_coarsen_flags = current_coarsen_flags;
    }
  while (continue_iterating == true);
}


// TODO: merge the following 3 functions since they are the same
template <>
bool
Triangulation<1, 1>::prepare_coarsening_and_refinement()
{
  // save the flags to determine whether something was changed in the
  // course of this function
  std::vector<bool> flags_before;
  save_coarsen_flags(flags_before);

  // do nothing in 1d, except setting the coarsening flags correctly
  fix_coarsen_flags();

  std::vector<bool> flags_after;
  save_coarsen_flags(flags_after);

  return (flags_before != flags_after);
}


template <>
bool
Triangulation<1, 2>::prepare_coarsening_and_refinement()
{
  // save the flags to determine whether something was changed in the
  // course of this function
  std::vector<bool> flags_before;
  save_coarsen_flags(flags_before);

  // do nothing in 1d, except setting the coarsening flags correctly
  fix_coarsen_flags();

  std::vector<bool> flags_after;
  save_coarsen_flags(flags_after);

  return (flags_before != flags_after);
}


template <>
bool
Triangulation<1, 3>::prepare_coarsening_and_refinement()
{
  // save the flags to determine whether something was changed in the
  // course of this function
  std::vector<bool> flags_before;
  save_coarsen_flags(flags_before);

  // do nothing in 1d, except setting the coarsening flags correctly
  fix_coarsen_flags();

  std::vector<bool> flags_after;
  save_coarsen_flags(flags_after);

  return (flags_before != flags_after);
}



namespace
{
  // check if the given @param cell marked for coarsening would
  // produce an unrefined island. To break up long chains of these
  // cells we recursively check our neighbors in case we change this
  // cell. This reduces the number of outer iterations dramatically.
  template <int dim, int spacedim>
  void
  possibly_do_not_produce_unrefined_islands(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell)
  {
    Assert(cell->has_children(), ExcInternalError());

    unsigned int n_neighbors = 0;
    // count all neighbors that will be refined along the face of our
    // cell after the next step
    unsigned int count = 0;
    for (unsigned int n : GeometryInfo<dim>::face_indices())
      {
        const typename Triangulation<dim, spacedim>::cell_iterator neighbor =
          cell->neighbor(n);
        if (neighbor.state() == IteratorState::valid)
          {
            ++n_neighbors;
            if (face_will_be_refined_by_neighbor(cell, n))
              ++count;
          }
      }
    // clear coarsen flags if either all existing neighbors will be
    // refined or all but one will be and the cell is in the interior
    // of the domain
    if (count == n_neighbors ||
        (count >= n_neighbors - 1 &&
         n_neighbors == GeometryInfo<dim>::faces_per_cell))
      {
        for (unsigned int c = 0; c < cell->n_children(); ++c)
          cell->child(c)->clear_coarsen_flag();

        for (const unsigned int face : GeometryInfo<dim>::face_indices())
          if (!cell->at_boundary(face) &&
              (!cell->neighbor(face)->is_active()) &&
              (cell_will_be_coarsened(cell->neighbor(face))))
            possibly_do_not_produce_unrefined_islands<dim, spacedim>(
              cell->neighbor(face));
      }
  }


  // see if the current cell needs to be refined to avoid unrefined
  // islands.
  //
  // there are sometimes chains of cells that induce refinement of
  // each other. to avoid running the loop in
  // prepare_coarsening_and_refinement over and over again for each
  // one of them, at least for the isotropic refinement case we seek
  // to flag neighboring elements as well as necessary. this takes
  // care of (slightly pathological) cases like
  // deal.II/mesh_smoothing_03
  template <int dim, int spacedim>
  void
  possibly_refine_unrefined_island(
    const typename Triangulation<dim, spacedim>::cell_iterator &cell,
    const bool allow_anisotropic_smoothing)
  {
    Assert(cell->is_active(), ExcInternalError());
    Assert(cell->refine_flag_set() == false, ExcInternalError());


    // now we provide two algorithms. the first one is the standard
    // one, coming from the time, where only isotropic refinement was
    // possible. it simply counts the neighbors that are or will be
    // refined and compares to the number of other ones. the second
    // one does this check independently for each direction: if all
    // neighbors in one direction (normally two, at the boundary only
    // one) are refined, the current cell is flagged to be refined in
    // an according direction.

    if (allow_anisotropic_smoothing == false)
      {
        // use first algorithm
        unsigned int refined_neighbors = 0, unrefined_neighbors = 0;
        for (const unsigned int face : GeometryInfo<dim>::face_indices())
          if (!cell->at_boundary(face))
            {
              if (face_will_be_refined_by_neighbor(cell, face))
                ++refined_neighbors;
              else
                ++unrefined_neighbors;
            }

        if (unrefined_neighbors < refined_neighbors)
          {
            cell->clear_coarsen_flag();
            cell->set_refine_flag();

            // ok, so now we have flagged this cell. if we know that
            // there were any unrefined neighbors at all, see if any
            // of those will have to be refined as well
            if (unrefined_neighbors > 0)
              for (const unsigned int face : GeometryInfo<dim>::face_indices())
                if (!cell->at_boundary(face) &&
                    (face_will_be_refined_by_neighbor(cell, face) == false) &&
                    (cell->neighbor(face)->has_children() == false) &&
                    (cell->neighbor(face)->refine_flag_set() == false))
                  possibly_refine_unrefined_island<dim, spacedim>(
                    cell->neighbor(face), allow_anisotropic_smoothing);
          }
      }
    else
      {
        // variable to store the cell refine case needed to fulfill
        // all smoothing requirements
        RefinementCase<dim> smoothing_cell_refinement_case =
          RefinementCase<dim>::no_refinement;

        // use second algorithm, do the check individually for each
        // direction
        for (unsigned int face_pair = 0;
             face_pair < GeometryInfo<dim>::faces_per_cell / 2;
             ++face_pair)
          {
            // variable to store the cell refine case needed to refine
            // at the current face pair in the same way as the
            // neighbors do...
            RefinementCase<dim> directional_cell_refinement_case =
              RefinementCase<dim>::isotropic_refinement;

            for (unsigned int face_index = 0; face_index < 2; ++face_index)
              {
                unsigned int face = 2 * face_pair + face_index;
                // variable to store the refine case (to come) of the
                // face under consideration
                RefinementCase<dim - 1> expected_face_ref_case =
                  RefinementCase<dim - 1>::no_refinement;

                if (cell->neighbor(face).state() == IteratorState::valid)
                  face_will_be_refined_by_neighbor<dim, spacedim>(
                    cell, face, expected_face_ref_case);
                // now extract which refine case would be necessary to
                // achieve the same face refinement. set the
                // intersection with other requirements for the same
                // direction.

                // note: using the intersection is not an obvious
                // decision, we could also argue that it is more
                // natural to use the union. however, intersection is
                // the less aggressive tactic and favours a smaller
                // number of refined cells over an intensive
                // smoothing. this way we try not to lose too much of
                // the effort we put in anisotropic refinement
                // indicators due to overly aggressive smoothing...
                directional_cell_refinement_case =
                  (directional_cell_refinement_case &
                   GeometryInfo<dim>::
                     min_cell_refinement_case_for_face_refinement(
                       expected_face_ref_case,
                       face,
                       cell->face_orientation(face),
                       cell->face_flip(face),
                       cell->face_rotation(face)));
              } // for both face indices
            // if both requirements sum up to something useful, add
            // this to the refine case for smoothing. note: if
            // directional_cell_refinement_case is isotropic still,
            // then something went wrong...
            Assert(directional_cell_refinement_case <
                     RefinementCase<dim>::isotropic_refinement,
                   ExcInternalError());
            smoothing_cell_refinement_case =
              smoothing_cell_refinement_case | directional_cell_refinement_case;
          } // for all face_pairs
        // no we collected contributions from all directions. combine
        // the new flags with the existing refine case, but only if
        // smoothing is required
        if (smoothing_cell_refinement_case)
          {
            cell->clear_coarsen_flag();
            cell->set_refine_flag(cell->refine_flag_set() |
                                  smoothing_cell_refinement_case);
          }
      }
  }
} // namespace


template <int dim, int spacedim>
bool
Triangulation<dim, spacedim>::prepare_coarsening_and_refinement()
{
  // save the flags to determine whether something was changed in the
  // course of this function
  std::vector<bool> flags_before[2];
  save_coarsen_flags(flags_before[0]);
  save_refine_flags(flags_before[1]);

  // save the flags at the outset of each loop. we do so in order to
  // find out whether something was changed in the present loop, in
  // which case we would have to re-run the loop. the other
  // possibility to find this out would be to set a flag
  // @p{something_changed} to true each time we change something.
  // however, sometimes one change in one of the parts of the loop is
  // undone by another one, so we might end up in an endless loop. we
  // could be tempted to break this loop at an arbitrary number of
  // runs, but that would not be a clean solution, since we would
  // either have to 1/ break the loop too early, in which case the
  // promise that a second call to this function immediately after the
  // first one does not change anything, would be broken, or 2/ we do
  // as many loops as there are levels. we know that information is
  // transported over one level in each run of the loop, so this is
  // enough. Unfortunately, each loop is rather expensive, so we chose
  // the way presented here
  std::vector<bool> flags_before_loop[2] = {flags_before[0], flags_before[1]};

  // now for what is done in each loop: we have to fulfill several
  // tasks at the same time, namely several mesh smoothing algorithms
  // and mesh regularization, by which we mean that the next mesh
  // fulfills several requirements such as no double refinement at
  // each face or line, etc.
  //
  // since doing these things at once seems almost impossible (in the
  // first year of this library, they were done in two functions, one
  // for refinement and one for coarsening, and most things within
  // these were done at once, so the code was rather impossible to
  // join into this, only, function), we do them one after each
  // other. the order in which we do them is such that the important
  // tasks, namely regularization, are done last and the least
  // important things are done the first. the following order is
  // chosen:
  //
  // 0/ Only if coarsest_level_1 or patch_level_1 is set: clear all
  //    coarsen flags on level 1 to avoid level 0 cells being created
  //    by coarsening.  As coarsen flags will never be added, this can
  //    be done once and for all before the actual loop starts.
  //
  // 1/ do not coarsen a cell if 'most of the neighbors' will be
  //    refined after the step. This is to prevent occurrence of
  //    unrefined islands.
  //
  // 2/ eliminate refined islands in the interior and at the
  //    boundary. since they don't do much harm besides increasing the
  //    number of degrees of freedom, doing this has a rather low
  //    priority.
  //
  // 3/ limit the level difference of neighboring cells at each
  //    vertex.
  //
  // 4/ eliminate unrefined islands. this has higher priority since
  //    this diminishes the approximation properties not only of the
  //    unrefined island, but also of the surrounding patch.
  //
  // 5/ ensure patch level 1. Then the triangulation consists of
  //    patches, i.e. of cells that are refined once. It follows that
  //    if at least one of the children of a cell is or will be
  //    refined than all children need to be refined. This step only
  //    sets refinement flags and does not set coarsening flags.  If
  //    the patch_level_1 flag is set, then
  //    eliminate_unrefined_islands, eliminate_refined_inner_islands
  //    and eliminate_refined_boundary_islands will be fulfilled
  //    automatically and do not need to be enforced separately.
  //
  // 6/ take care of the requirement that no double refinement is done
  //    at each face
  //
  // 7/ take care that no double refinement is done at each line in 3d
  //    or higher dimensions.
  //
  // 8/ make sure that all children of each cell are either flagged
  //    for coarsening or none of the children is
  //
  // For some of these steps, it is known that they interact. Namely,
  // it is not possible to guarantee that after step 6 another step 5
  // would have no effect; the same holds for the opposite order and
  // also when taking into account step 7. however, it is important to
  // guarantee that step five or six do not undo something that step 5
  // did, and step 7 not something of step 6, otherwise the
  // requirements will not be satisfied even if the loop
  // terminates. this is accomplished by the fact that steps 5 and 6
  // only *add* refinement flags and delete coarsening flags
  // (therefore, step 6 can't undo something that step 4 already did),
  // and step 7 only deletes coarsening flags, never adds some. step 7
  // needs also take care that it won't tag cells for refinement for
  // which some neighbors are more refined or will be refined.

  //------------------------------------
  // STEP 0:
  //    Only if coarsest_level_1 or patch_level_1 is set: clear all
  //    coarsen flags on level 1 to avoid level 0 cells being created
  //    by coarsening.
  if (((smooth_grid & coarsest_level_1) || (smooth_grid & patch_level_1)) &&
      n_levels() >= 2)
    {
      for (const auto &cell : active_cell_iterators_on_level(1))
        cell->clear_coarsen_flag();
    }

  bool mesh_changed_in_this_loop = false;
  do
    {
      //------------------------------------
      // STEP 1:
      //    do not coarsen a cell if 'most of the neighbors' will be
      //    refined after the step. This is to prevent the occurrence
      //    of unrefined islands.  If patch_level_1 is set, this will
      //    be automatically fulfilled.
      if (smooth_grid & do_not_produce_unrefined_islands &&
          !(smooth_grid & patch_level_1))
        {
          for (const auto &cell : cell_iterators())
            {
              // only do something if this
              // cell will be coarsened
              if (!cell->is_active() && cell_will_be_coarsened(cell))
                possibly_do_not_produce_unrefined_islands<dim, spacedim>(cell);
            }
        }


      //------------------------------------
      // STEP 2:
      //    eliminate refined islands in the interior and at the
      //    boundary. since they don't do much harm besides increasing
      //    the number of degrees of freedom, doing this has a rather
      //    low priority.  If patch_level_1 is set, this will be
      //    automatically fulfilled.
      //
      //    there is one corner case to consider: if this is a
      //    distributed triangulation, there may be refined islands on
      //    the boundary of which we own only part (e.g. a single cell
      //    in the corner of a domain). the rest of the island is
      //    ghost cells and it *looks* like the area around it
      //    (artificial cells) are coarser but this is only because
      //    they may actually be equally fine on other
      //    processors. it's hard to detect this case but we can do
      //    the following: only set coarsen flags to remove this
      //    refined island if all cells we want to set flags on are
      //    locally owned
      if (smooth_grid & (eliminate_refined_inner_islands |
                         eliminate_refined_boundary_islands) &&
          !(smooth_grid & patch_level_1))
        {
          for (const auto &cell : cell_iterators())
            if (!cell->is_active() ||
                (cell->is_active() && cell->refine_flag_set() &&
                 cell->is_locally_owned()))
              {
                // check whether all children are active, i.e. not
                // refined themselves. This is a precondition that the
                // children may be coarsened away. If the cell is only
                // flagged for refinement, then all future children
                // will be active
                bool all_children_active = true;
                if (!cell->is_active())
                  for (unsigned int c = 0; c < cell->n_children(); ++c)
                    if (!cell->child(c)->is_active() ||
                        cell->child(c)->is_ghost() ||
                        cell->child(c)->is_artificial())
                      {
                        all_children_active = false;
                        break;
                      }

                if (all_children_active)
                  {
                    // count number of refined and unrefined neighbors
                    // of cell.  neighbors on lower levels are counted
                    // as unrefined since they can only get to the
                    // same level as this cell by the next refinement
                    // cycle
                    unsigned int unrefined_neighbors = 0, total_neighbors = 0;

                    // Keep track if this cell is at a periodic
                    // boundary or not.  TODO: We do not currently run
                    // the algorithm for inner islands at a periodic
                    // boundary (remains to be implemented), but we
                    // also don't want to consider them
                    // boundary_island cells as this can interfere
                    // with 2:1 refinement across periodic faces.
                    // Instead: just ignore those cells for this
                    // smoothing operation below.
                    bool at_periodic_boundary = false;

                    for (const unsigned int n :
                         GeometryInfo<dim>::face_indices())
                      {
                        const cell_iterator neighbor = cell->neighbor(n);
                        if (neighbor.state() == IteratorState::valid)
                          {
                            ++total_neighbors;

                            if (!face_will_be_refined_by_neighbor(cell, n))
                              ++unrefined_neighbors;
                          }
                        else if (cell->has_periodic_neighbor(n))
                          {
                            ++total_neighbors;
                            at_periodic_boundary = true;
                          }
                      }

                    // if all neighbors unrefined: mark this cell for
                    // coarsening or don't refine if marked for that
                    //
                    // also do the distinction between the two
                    // versions of the eliminate_refined_*_islands
                    // flag
                    //
                    // the last check is whether there are any
                    // neighbors at all. if not so, then we are (e.g.)
                    // on the coarsest grid with one cell, for which,
                    // of course, we do not remove the refine flag.
                    if ((unrefined_neighbors == total_neighbors) &&
                        ((!cell->at_boundary() &&
                          (smooth_grid & eliminate_refined_inner_islands)) ||
                         (cell->at_boundary() && !at_periodic_boundary &&
                          (smooth_grid &
                           eliminate_refined_boundary_islands))) &&
                        (total_neighbors != 0))
                      {
                        if (!cell->is_active())
                          for (unsigned int c = 0; c < cell->n_children(); ++c)
                            {
                              cell->child(c)->clear_refine_flag();
                              cell->child(c)->set_coarsen_flag();
                            }
                        else
                          cell->clear_refine_flag();
                      }
                  }
              }
        }

      //------------------------------------
      // STEP 3:
      //    limit the level difference of neighboring cells at each
      //    vertex.
      //
      //    in case of anisotropic refinement this does not make
      //    sense. as soon as one cell is anisotropically refined, an
      //    Assertion is thrown. therefore we can ignore this problem
      //    later on
      if (smooth_grid & limit_level_difference_at_vertices)
        {
          Assert(!anisotropic_refinement,
                 ExcMessage("In case of anisotropic refinement the "
                            "limit_level_difference_at_vertices flag for "
                            "mesh smoothing must not be set!"));

          // store highest level one of the cells adjacent to a vertex
          // belongs to
          std::vector<int> vertex_level(vertices.size(), 0);
          for (const auto &cell : active_cell_iterators())
            {
              if (cell->refine_flag_set())
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  vertex_level[cell->vertex_index(vertex)] =
                    std::max(vertex_level[cell->vertex_index(vertex)],
                             cell->level() + 1);
              else if (!cell->coarsen_flag_set())
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  vertex_level[cell->vertex_index(vertex)] =
                    std::max(vertex_level[cell->vertex_index(vertex)],
                             cell->level());
              else
                {
                  // if coarsen flag is set then tentatively assume
                  // that the cell will be coarsened. this isn't
                  // always true (the coarsen flag could be removed
                  // again) and so we may make an error here
                  Assert(cell->coarsen_flag_set(), ExcInternalError());
                  for (const unsigned int vertex :
                       GeometryInfo<dim>::vertex_indices())
                    vertex_level[cell->vertex_index(vertex)] =
                      std::max(vertex_level[cell->vertex_index(vertex)],
                               cell->level() - 1);
                }
            }


          // loop over all cells in reverse order. do so because we
          // can then update the vertex levels on the adjacent
          // vertices and maybe already flag additional cells in this
          // loop
          //
          // note that not only may we have to add additional
          // refinement flags, but we will also have to remove
          // coarsening flags on cells adjacent to vertices that will
          // see refinement
          for (active_cell_iterator cell = last_active(); cell != end(); --cell)
            if (cell->refine_flag_set() == false)
              {
                for (const unsigned int vertex :
                     GeometryInfo<dim>::vertex_indices())
                  if (vertex_level[cell->vertex_index(vertex)] >=
                      cell->level() + 1)
                    {
                      // remove coarsen flag...
                      cell->clear_coarsen_flag();

                      // ...and if necessary also refine the current
                      // cell, at the same time updating the level
                      // information about vertices
                      if (vertex_level[cell->vertex_index(vertex)] >
                          cell->level() + 1)
                        {
                          cell->set_refine_flag();

                          for (const unsigned int v :
                               GeometryInfo<dim>::vertex_indices())
                            vertex_level[cell->vertex_index(v)] =
                              std::max(vertex_level[cell->vertex_index(v)],
                                       cell->level() + 1);
                        }

                      // continue and see whether we may, for example,
                      // go into the inner'if'
                      // above based on a
                      // different vertex
                    }
              }
        }

      //-----------------------------------
      // STEP 4:
      //    eliminate unrefined islands. this has higher priority
      //    since this diminishes the approximation properties not
      //    only of the unrefined island, but also of the surrounding
      //    patch.
      //
      //    do the loop from finest to coarsest cells since we may
      //    trigger a cascade by marking cells for refinement which
      //    may trigger more cells further down below
      if (smooth_grid & eliminate_unrefined_islands)
        {
          for (active_cell_iterator cell = last_active(); cell != end(); --cell)
            // only do something if cell is not already flagged for
            // (isotropic) refinement
            if (cell->refine_flag_set() !=
                RefinementCase<dim>::isotropic_refinement)
              possibly_refine_unrefined_island<dim, spacedim>(
                cell, (smooth_grid & allow_anisotropic_smoothing) != 0);
        }

      //-------------------------------
      // STEP 5:
      //    ensure patch level 1.
      //
      //    Introduce some terminology:
      //    - a cell that is refined
      //      once is a patch of
      //      level 1 simply called patch.
      //    - a cell that is globally
      //      refined twice is called
      //      a patch of level 2.
      //    - patch level n says that
      //      the triangulation consists
      //      of patches of level n.
      //      This makes sense only
      //      if the grid is already at
      //      least n times globally
      //      refined.
      //
      //    E.g. from patch level 1 follows: if at least one of the
      //    children of a cell is or will be refined than enforce all
      //    children to be refined.

      //    This step 4 only sets refinement flags and does not set
      //    coarsening flags.
      if (smooth_grid & patch_level_1)
        {
          // An important assumption (A) is that before calling this
          // function the grid was already of patch level 1.

          // loop over all cells whose children are all active.  (By
          // assumption (A) either all or none of the children are
          // active).  If the refine flag of at least one of the
          // children is set then set_refine_flag and
          // clear_coarsen_flag of all children.
          for (const auto &cell : cell_iterators())
            if (!cell->is_active())
              {
                // ensure the invariant. we can then check whether all
                // of its children are further refined or not by
                // simply looking at the first child
                Assert(cell_is_patch_level_1(cell), ExcInternalError());
                if (cell->child(0)->has_children() == true)
                  continue;

                // cell is found to be a patch.  combine the refine
                // cases of all children
                RefinementCase<dim> combined_ref_case =
                  RefinementCase<dim>::no_refinement;
                for (unsigned int i = 0; i < cell->n_children(); ++i)
                  combined_ref_case =
                    combined_ref_case | cell->child(i)->refine_flag_set();
                if (combined_ref_case != RefinementCase<dim>::no_refinement)
                  for (unsigned int i = 0; i < cell->n_children(); ++i)
                    {
                      cell_iterator child = cell->child(i);

                      child->clear_coarsen_flag();
                      child->set_refine_flag(combined_ref_case);
                    }
              }

          // The code above dealt with the case where we may get a
          // non-patch_level_1 mesh from refinement. Now also deal
          // with the case where we could get such a mesh by
          // coarsening.  Coarsen the children (and remove the
          // grandchildren) only if all cell->grandchild(i)
          // ->coarsen_flag_set() are set.
          //
          // for a case where this is a bit tricky, take a look at the
          // mesh_smoothing_0[12] testcases
          for (const auto &cell : cell_iterators())
            {
              // check if this cell has active grandchildren. note
              // that we know that it is patch_level_1, i.e. if one of
              // its children is active then so are all, and it isn't
              // going to have any grandchildren at all:
              if (cell->is_active() || cell->child(0)->is_active())
                continue;

              // cell is not active, and so are none of its
              // children. check the grandchildren. note that the
              // children are also patch_level_1, and so we only ever
              // need to check their first child
              const unsigned int n_children               = cell->n_children();
              bool               has_active_grandchildren = false;

              for (unsigned int i = 0; i < n_children; ++i)
                if (cell->child(i)->child(0)->is_active())
                  {
                    has_active_grandchildren = true;
                    break;
                  }

              if (has_active_grandchildren == false)
                continue;


              // ok, there are active grandchildren. see if either all
              // or none of them are flagged for coarsening
              unsigned int n_grandchildren = 0;

              // count all coarsen flags of the grandchildren.
              unsigned int n_coarsen_flags = 0;

              // cell is not a patch (of level 1) as it has a
              // grandchild.  Is cell a patch of level 2??  Therefore:
              // find out whether all cell->child(i) are patches
              for (unsigned int c = 0; c < n_children; ++c)
                {
                  // get at the child. by assumption (A), and the
                  // check by which we got here, the child is not
                  // active
                  cell_iterator child = cell->child(c);

                  const unsigned int nn_children = child->n_children();
                  n_grandchildren += nn_children;

                  // if child is found to be a patch of active cells
                  // itself, then add up how many of its children are
                  // supposed to be coarsened
                  if (child->child(0)->is_active())
                    for (unsigned int cc = 0; cc < nn_children; ++cc)
                      if (child->child(cc)->coarsen_flag_set())
                        ++n_coarsen_flags;
                }

              // if not all grandchildren are supposed to be coarsened
              // (e.g. because some simply don't have the flag set, or
              // because they are not active and therefore cannot
              // carry the flag), then remove the coarsen flag from
              // all of the active grandchildren. note that there may
              // be coarsen flags on the grandgrandchildren -- we
              // don't clear them here, but we'll get to them in later
              // iterations if necessary
              //
              // there is nothing we have to do if no coarsen flags
              // have been set at all
              if ((n_coarsen_flags != n_grandchildren) && (n_coarsen_flags > 0))
                for (unsigned int c = 0; c < n_children; ++c)
                  {
                    const cell_iterator child = cell->child(c);
                    if (child->child(0)->is_active())
                      for (unsigned int cc = 0; cc < child->n_children(); ++cc)
                        child->child(cc)->clear_coarsen_flag();
                  }
            }
        }

      //--------------------------------
      //
      //  at the boundary we could end up with cells with negative
      //  volume or at least with a part, that is negative, if the
      //  cell is refined anisotropically. we have to check, whether
      //  that can happen
      this->policy->prevent_distorted_boundary_cells(*this);

      //-------------------------------
      // STEP 6:
      //    take care of the requirement that no
      //    double refinement is done at each face
      //
      //    in case of anisotropic refinement it is only likely, but
      //    not sure, that the cells, which are more refined along a
      //    certain face common to two cells are on a higher
      //    level. therefore we cannot be sure, that the requirement
      //    of no double refinement is fulfilled after a single pass
      //    of the following actions. We could just wait for the next
      //    global loop. when this function terminates, the
      //    requirement will be fulfilled. However, it might be faster
      //    to insert an inner loop here.
      bool changed = true;
      while (changed)
        {
          changed                   = false;
          active_cell_iterator cell = last_active(), endc = end();

          for (; cell != endc; --cell)
            if (cell->refine_flag_set())
              {
                // loop over neighbors of cell
                for (const auto i : cell->face_indices())
                  {
                    // only do something if the face is not at the
                    // boundary and if the face will be refined with
                    // the RefineCase currently flagged for
                    const bool has_periodic_neighbor =
                      cell->has_periodic_neighbor(i);
                    const bool has_neighbor_or_periodic_neighbor =
                      !cell->at_boundary(i) || has_periodic_neighbor;
                    if (has_neighbor_or_periodic_neighbor &&
                        GeometryInfo<dim>::face_refinement_case(
                          cell->refine_flag_set(), i) !=
                          RefinementCase<dim - 1>::no_refinement)
                      {
                        // 1) if the neighbor has children: nothing to
                        // worry about.  2) if the neighbor is active
                        // and a coarser one, ensure, that its
                        // refine_flag is set 3) if the neighbor is
                        // active and as refined along the face as our
                        // current cell, make sure, that no
                        // coarsen_flag is set. if we remove the
                        // coarsen flag of our neighbor,
                        // fix_coarsen_flags() makes sure, that the
                        // mother cell will not be coarsened
                        if (cell->neighbor_or_periodic_neighbor(i)->is_active())
                          {
                            if ((!has_periodic_neighbor &&
                                 cell->neighbor_is_coarser(i)) ||
                                (has_periodic_neighbor &&
                                 cell->periodic_neighbor_is_coarser(i)))
                              {
                                if (cell->neighbor_or_periodic_neighbor(i)
                                      ->coarsen_flag_set())
                                  cell->neighbor_or_periodic_neighbor(i)
                                    ->clear_coarsen_flag();
                                // we'll set the refine flag for this
                                // neighbor below. we note, that we
                                // have changed something by setting
                                // the changed flag to true. We do not
                                // need to do so, if we just removed
                                // the coarsen flag, as the changed
                                // flag only indicates the need to
                                // re-run the inner loop. however, we
                                // only loop over cells flagged for
                                // refinement here, so nothing to
                                // worry about if we remove coarsen
                                // flags

                                if (dim == 2)
                                  {
                                    if (smooth_grid &
                                        allow_anisotropic_smoothing)
                                      changed =
                                        has_periodic_neighbor ?
                                          cell->periodic_neighbor(i)
                                            ->flag_for_face_refinement(
                                              cell
                                                ->periodic_neighbor_of_coarser_periodic_neighbor(
                                                  i)
                                                .first,
                                              RefinementCase<dim - 1>::cut_x) :
                                          cell->neighbor(i)
                                            ->flag_for_face_refinement(
                                              cell
                                                ->neighbor_of_coarser_neighbor(
                                                  i)
                                                .first,
                                              RefinementCase<dim - 1>::cut_x);
                                    else
                                      {
                                        if (!cell
                                               ->neighbor_or_periodic_neighbor(
                                                 i)
                                               ->refine_flag_set())
                                          changed = true;
                                        cell->neighbor_or_periodic_neighbor(i)
                                          ->set_refine_flag();
                                      }
                                  }
                                else // i.e. if (dim==3)
                                  {
                                    // ugly situations might arise here,
                                    // consider the following situation, which
                                    // shows neighboring cells at the common
                                    // face, where the upper right element is
                                    // coarser at the given face. Now the upper
                                    // child element of the lower left wants to
                                    // refine according to cut_z, such that
                                    // there is a 'horizontal' refinement of the
                                    // face marked with #####
                                    //
                                    //                            / /
                                    //                           / /
                                    //                          *---------------*
                                    //                          |               |
                                    //                          |               |
                                    //                          |               |
                                    //                          |               |
                                    //                          |               |
                                    //                          |               | /
                                    //                          |               |/
                                    //                          *---------------*
                                    //
                                    //
                                    //     *---------------*
                                    //    /|              /|
                                    //   / |     #####   / |
                                    //     |               |
                                    //     *---------------*
                                    //    /|              /|
                                    //   / |             / |
                                    //     |               |
                                    //     *---------------*
                                    //    /               /
                                    //   /               /
                                    //
                                    // this introduces too many hanging nodes
                                    // and the neighboring (coarser) cell (upper
                                    // right) has to be refined. If it is only
                                    // refined according to cut_z, then
                                    // everything is ok:
                                    //
                                    //                            / /
                                    //                           / /
                                    //                          *---------------*
                                    //                          |               |
                                    //                          |               | /
                                    //                          |               |/
                                    //                          *---------------*
                                    //                          |               |
                                    //                          |               | /
                                    //                          |               |/
                                    //                          *---------------*
                                    //
                                    //
                                    //     *---------------*
                                    //    /|              /|
                                    //   / *---------------*
                                    //    /|              /|
                                    //     *---------------*
                                    //    /|              /|
                                    //   / |             / |
                                    //     |               |
                                    //     *---------------*
                                    //    /               /
                                    //   /               /
                                    //
                                    // if however the cell wants to refine
                                    // itself in an other way, or if we disallow
                                    // anisotropic smoothing, then simply
                                    // refining the neighbor isotropically is
                                    // not going to work, since this introduces
                                    // a refinement of face ##### with both
                                    // cut_x and cut_y, which is not possible:
                                    //
                                    //                            /       / /
                                    //                           /       / /
                                    //                          *-------*-------*
                                    //                          |       |       |
                                    //                          |       |       | /
                                    //                          |       |       |/
                                    //                          *-------*-------*
                                    //                          |       |       |
                                    //                          |       |       | /
                                    //                          |       |       |/
                                    //                          *-------*-------*
                                    //
                                    //
                                    //     *---------------*
                                    //    /|              /|
                                    //   / *---------------*
                                    //    /|              /|
                                    //     *---------------*
                                    //    /|              /|
                                    //   / |             / |
                                    //     |               |
                                    //     *---------------*
                                    //    /               /
                                    //   /               /
                                    //
                                    // thus, in this case we also need to refine
                                    // our current cell in the new direction:
                                    //
                                    //                            /       / /
                                    //                           /       / /
                                    //                          *-------*-------*
                                    //                          |       |       |
                                    //                          |       |       | /
                                    //                          |       |       |/
                                    //                          *-------*-------*
                                    //                          |       |       |
                                    //                          |       |       | /
                                    //                          |       |       |/
                                    //                          *-------*-------*
                                    //
                                    //
                                    //     *-------*-------*
                                    //    /|      /|      /|
                                    //   / *-------*-------*
                                    //    /|      /|      /|
                                    //     *-------*-------*
                                    //    /|      /       /|
                                    //   / |             / |
                                    //     |               |
                                    //     *---------------*
                                    //    /               /
                                    //   /               /

                                    std::pair<unsigned int, unsigned int>
                                      nb_indices =
                                        has_periodic_neighbor ?
                                          cell
                                            ->periodic_neighbor_of_coarser_periodic_neighbor(
                                              i) :
                                          cell->neighbor_of_coarser_neighbor(i);
                                    unsigned int refined_along_x       = 0,
                                                 refined_along_y       = 0,
                                                 to_be_refined_along_x = 0,
                                                 to_be_refined_along_y = 0;

                                    const int this_face_index =
                                      cell->face_index(i);

                                    // step 1: detect, along which axis the face
                                    // is currently refined

                                    // first, we need an iterator pointing to
                                    // the parent face. This requires a slight
                                    // detour in case the neighbor is behind a
                                    // periodic face.
                                    const auto parent_face = [&]() {
                                      if (has_periodic_neighbor)
                                        {
                                          const auto neighbor =
                                            cell->periodic_neighbor(i);
                                          const auto parent_face_no =
                                            neighbor
                                              ->periodic_neighbor_of_periodic_neighbor(
                                                nb_indices.first);
                                          auto parent =
                                            neighbor->periodic_neighbor(
                                              nb_indices.first);
                                          return parent->face(parent_face_no);
                                        }
                                      else
                                        return cell->neighbor(i)->face(
                                          nb_indices.first);
                                    }();

                                    if ((this_face_index ==
                                         parent_face->child_index(0)) ||
                                        (this_face_index ==
                                         parent_face->child_index(1)))
                                      {
                                        // this might be an
                                        // anisotropic child. get the
                                        // face refine case of the
                                        // neighbors face and count
                                        // refinements in x and y
                                        // direction.
                                        RefinementCase<dim - 1> frc =
                                          parent_face->refinement_case();
                                        if (frc & RefinementCase<dim>::cut_x)
                                          ++refined_along_x;
                                        if (frc & RefinementCase<dim>::cut_y)
                                          ++refined_along_y;
                                      }
                                    else
                                      // this has to be an isotropic
                                      // child
                                      {
                                        ++refined_along_x;
                                        ++refined_along_y;
                                      }
                                    // step 2: detect, along which axis the face
                                    // has to be refined given the current
                                    // refine flag
                                    RefinementCase<dim - 1> flagged_frc =
                                      GeometryInfo<dim>::face_refinement_case(
                                        cell->refine_flag_set(),
                                        i,
                                        cell->face_orientation(i),
                                        cell->face_flip(i),
                                        cell->face_rotation(i));
                                    if (flagged_frc &
                                        RefinementCase<dim>::cut_x)
                                      ++to_be_refined_along_x;
                                    if (flagged_frc &
                                        RefinementCase<dim>::cut_y)
                                      ++to_be_refined_along_y;

                                    // step 3: set the refine flag of the
                                    // (coarser and active) neighbor.
                                    if ((smooth_grid &
                                         allow_anisotropic_smoothing) ||
                                        cell->neighbor_or_periodic_neighbor(i)
                                          ->refine_flag_set())
                                      {
                                        if (refined_along_x +
                                              to_be_refined_along_x >
                                            1)
                                          changed |=
                                            cell
                                              ->neighbor_or_periodic_neighbor(i)
                                              ->flag_for_face_refinement(
                                                nb_indices.first,
                                                RefinementCase<dim -
                                                               1>::cut_axis(0));
                                        if (refined_along_y +
                                              to_be_refined_along_y >
                                            1)
                                          changed |=
                                            cell
                                              ->neighbor_or_periodic_neighbor(i)
                                              ->flag_for_face_refinement(
                                                nb_indices.first,
                                                RefinementCase<dim -
                                                               1>::cut_axis(1));
                                      }
                                    else
                                      {
                                        if (cell
                                              ->neighbor_or_periodic_neighbor(i)
                                              ->refine_flag_set() !=
                                            RefinementCase<
                                              dim>::isotropic_refinement)
                                          changed = true;
                                        cell->neighbor_or_periodic_neighbor(i)
                                          ->set_refine_flag();
                                      }

                                    // step 4: if necessary (see above) add to
                                    // the refine flag of the current cell
                                    cell_iterator nb =
                                      cell->neighbor_or_periodic_neighbor(i);
                                    RefinementCase<dim - 1> nb_frc =
                                      GeometryInfo<dim>::face_refinement_case(
                                        nb->refine_flag_set(),
                                        nb_indices.first,
                                        nb->face_orientation(nb_indices.first),
                                        nb->face_flip(nb_indices.first),
                                        nb->face_rotation(nb_indices.first));
                                    if ((nb_frc & RefinementCase<dim>::cut_x) &&
                                        !((refined_along_x != 0u) ||
                                          (to_be_refined_along_x != 0u)))
                                      changed |= cell->flag_for_face_refinement(
                                        i,
                                        RefinementCase<dim - 1>::cut_axis(0));
                                    if ((nb_frc & RefinementCase<dim>::cut_y) &&
                                        !((refined_along_y != 0u) ||
                                          (to_be_refined_along_y != 0u)))
                                      changed |= cell->flag_for_face_refinement(
                                        i,
                                        RefinementCase<dim - 1>::cut_axis(1));
                                  }
                              }  // if neighbor is coarser
                            else // -> now the neighbor is not coarser
                              {
                                cell->neighbor_or_periodic_neighbor(i)
                                  ->clear_coarsen_flag();
                                const unsigned int nb_nb =
                                  has_periodic_neighbor ?
                                    cell
                                      ->periodic_neighbor_of_periodic_neighbor(
                                        i) :
                                    cell->neighbor_of_neighbor(i);
                                const cell_iterator neighbor =
                                  cell->neighbor_or_periodic_neighbor(i);
                                RefinementCase<dim - 1> face_ref_case =
                                  GeometryInfo<dim>::face_refinement_case(
                                    neighbor->refine_flag_set(),
                                    nb_nb,
                                    neighbor->face_orientation(nb_nb),
                                    neighbor->face_flip(nb_nb),
                                    neighbor->face_rotation(nb_nb));
                                RefinementCase<dim - 1> needed_face_ref_case =
                                  GeometryInfo<dim>::face_refinement_case(
                                    cell->refine_flag_set(),
                                    i,
                                    cell->face_orientation(i),
                                    cell->face_flip(i),
                                    cell->face_rotation(i));
                                // if the neighbor wants to refine the
                                // face with cut_x and we want cut_y
                                // or vice versa, we have to refine
                                // isotropically at the given face
                                if ((face_ref_case ==
                                       RefinementCase<dim>::cut_x &&
                                     needed_face_ref_case ==
                                       RefinementCase<dim>::cut_y) ||
                                    (face_ref_case ==
                                       RefinementCase<dim>::cut_y &&
                                     needed_face_ref_case ==
                                       RefinementCase<dim>::cut_x))
                                  {
                                    changed = cell->flag_for_face_refinement(
                                      i, face_ref_case);
                                    neighbor->flag_for_face_refinement(
                                      nb_nb, needed_face_ref_case);
                                  }
                              }
                          }
                        else //-> the neighbor is not active
                          {
                            RefinementCase<dim - 1>
                              face_ref_case = cell->face(i)->refinement_case(),
                              needed_face_ref_case =
                                GeometryInfo<dim>::face_refinement_case(
                                  cell->refine_flag_set(),
                                  i,
                                  cell->face_orientation(i),
                                  cell->face_flip(i),
                                  cell->face_rotation(i));
                            // if the face is refined with cut_x and
                            // we want cut_y or vice versa, we have to
                            // refine isotropically at the given face
                            if ((face_ref_case == RefinementCase<dim>::cut_x &&
                                 needed_face_ref_case ==
                                   RefinementCase<dim>::cut_y) ||
                                (face_ref_case == RefinementCase<dim>::cut_y &&
                                 needed_face_ref_case ==
                                   RefinementCase<dim>::cut_x))
                              changed =
                                cell->flag_for_face_refinement(i,
                                                               face_ref_case);
                          }
                      }
                  }
              }
        }

      //------------------------------------
      // STEP 7:
      //    take care that no double refinement
      //    is done at each line in 3d or higher
      //    dimensions.
      this->policy->prepare_refinement_dim_dependent(*this);

      //------------------------------------
      // STEP 8:
      //    make sure that all children of each
      //    cell are either flagged for coarsening
      //    or none of the children is
      fix_coarsen_flags();
      // get the refinement and coarsening
      // flags
      std::vector<bool> flags_after_loop[2];
      save_coarsen_flags(flags_after_loop[0]);
      save_refine_flags(flags_after_loop[1]);

      // find out whether something was
      // changed in this loop
      mesh_changed_in_this_loop =
        ((flags_before_loop[0] != flags_after_loop[0]) ||
         (flags_before_loop[1] != flags_after_loop[1]));

      // set the flags for the next loop
      // already
      flags_before_loop[0].swap(flags_after_loop[0]);
      flags_before_loop[1].swap(flags_after_loop[1]);
    }
  while (mesh_changed_in_this_loop);


  // find out whether something was really changed in this
  // function. Note that @p{flags_before_loop} represents the state
  // after the last loop, i.e.  the present state
  return ((flags_before[0] != flags_before_loop[0]) ||
          (flags_before[1] != flags_before_loop[1]));
}



template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::write_bool_vector(
  const unsigned int       magic_number1,
  const std::vector<bool> &v,
  const unsigned int       magic_number2,
  std::ostream &           out)
{
  const unsigned int N     = v.size();
  unsigned char *    flags = new unsigned char[N / 8 + 1];
  for (unsigned int i = 0; i < N / 8 + 1; ++i)
    flags[i] = 0;

  for (unsigned int position = 0; position < N; ++position)
    flags[position / 8] |= (v[position] ? (1 << (position % 8)) : 0);

  AssertThrow(out, ExcIO());

  // format:
  // 0. magic number
  // 1. number of flags
  // 2. the flags
  // 3. magic number
  out << magic_number1 << ' ' << N << std::endl;
  for (unsigned int i = 0; i < N / 8 + 1; ++i)
    out << static_cast<unsigned int>(flags[i]) << ' ';

  out << std::endl << magic_number2 << std::endl;

  delete[] flags;

  AssertThrow(out, ExcIO());
}


template <int dim, int spacedim>
void
Triangulation<dim, spacedim>::read_bool_vector(const unsigned int magic_number1,
                                               std::vector<bool> &v,
                                               const unsigned int magic_number2,
                                               std::istream &     in)
{
  AssertThrow(in, ExcIO());

  unsigned int magic_number;
  in >> magic_number;
  AssertThrow(magic_number == magic_number1, ExcGridReadError());

  unsigned int N;
  in >> N;
  v.resize(N);

  unsigned char *    flags = new unsigned char[N / 8 + 1];
  unsigned short int tmp;
  for (unsigned int i = 0; i < N / 8 + 1; ++i)
    {
      in >> tmp;
      flags[i] = tmp;
    }

  for (unsigned int position = 0; position != N; ++position)
    v[position] = ((flags[position / 8] & (1 << (position % 8))) != 0);

  in >> magic_number;
  AssertThrow(magic_number == magic_number2, ExcGridReadError());

  delete[] flags;

  AssertThrow(in, ExcIO());
}



template <int dim, int spacedim>
std::size_t
Triangulation<dim, spacedim>::memory_consumption() const
{
  std::size_t mem = 0;
  mem += MemoryConsumption::memory_consumption(levels);
  for (const auto &level : levels)
    mem += MemoryConsumption::memory_consumption(*level);
  mem += MemoryConsumption::memory_consumption(vertices);
  mem += MemoryConsumption::memory_consumption(vertices_used);
  mem += sizeof(manifolds);
  mem += sizeof(smooth_grid);
  mem += MemoryConsumption::memory_consumption(number_cache);
  mem += sizeof(faces);
  if (faces)
    mem += MemoryConsumption::memory_consumption(*faces);

  return mem;
}



template <int dim, int spacedim>
Triangulation<dim, spacedim>::DistortedCellList::~DistortedCellList() noexcept =
  default;


// explicit instantiations
#include "tria.inst"

DEAL_II_NAMESPACE_CLOSE
