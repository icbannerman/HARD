#pragma once

#include "../../types.hh"
#include "../utils.hh"

namespace hard::tasks::hydro {

template<std::size_t D>
void
apply_gravity(typename mesh<D>::template accessor<ro> m,
  field<double>::accessor<ro, na> mass_density_a,
  typename field<vec<D>>::template accessor<ro, na> velocity_a,
  typename field<vec<D>>::template accessor<rw, na> dt_momentum_density_a,
  field<double>::accessor<rw, na> dt_total_energy_density_a,
  single<vec<D>>::accessor<ro> gravity_a) {

  auto mass_density = m.template mdcolex<is::cells>(mass_density_a);
  auto velocity = m.template mdcolex<is::cells>(velocity_a);
  auto dt_momentum_density =
    m.template mdcolex<is::cells>(dt_momentum_density_a);
  auto dt_total_energy_density =
    m.template mdcolex<is::cells>(dt_total_energy_density_a);
  vec<D> g = *gravity_a;

  using hard::tasks::util::get_mdiota_policy;

  if constexpr(D == 1) {
    forall(i, (m.template cells<ax::x, dm::quantities>()), "gravity_1d") {
      dt_momentum_density(i) += mass_density(i) * g;
      dt_total_energy_density(i) += mass_density(i) * velocity(i).x * g.x;
    };
  }
  else if constexpr(D == 2) {
    auto mdpolicy_qq = get_mdiota_policy(mass_density,
      m.template cells<ax::y, dm::quantities>(),
      m.template cells<ax::x, dm::quantities>());
    forall(ji, mdpolicy_qq, "gravity_2d") {
      auto [j, i] = ji;
      dt_momentum_density(i, j) += mass_density(i, j) * g;
      double dot = velocity(i, j).x * g.x + velocity(i, j).y * g.y;
      dt_total_energy_density(i, j) += mass_density(i, j) * dot;
    };
  }
  else {
    auto mdpolicy_qqq = get_mdiota_policy(mass_density,
      m.template cells<ax::z, dm::quantities>(),
      m.template cells<ax::y, dm::quantities>(),
      m.template cells<ax::x, dm::quantities>());
    forall(kji, mdpolicy_qqq, "gravity_3d") {
      auto [k, j, i] = kji;
      dt_momentum_density(i, j, k) += mass_density(i, j, k) * g;
      double dot = velocity(i, j, k).x * g.x + velocity(i, j, k).y * g.y +
                   velocity(i, j, k).z * g.z;
      dt_total_energy_density(i, j, k) += mass_density(i, j, k) * dot;
    };
  }
}

} // namespace hard::tasks::hydro
