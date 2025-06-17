#pragma once

#include "../../options.hh"
#include "../../types.hh"
#include "../utils.hh"
#include <cmath>
#include <cstddef>
#include <yaml-cpp/yaml.h>

namespace hard::tasks::initial_data {

/*----------------------------------------------------------------------------*
  Rayleigh-Taylor Instability.
 *----------------------------------------------------------------------------*/

template<std::size_t D>
auto
rayleigh_taylor(typename mesh<D>::template accessor<ro> m,
  field<double>::accessor<wo, na> mass_density_a,
  typename field<vec<D>>::template accessor<wo, na> momentum_density_a,
  field<double>::accessor<wo, na> total_energy_density_a,
  field<double>::accessor<wo, na> radiation_energy_density_a,
  const eos::eos_wrapper & eos) {

  auto mass_density = m.template mdcolex<is::cells>(mass_density_a);
  auto momentum_density = m.template mdcolex<is::cells>(momentum_density_a);
  auto total_energy_density =
    m.template mdcolex<is::cells>(total_energy_density_a);
  auto radiation_energy_density =
    m.template mdcolex<is::cells>(radiation_energy_density_a);

  YAML::Node config = YAML::LoadFile(opt::config.value());

  const double rL = config["problem_parameters"]["density_low"].as<double>();
  const double rH = config["problem_parameters"]["density_high"].as<double>();
  const double p0 = config["problem_parameters"]["pressure"].as<double>();
  const double amp = config["problem_parameters"]["perturbation"].as<double>();
  const double interface = config["problem_parameters"]["interface"].as<double>();

  const double wavenumber = 2.0 * M_PI;

  if constexpr(D == 2) {
    forall(j, (m.template cells<ax::y, dm::quantities>()), "init_rt_2d") {
      for(auto i : m.template cells<ax::x, dm::quantities>()) {
        const double x = m.template center<ax::x>(i);
        const double y = m.template center<ax::y>(j);

        const double interface_y = interface + amp * std::cos(wavenumber * x);
        const bool upper = y > interface_y;
        const double rho = upper ? rH : rL;

        mass_density(i, j) = rho;
        momentum_density(i, j) = vec<D>(0.0);
        if(std::abs(y - interface) < 0.05)
          momentum_density(i, j).y = rho * amp * std::sin(wavenumber * x);
        const double e = util::find_sie(eos, rho, p0);
        total_energy_density(i, j) = rho * e;
        radiation_energy_density(i, j) = 0.0;
      }
    };
  }
  else {
    flog_fatal("Rayleigh-Taylor problem is only implemented for D == 2");
  }
}

} // namespace hard::tasks::initial_data
