// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  // optionally overwrite benchParameters here

  // Get parameters from argv
  if (parseParameters(argc, argv, benchParameter)!=0) return -1;

  char *use_cpu_str = getenv("USE_CPU");
  bool use_cpu = !use_cpu_str ? false : atoi(use_cpu_str);
  if (use_cpu)
  {
    std::cout << "Benchmark FDD on CPU" << std::endl;
    return run<dedisp::FDDCPUPlan>(benchParameter);
  } else
  {
    std::cout << "Benchmark FDD on GPU" << std::endl;
    return run<dedisp::FDDGPUPlan>(benchParameter);
  }

}