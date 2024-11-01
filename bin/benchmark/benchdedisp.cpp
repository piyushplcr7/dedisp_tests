// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <dedisp/DedispPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  // optionally overwrite benchParameters here

  // Get parameters from argv
  if (parseParameters(argc, argv, benchParameter)!=0) return -1;

  std::cout << "Benchmark dedisp on GPU" << std::endl;
  return run<dedisp::DedispPlan>(benchParameter);
}