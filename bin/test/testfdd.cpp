// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include <iostream>

#include "test.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  // Set environment variable USE_CPU to switch to CPU implementation of FDD
  // Using GPU implementation by default
  char *use_cpu_str = getenv("USE_CPU");
  bool use_cpu = !use_cpu_str ? false : atoi(use_cpu_str);
  if (use_cpu)
  {
    std::cout << "Test FDD on CPU" << std::endl;
    return run<dedisp::FDDCPUPlan>(); // uses run method from test.hpp
  } else
  {
    std::cout << "Test FDD on GPU" << std::endl;
    return run<dedisp::FDDGPUPlan>(); // uses run method from test.hpp
  }
}