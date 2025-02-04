// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <fdd/FDDGPUPlan.hpp>

#include <iostream>

#include "fil.hpp"

int main(int argc,char *argv[])
{
  // Set environment variable USE_CPU to switch to CPU implementation of FDD
  // Using GPU implementation by default
  char *use_cpu_str = getenv("USE_CPU");
  bool use_cpu = !use_cpu_str ? false : atoi(use_cpu_str);
  if (use_cpu)
  {
    std::cout << "Run FDD on CPU" << std::endl;
    return run<dedisp::FDDCPUPlan>(argc, argv);
  } else
  {
    std::cout << "Run FDD on GPU" << std::endl;
    return run<dedisp::FDDGPUPlan>(argc, argv);
  }
}