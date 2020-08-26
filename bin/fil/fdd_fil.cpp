#include <fdd/FDDGPUPlan.hpp>

#include <iostream>

#include "fil.hpp"

int main(int argc,char *argv[])
{
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
  }}