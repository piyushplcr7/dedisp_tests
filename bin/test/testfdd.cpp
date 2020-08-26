#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include <iostream>

#include "test.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  char *use_cpu_str = getenv("USE_CPU");
  bool use_cpu = !use_cpu_str ? false : atoi(use_cpu_str);
  if (use_cpu)
  {
    std::cout << "Test FDD on CPU" << std::endl;
    return run<dedisp::FDDCPUPlan>();
  } else
  {
    std::cout << "Test FDD on GPU" << std::endl;
    return run<dedisp::FDDGPUPlan>();
  }
}