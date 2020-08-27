#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  // Default
  BenchParameters benchParameter;
  benchParameter.dm_start = 2;
  benchParameter.dm_end = 100;
  benchParameter.nchans = 1024;
  benchParameter.Tobs = 30.0;
  benchParameter.verbose = true;

  // Get parameters from argv

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