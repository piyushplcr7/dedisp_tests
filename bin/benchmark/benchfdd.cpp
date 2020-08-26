#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  benchParameter.dm_start = 2;
  benchParameter.dm_end = 100;
  benchParameter.nchans = 1024;
  benchParameter.Tobs = 30.0;
  benchParameter.verbose = false;

  return run<dedisp::FDDGPUPlan>(benchParameter);
}