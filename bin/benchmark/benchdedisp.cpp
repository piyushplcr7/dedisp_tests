#include <tdd/TDDPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  benchParameter.dm_start = 2;
  benchParameter.dm_end = 1000;
  benchParameter.nchans = 1600;
  benchParameter.Tobs = 30.0;
  benchParameter.verbose = true;
  return run<dedisp::TDDPlan>(benchParameter);
}