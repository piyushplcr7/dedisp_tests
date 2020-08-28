#include <tdd/TDDPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  // optionally overwrite benchParameters here

  // Get parameters from argv
  if (parseParameters(argc, argv, benchParameter)!=0) return -1;

  std::cout << "Benchmark TDD on GPU" << std::endl;
  return run<dedisp::TDDPlan>(benchParameter);
}