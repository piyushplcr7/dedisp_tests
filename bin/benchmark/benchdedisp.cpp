#include <tdd/TDDPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  return run<dedisp::TDDPlan>(benchParameter);
}