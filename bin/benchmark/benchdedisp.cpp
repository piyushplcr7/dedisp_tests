#include <dedisp/DedispPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  BenchParameters benchParameter;
  return run<dedisp::DedispPlan>(benchParameter);
}