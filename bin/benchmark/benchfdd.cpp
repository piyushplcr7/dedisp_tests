#include <fdd/FDDPlan.hpp>

#include "bench.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  return run<dedisp::FDDPlan>();
}