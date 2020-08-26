#include <fdd/FDDCPUPlan.hpp>
#include <fdd/FDDGPUPlan.hpp>

#include "test.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  return run<dedisp::FDDGPUPlan>();
}