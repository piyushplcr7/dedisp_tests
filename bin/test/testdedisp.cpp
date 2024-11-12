// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <dedisp/DedispPlan.hpp>
#include "fdd/FDDGPUPlan.hpp"
#include "fdd/FDDCPUPlan.hpp"
#include "tdd/TDDPlan.hpp"

//#include "testfits.hpp"
#include "testtwochansin.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  //return run<dedisp::DedispPlan>(); // uses run method from test.hpp
  //return run<dedisp::TDDPlan>(); // uses run method from test.hpp
  //return run<dedisp::FDDCPUPlan>(); // uses run method from test.hpp
  return run<dedisp::FDDGPUPlan>(); // uses run method from test.hpp
}