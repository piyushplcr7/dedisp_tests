// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
#include <dedisp/DedispPlan.hpp>

#include "test.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
  return run<dedisp::DedispPlan>(); // uses run method from test.hpp
}