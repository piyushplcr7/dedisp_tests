// Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
// SPDX-License-Identifier: GPL-3.0-or-later
// CPU-only FDD test driver for the OpenMP backend.
#include <fdd/FDDCPUPlan.hpp>

#include <iostream>

#include "test.hpp"

template<typename PlanType>
int run();

int main(int argc, char* argv[])
{
    std::cout << "Test FDD on CPU (OpenMP backend)" << std::endl;
    return run<dedisp::FDDCPUPlan>(); // uses run() from test.hpp
}
