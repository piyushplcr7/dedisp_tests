/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* This Stopwatch class is copied from the IDG repository
* (https://git.astron.nl/RD/idg/-/commit/9bed1e356bcfde0e96b1a4d21f143a1eae4dbfcb)
* and modified to work without the data.h header file.
* The ToString method now only reports time in seconds and milliseconds.
* Original:
* SPDX-License-Identifier: GPL-3.0
* SPDX-FileCopyrightText: 2020 ASTRON (Netherlands Institute for Radio Astronomy)
*/
#include "StopwatchImpl.h"

#include <cmath>
#include <ctime>
#include <sstream>

// Create a new Stopwatch
Stopwatch* Stopwatch::create()
{
    return new StopwatchImpl();
}

StopwatchImpl::StopwatchImpl() :
    m_running(false),
    m_time_sum(std::chrono::duration<double>::zero()),
    m_count(0)
{
}

StopwatchImpl::~StopwatchImpl()
{
}

// Start Stopwatch
void StopwatchImpl::Start()
{
    if (!m_running)
    {
        m_time_start = std::chrono::high_resolution_clock::now();
        m_running = true;
    }
}

// Pause Stopwatch
void StopwatchImpl::Pause()
{
    if (m_running)
    {
        auto time_now = std::chrono::high_resolution_clock::now();
        m_time_sum += time_now - m_time_start;
        m_running = false;
        m_count++;
    }
}

// Reset the Stopwatch
void StopwatchImpl::Reset()
{
    m_running = false;
    m_time_sum = std::chrono::duration<double>::zero();
}

// Add ms [double] to this Stopwatch
void StopwatchImpl::Add(
    double ms)
{
    auto microseconds = (int64_t) (ms * 1e3);
    m_time_sum += std::chrono::microseconds(microseconds);
}

// Return string, convert from input ms to output seconds
std::string Stopwatch::ToString(
    int64_t ms)
{
    auto seconds = ms * 1e-3;

    std::stringstream output;
    output << std::fixed;
    output << seconds;

    return output.str();
}

// Return string, convert from duration in ms to string in ms
std::string StopwatchImpl::ToString(
    const std::chrono::duration<double>& duration) const
{
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return Stopwatch::ToString(ms);
}

// Return Stopwatch sum as string
std::string StopwatchImpl::ToString() const
{
    if (m_running)
    {
        auto time_now = std::chrono::high_resolution_clock::now();
        auto time_current = m_time_sum + (time_now - m_time_start);
        return ToString(m_time_sum + time_current);
    } else {
        return ToString(m_time_sum);
    }
}

// Return Stopwatch sum as milliseconds
long double StopwatchImpl::Milliseconds() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_time_sum).count();
}

// Return Stopwatch sum as seconds
long double StopwatchImpl::Seconds() const
{
    return std::chrono::duration_cast<std::chrono::seconds>(m_time_sum).count();
}

// Return Stopwatch count [unsigned int]
unsigned int StopwatchImpl::Count() const
{
    return m_count;
}