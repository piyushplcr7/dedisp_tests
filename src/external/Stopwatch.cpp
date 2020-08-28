#include "StopwatchImpl.h"

#include <cmath>
#include <ctime>
#include <sstream>

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

void StopwatchImpl::Start()
{
    if (!m_running)
    {
        m_time_start = std::chrono::high_resolution_clock::now();
        m_running = true;
    }
}

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

void StopwatchImpl::Reset()
{
    m_running = false;
    m_time_sum = std::chrono::duration<double>::zero();
}

void StopwatchImpl::Add(
    double ms)
{
    auto microseconds = (int64_t) (ms * 1e3);
    m_time_sum += std::chrono::microseconds(microseconds);
}

std::string Stopwatch::ToString(
    int64_t ms)
{
    auto seconds = ms * 1e-3;

    std::stringstream output;
    output << std::fixed;
    output << seconds;

    return output.str();
}

std::string StopwatchImpl::ToString(
    const std::chrono::duration<double>& duration) const
{
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(duration).count();
    return Stopwatch::ToString(ms);
}

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

long double StopwatchImpl::Milliseconds() const
{
    return std::chrono::duration_cast<std::chrono::milliseconds>(m_time_sum).count();
}

long double StopwatchImpl::Seconds() const
{
    return std::chrono::duration_cast<std::chrono::seconds>(m_time_sum).count();
}

unsigned int StopwatchImpl::Count() const
{
    return m_count;
}