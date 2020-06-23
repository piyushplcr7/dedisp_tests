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
    int64_t ms)
{
    m_time_sum += std::chrono::milliseconds(ms);
}

std::string Stopwatch::ToString(
    int64_t ms)
{
    std::chrono::milliseconds d(ms);
    auto s_(std::chrono::duration_cast<std::chrono::seconds>(d));
    auto sub_s_(std::chrono::duration_cast<std::chrono::milliseconds>(d - s_));

    std::stringstream output;
    output << std::scientific;
    output << s_.count() << ".";
    output << std::fixed;
    output.fill('0');
    output.flags(std::ios::dec | std::ios::right);
    output.width(3);
    output << sub_s_.count();

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