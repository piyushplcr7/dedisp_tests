/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* This Stopwatch class is copied from the IDG repository
* (https://git.astron.nl/RD/idg)
*/
#include "Stopwatch.h"

#include <string>
#include <chrono>

class StopwatchImpl : public virtual Stopwatch
{

    public:
        StopwatchImpl();

        ~StopwatchImpl();

        virtual void Start() final override;
		virtual void Pause() final override;
		virtual void Reset() final override;
		virtual void Add(double ms) final override;

		virtual std::string ToString() const final override;
		virtual long double Milliseconds() const final override;
		virtual long double Seconds() const final override;
		virtual unsigned int Count() const final override;

    private:
		std::string ToString(const std::chrono::duration<double>& duration) const;

		bool m_running;
		std::chrono::system_clock::time_point m_time_start;
		std::chrono::duration<double> m_time_sum;
		unsigned int m_count = 0;
};
