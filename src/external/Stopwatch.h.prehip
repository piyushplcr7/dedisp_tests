/*
* Copyright (C) 2021 ASTRON (Netherlands Institute for Radio Astronomy)
* SPDX-License-Identifier: GPL-3.0-or-later
* This Stopwatch class is copied from the IDG repository
* (https://git.astron.nl/RD/idg)
*/
#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <string>

class Stopwatch
{
	public:
        static Stopwatch* create();

        virtual ~Stopwatch() {};

		// Start Stopwatch
		virtual void Start() = 0;
		// Pause Stopwatch
		virtual void Pause() = 0;
		// Reset the Stopwatch
		virtual void Reset() = 0;
		// Add ms [double] to this Stopwatch
		virtual void Add(double ms) = 0;

		// Return string, convert from input ms to output seconds
		static std::string ToString(int64_t ms);
		// Return Stopwatch sum as string
		virtual std::string ToString() const = 0;
		// Return Stopwatch sum as milliseconds
		virtual long double Milliseconds() const = 0;
		// Return Stopwatch sum as seconds
		virtual long double Seconds() const = 0;
		// Return Stopwatch count [unsigned int]
		virtual unsigned int Count() const = 0;

    protected:
        Stopwatch() {}
};

#endif
