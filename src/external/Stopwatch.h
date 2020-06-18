#ifndef STOPWATCH_H
#define STOPWATCH_H

#include <string>

class Stopwatch
{
	public:
        static Stopwatch* create();

        virtual ~Stopwatch() {};

		virtual void Start() = 0;
		virtual void Pause() = 0;
		virtual void Reset() = 0;

		static std::string ToString(int64_t ms);
		virtual std::string ToString() const = 0;
		virtual long double Milliseconds() const = 0;
		virtual long double Seconds() const = 0;
		virtual unsigned int Count() const = 0;

    protected:
        Stopwatch() {}
};

#endif
