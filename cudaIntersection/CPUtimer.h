#ifndef __CPUTIMER__  
#define __CPUTIMER__

#include <windows.h>
#include <iostream>

using std::cout;



class CPUTimer
{
	public:
		double PCFreq;
		__int64 CounterStart;


		void StartCounter()
		{

			CounterStart = 0;
			PCFreq = 0.0;

			LARGE_INTEGER li;
			if(!QueryPerformanceFrequency(&li))
			cout << "QueryPerformanceFrequency failed!\n";

			PCFreq = double(li.QuadPart)/1000.0;

			QueryPerformanceCounter(&li);
			CounterStart = li.QuadPart;
		}
		double GetCounter()
		{
			LARGE_INTEGER li;
			QueryPerformanceCounter(&li);
			return double(li.QuadPart-CounterStart)/PCFreq;
	}
};

#endif