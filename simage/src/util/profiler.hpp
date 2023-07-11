#pragma once

#ifdef _WIN32

#include <intrin.h>

#else

#include <x86intrin.h>

#endif


inline unsigned long long read_cpu_ticks()
{
    return __rdtsc();
}