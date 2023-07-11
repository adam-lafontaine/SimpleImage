#ifndef SIMAGE_NO_PROFILE

#include "profiler.hpp"

#include <cstdio>
#include <cstdarg>
#include <algorithm>


namespace
{
    class ProfileRecord
    {
    public:

        u64 cpu_total = 0;

        u32 hit_count = 0;

        b32 is_active = false;

        u64 cpu_avg() { return cpu_total == 0 ? 1 : cpu_total / hit_count; }
    };
}


constexpr auto N_PROFILE_RECORDS = (int)perf::ProfileLabel::Count;


static ProfileRecord g_records[N_PROFILE_RECORDS] = { 0 };


#ifdef _WIN32

#include <intrin.h>

#else

#include <x86intrin.h>

#endif


static u64 cpu_read_ticks()
{
    return __rdtsc();
}


namespace perf
{
    Profile::Profile(ProfileLabel label)
    {
        profile_id = (int)label;

        auto& record = g_records[profile_id];

        if (record.is_active)
        {
            // recursive
            profile_id = -1;
        }
        else
        {
            record.is_active = true;
            cpu_start = cpu_read_ticks();
        }
    }


    Profile::~Profile()
    {
        if (profile_id == -1)
        {
            return;
        }

        cpu_end = cpu_read_ticks();
        auto& record = g_records[profile_id];

        record.cpu_total += (cpu_end - cpu_start);
        ++record.hit_count;
        record.is_active = false;
    }
}




namespace perf
{
    using PL = perf::ProfileLabel;


    static void print(FILE* file, cstr format, ...)
    {
        va_list args;
        va_start(args, format);
        vfprintf(stdout, format, args);
        va_end(args);

        va_start(args, format);
        vfprintf(file, format, args);
        va_end(args);
    }


    static int count_digits(u64 n)
    {
        return (int)floor(log10(n) + 1);
    }


    void profile_init()
    {
        for (u32 i = 0; i < (u32)PL::Count; ++i)
        {
            g_records[i] = { 0 };
        }
    }


    void profile_clear()
    {
        profile_init();
    }


    void profile_report()
    {
        auto begin = g_records;
        auto end = g_records + N_PROFILE_RECORDS;
        auto const compare = [](auto lhs, auto rhs){ return lhs.cpu_avg() < rhs.cpu_avg(); };
        auto min = std::min_element(begin, end, compare);

        FILE* out = fopen("build_files/profile.txt", "a");

        print(out, "\nProfile Report:\n");

        int label_len = 10;
        int abs_len = 6;
        int rel_len = 1;
        for (u32 i = 0; i < (u32)PL::Count; ++i)
        {
            auto& record = g_records[i];

            auto len = strlen(to_cstr((PL)i));
            if (len > label_len)
            {
                label_len = (int)len;
            }

            len = count_digits(record.cpu_avg());
            if (len > abs_len)
            {
                abs_len = len;
            }

            len = count_digits(record.cpu_avg() / min->cpu_avg());
            if (len > rel_len)
            {
                rel_len = len;
            }
        }

        rel_len += 3;

        for (u32 i = 0; i < (u32)PL::Count; ++i)
        {
            auto& record = g_records[i];

            auto label = to_cstr((PL)i);
            auto cpu_abs = record.cpu_avg();
            auto cpu_rel = (f32)cpu_abs / min->cpu_avg();
            auto count = record.hit_count;

            auto format = "%*s: %*.2f (%*lu x %u) - %d\n";

            print(out, format, label_len, label, rel_len, cpu_rel, abs_len, cpu_abs, count, count_digits(cpu_abs));
        }

        fclose(out);
    }
}

#else

namespace perf
{
    void profile_init(){}


    void profile_clear(){}


    void profile_report(){}
}

#endif // SIMAGE_NO_PROFILE