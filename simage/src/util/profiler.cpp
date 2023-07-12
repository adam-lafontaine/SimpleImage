#ifndef SIMAGE_NO_PROFILE

#include "profiler.hpp"

#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <algorithm>


namespace
{
    class ProfileRecord
    {
    public:

        cstr label = 0;

        u64 cpu_total = 0;

        u32 hit_count = 0;

        b32 is_active = false;

        u64 cpu_avg() { return cpu_total == 0 ? 1 : cpu_total / hit_count; }
    };
}


static ProfileRecord g_records[1024] = { 0 };

static u32 g_n_records = 0;


static int find_record_id(cstr label)
{
    if (g_n_records == 0)
    {
        return -1;
    }

    for (u32 i = 0; i < g_n_records; ++i)
    {
        auto& record = g_records[i];
        if (strcmp(label, record.label) == 0)
        {
            return (int)i;
        }
    }

    return -1;
}


static int get_record_id(cstr label)
{
    auto id = find_record_id(label);
    if (id < 0)
    {
        id = g_n_records++;
        g_records[id].label = label;
    }

    return id;
}


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
    Profile::Profile(cstr label)
    {
        profile_id = get_record_id(label);

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
        for (u32 i = 0; i < 1024; ++i)
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
        auto end = g_records + g_n_records;
        auto const compare = [](auto lhs, auto rhs)
        {
            return 
                lhs.hit_count > 0 && 
                rhs.hit_count > 0 && 
                lhs.cpu_avg() < rhs.cpu_avg(); 
        };

        auto min = std::min_element(begin, end, compare);

        FILE* out = fopen("build_files/profile.txt", "a");

        print(out, "\nProfile Report:\n");

        int label_len = 10;
        int abs_len = 6;
        int rel_len = 1;
        for (u32 i = 0; i < g_n_records; ++i)
        {
            auto& record = g_records[i];
            if (record.hit_count == 0)
            {
                continue;
            }

            auto len = strlen(record.label);
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

        for (u32 i = 0; i < g_n_records; ++i)
        {
            auto& record = g_records[i];
            if (record.hit_count == 0)
            {
                continue;
            }

            auto label = record.label;
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