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


static constexpr auto N_RECORDS = 64;


static ProfileRecord profile_records[N_RECORDS] = { 0 };

static u32 n_records = 0;


static int find_record_id(cstr label)
{
    if (n_records == 0)
    {
        return -1;
    }

    for (u32 i = 0; i < n_records; ++i)
    {
        auto& record = profile_records[i];
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
        id = n_records++;
        profile_records[id].label = label;
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

        auto& record = profile_records[profile_id];

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
        auto& record = profile_records[profile_id];

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
        auto begin = profile_records;
        auto end = profile_records + (n_records == 0 ? N_RECORDS : n_records);
        
        for (auto i = begin; i < end; ++i)
        {
            *i = { 0 };
        }
    }


    void profile_clear()
    {
        profile_init();
    }


    void profile_report()
    {
        auto begin = profile_records;
        auto end = profile_records + n_records;
        auto const compare = [](auto lhs, auto rhs)
        {
            return 
                lhs.hit_count > 0 && 
                rhs.hit_count > 0 && 
                lhs.cpu_avg() < rhs.cpu_avg(); 
        };

        std::sort(begin, end, compare);

        auto& min = profile_records[0];

        FILE* out = fopen("build_files/profile.txt", "a");

        print(out, "\nProfile Report:\n");

        int label_len = 10;
        int abs_len = 6;
        int rel_len = 1;
        for (u32 i = 0; i < n_records; ++i)
        {
            auto& record = profile_records[i];
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

            len = count_digits(record.cpu_avg() / min.cpu_avg());
            if (len > rel_len)
            {
                rel_len = len;
            }
        }

        rel_len += 3;

        //auto format = "%*s: %*.2f (%3.2e x %u)\n";
        auto format = "%3.2e: %-*s (%*.2f) x %u\n";

        for (u32 i = 0; i < n_records; ++i)
        {
            auto& record = profile_records[i];
            if (record.hit_count == 0)
            {
                continue;
            }

            auto label = record.label;
            auto cpu_abs = (f32)record.cpu_avg();
            auto cpu_rel = cpu_abs / min.cpu_avg();
            auto count = record.hit_count;            

            //print(out, format, label_len, label, rel_len, cpu_rel, cpu_abs, count);
            print(out, format, cpu_abs, label_len, label, rel_len, cpu_rel, count);
        }

        fclose(out);
    }
}