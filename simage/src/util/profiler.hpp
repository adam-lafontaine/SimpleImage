#pragma once

#include "types.hpp"


namespace perf
{
    void profile_init();


    void profile_clear();


    void profile_report();
}


namespace perf
{
    class Profile
    {
    public:
        
        Profile(cstr label);

        ~Profile();

        int profile_id = 0;

    
    private:

        u64 cpu_start;
        u64 cpu_end;
    };

}


//#define PROFILE_BLOCK(label) perf::Profile profile_block(label);

//#define PROFILE(func_call, label) [&](){ perf::Profile p(label); return func_call; }();

#define PROFILE(func_call) [&](){ perf::Profile p( #func_call ); return func_call; }();

//#define PROFILE(func, ...) [&](){ perf::Profile p("\""#func"\""); return func(...); }();
