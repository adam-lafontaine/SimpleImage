#pragma once

namespace perf
{
    void profile_init();


    void profile_clear();


    void profile_report();
}


#ifndef SIMAGE_NO_PROFILE

#include "types.hpp"


namespace perf
{
    enum class ProfileLabel : int
    {
        Anything,

        Count,
        None = -1
    };


    inline cstr to_cstr(ProfileLabel label)
    {
        using PL = perf::ProfileLabel;

        switch(label)
        {
            default: return "err";
        }

        return "err";
    }
}


namespace perf
{
    class Profile
    {
    public:
        
        Profile(ProfileLabel label);

        ~Profile();

        int profile_id = 0;

    
    private:

        u64 cpu_start;
        u64 cpu_end;
    };
}


using PL = perf::ProfileLabel;


#define PROFILE_BLOCK(label) perf::Profile profile_block(label);

#else

#define PROFILE_BLOCK(label) /* SIMAGE_NO_PROFILE */

#endif