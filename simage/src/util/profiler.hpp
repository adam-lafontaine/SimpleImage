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
        CopyView,
        CopyViewGray,
        FillView,
        FillViewGray,
        SplitRGB,
        AlphaBlendView,
        ThresholdViewMin,
        ThresholdViewMinMax,
        BlurViewGray,
        GradientViewGray,
        GradientXYViewGray,
        RotateView,
        RotateViewGray,
        CentroidView,
        CentroidViewGray,
        Skeleton,

        Count,
        None = -1
    };


    inline cstr to_cstr(ProfileLabel label)
    {
        using PL = perf::ProfileLabel;

        switch(label)
        {
            case PL::CopyView: return "CopyView";
            case PL::CopyViewGray: return "CopyViewGray";
            case PL::FillView: return "FillView";
            case PL::FillViewGray: return "FillViewGray";
            case PL::SplitRGB: return "SplitRGB";
            case PL::AlphaBlendView: return "AlphaBlend";
            case PL::ThresholdViewMin: return "ThresholdViewMin";
            case PL::ThresholdViewMinMax: return "ThresholdViewMinMax";
            case PL::BlurViewGray: return "BlurViewGray";
            case PL::GradientViewGray: return "GradientViewGray";
            case PL::GradientXYViewGray: return "GradientXYViewGray";
            case PL::RotateView: return "RotateView";
            case PL::RotateViewGray: return "RotateViewGray";
            case PL::CentroidView: return "CentroidView";
            case PL::CentroidViewGray: return "CentroidViewGray";
            case PL::Skeleton: return "Skeleton";
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