#include "tests_include.hpp"
#include "../src/util/execute.hpp"
#include "../src/util/color_space.hpp"

#include <vector>
#include <algorithm>

namespace rng = std::ranges;

static bool yuv_conversion_test()
{
    printf("yuv converstion_test\n");
    auto const not_equals = [](r32 lhs, r32 rhs) { return std::abs(lhs - rhs) > (1.0f / 255.0f); };

    std::vector<int> results(256, 1);

    auto const red_func = [&](u32 r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 256; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 256; ++b)
            {
                auto blue = b / 255.0f;

                auto yuv = yuv::r32_from_rgb_r32(red, green, blue);
                auto rgb = yuv::r32_to_rgb_r32(yuv.y, yuv.u, yuv.v);

                if (not_equals(red, rgb.red) || not_equals(green, rgb.green) || not_equals(blue, rgb.blue))
                {
                    results[r] = false;
                    return;
                }
            }
        }
    };

    process_range(0, 256, red_func);

    if (rng::any_of(results, [](int r) { return !r; }))
    {
        printf("FAIL\n");
        return false;
    }

    printf("OK\n");
    return true;
}


bool map_rgb_yuv_tests()
{
    printf("\n*** map_rgb_yuv tests ***\n");

    auto result = 
        yuv_conversion_test();

    if (result)
    {
        printf("map_rgb_yuv tests OK\n");
    }
    
    return result;
}