#include "tests_include.hpp"
#include "../src/util/execute.hpp"
#include "../src/util/color_space.hpp"

#include <vector>
#include <algorithm>

namespace rng = std::ranges;

static bool yuv_conversion_test()
{
    printf("yuv_converstion_test\n");
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


static bool yuv_draw_test()
{
    auto title = "yuv_draw_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    u8 Y = 255;

    img::Image image;
    img::create_image(image, 256, 256);

    auto const row_func = [&](u32 y) 
    {
        auto v = (u8)y;
        auto d = img::row_begin(image, y);
        for (u32 x = 0; x < 256; ++x)
        {
            auto u = (u8)x;
            auto rgba = yuv::u8_to_rgba_u8(Y, u, v);
            auto& p = d[x].rgba;
            p.red = rgba.red;
            p.green = rgba.green;
            p.blue = rgba.blue;
            p.alpha = 255;
        }
    };

    process_range(0, 256, row_func);
    write_image(image, "yuv_255.bmp");

    Y = 128;
    process_range(0, 256, row_func);
    write_image(image, "yuv_128.bmp");

    img::destroy_image(image);

    printf("OK\n");
    return true;
}


static bool yuv_camera_test()
{
    printf("yuv_camera_test\n");


    printf("NOT IMPLEMENTED\n");
    return true;
}


bool map_rgb_yuv_tests()
{
    printf("\n*** map_rgb_yuv tests ***\n");

    auto result = 
        yuv_conversion_test() &&
        yuv_draw_test() &&
        yuv_camera_test();

    if (result)
    {
        printf("map_rgb_yuv tests OK\n");
    }
    
    return result;
}