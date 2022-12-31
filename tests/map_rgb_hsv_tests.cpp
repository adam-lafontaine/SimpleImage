#include "tests_include.hpp"
#include "../src/util/execute.hpp"
#include "../src/util/color_space.hpp"

#include <vector>
#include <algorithm>

namespace rng = std::ranges;

static bool hsv_conversion_test()
{
    printf("hsv converstion_test\n");
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

                auto hsv = hsv::r32_from_rgb_r32(red, green, blue);
                auto rgb = hsv::r32_to_rgb_r32(hsv.hue, hsv.sat, hsv.val);

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


static bool map_hsv_test()
{
    auto title = "map_hsv_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;

    Image caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    Image caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 3 * 2);

    auto hsv_vette = img::make_view_3(width, height, buffer);
    auto hsv_caddy = img::make_view_3(width, height, buffer);

    img::map_rgb_hsv(vette_v, hsv_vette);
    img::map_rgb_hsv(caddy_v, hsv_caddy);

    img::map_hsv_rgb(hsv_caddy, vette_v);
    write_image(vette, "vette_2.bmp");

    img::map_hsv_rgb(hsv_vette, caddy_v);
    write_image(caddy, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}


static bool map_hsv_planar_test()
{
    auto title = "map_hsv_planar_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;

    Image caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    Image caddy;
    caddy.width = width;
    caddy.height = height;
    img::resize_image(caddy_read, caddy);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 3 * 4);

    auto rgb_vette = img::make_view_3(width, height, buffer);
    auto rgb_caddy = img::make_view_3(width, height, buffer);

    auto hsv_vette = img::make_view_3(width, height, buffer);
    auto hsv_caddy = img::make_view_3(width, height, buffer);

    img::map_rgb(vette_v, rgb_vette);
    img::map_rgb(caddy_v, rgb_caddy);

    img::map_rgb_hsv(rgb_vette, hsv_vette);
    img::map_rgb_hsv(rgb_caddy, hsv_caddy);

    img::map_hsv_rgb(hsv_vette, rgb_caddy);
    img::map_hsv_rgb(hsv_caddy, rgb_vette);

    img::map_rgb(rgb_vette, vette_v);
    write_image(vette, "vette_2.bmp");

    img::map_rgb(rgb_caddy, caddy_v);
    write_image(caddy, "caddy_2.bmp");

    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);

    return true;
}


bool map_rgb_hsv_tests()
{
    printf("\n*** map_rgb_hsv tests ***\n");

    auto result = 
        hsv_conversion_test() &&
        map_hsv_test() &&
        map_hsv_planar_test();

    if (result)
    {
        printf("map_rgb_hsv tests OK\n");
    }
    
    return result;
}