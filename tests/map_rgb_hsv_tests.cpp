#include "tests_include.hpp"
#include "../src/util/execute.hpp"
#include "../src/util/color_space.hpp"

#include <vector>
#include <algorithm>

namespace rng = std::ranges;


static bool equals(r32 lhs, r32 rhs)
{
    return std::abs(lhs - rhs) < (1.0f / 255.0f);
}


static bool equals(u8 lhs, u8 rhs)
{
    return std::abs((int)lhs - (int)rhs) <= 2;
}


static bool hsv_conversion_r32_test()
{
    printf("hsv converstion_r32_test\n");

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

                if (!equals(red, rgb.red) || !equals(green, rgb.green) || !equals(blue, rgb.blue))
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


static bool map_hsv_gray_test()
{
    auto title = "map_hsv_gray_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    GrayImage vette_gray;
    img::read_image_from_file(CORVETTE_PATH, vette_gray);
    auto width = vette_gray.width;
    auto height = vette_gray.height;

    GrayImage caddy_read;
    img::read_image_from_file(CADILLAC_PATH, caddy_read);

    GrayImage caddy_gray;
    caddy_gray.width = width;
    caddy_gray.height = height;
    img::resize_image(caddy_read, caddy_gray);

    Image vette;
    img::create_image(vette, width, height);
    Image caddy;
    img::create_image(caddy, width, height);

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    img::map(img::make_view(vette_gray), vette_v);
    img::map(img::make_view(caddy_gray), caddy_v);

    write_image(vette, "vette_1.bmp");
    write_image(caddy, "caddy_1.bmp");    

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

    img::destroy_image(vette_gray);
    img::destroy_image(caddy_gray);
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


static bool hsv_draw_test()
{
    auto title = "hsv_draw_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    u8 V = 255;

    img::Image image;
    img::create_image(image, 256, 256);

    auto const row_func = [&](u32 y)
    {
        auto s = (u8)y;
        auto d = img::row_begin(image, y);
        for (u32 x = 0; x < 256; ++x)
        {
            auto h = (u8)x;
            auto rgba = hsv::u8_to_rgb_u8(h, s, V);
            auto& p = d[x].rgba;
            p.red = rgba.red;
            p.green = rgba.green;
            p.blue = rgba.blue;
            p.alpha = 255;
        }
    };

    process_range(0, 256, row_func);
    write_image(image, "hsv_255.bmp");

    V = 128;
    process_range(0, 256, row_func);
    write_image(image, "hsv_128.bmp");

    img::destroy_image(image);

    printf("OK\n");
    return true;
}


bool map_rgb_hsv_tests()
{
    printf("\n*** map_rgb_hsv tests ***\n");

    auto result = 
        hsv_conversion_r32_test() &&
        map_hsv_test() &&
        map_hsv_gray_test() &&
        map_hsv_planar_test() &&
        hsv_draw_test();

    if (result)
    {
        printf("map_rgb_hsv tests OK\n");
    }
    
    return result;
}