#include "tests_include.hpp"


#include <array>

class HSVr32
{
public:
    r32 hue;
    r32 sat;
    r32 val;
};


class RGBr32
{
public:
    r32 red;
    r32 green;
    r32 blue;
};


constexpr r32 zerof = 1.0f / 360.0f;


// https://github.com/QuantitativeBytes/qbColor/blob/main/qbColor.cpp
// https://www.youtube.com/watch?v=I8i0W8ve-JI



static constexpr RGBr32 hsv_rgb(r32 h, r32 s, r32 v)
{
    if (v <= zerof)
    {
        return { 0.0f, 0.0f, 0.0f };
    }

    if (s <= zerof)
    {
        return { v, v, v };
    }

    if (h < 0)
    {
        //assert(false);
        return { 0.0f, 0.0f, 0.0f };
    }

    auto max = v;
    auto range = s * v;
    auto min = max - range;

    auto d = h * 6.0f; // 360.0f / 60.0f;

    auto h_id = (int)d;

    auto rise = min + (d - h_id) * range;
    auto fall = max - (d - h_id) * range;

    r32 r = 0.0f;
    r32 g = 0.0f;
    r32 b = 0.0f;

    switch (h_id)
    {
    case 0:
        r = max;
        g = rise;
        b = min;
        break;
    case 1:
        r = fall;
        g = max;
        b = min;
        break;
    case 2:
        r = min;
        g = max;
        b = rise;
        break;
    case 3:
        r = min;
        g = fall;
        b = max;
        break;
    case 4:
        r = rise;
        g = min;
        b = max;
        break;
    default:
        r = max;
        g = min;
        b = fall;
        break;
    }

    return { r, g, b };
}


//constexpr size_t H_PRECISION = 360;
//
//constexpr std::array<RGBr32, H_PRECISION> make_max_rgb_values()
//{
//    std::array<RGBr32, H_PRECISION> rgb_values = {};
//
//    for (int i = 0; i < H_PRECISION; ++i)
//    {
//        auto h = (r32)i / H_PRECISION;
//        rgb_values[i] = hsv_rgb(h, 1.0f, 1.0f);
//    }
//
//    return rgb_values;
//}


static HSVr32 rgb_hsv(r32 r, r32 g, r32 b)
{
    //constexpr auto rgb_values = make_max_rgb_values();

    auto max = std::max(r, std::max(g, b));
    auto min = std::min(r, std::min(g, b));

    auto constexpr equals = [](r32 lhs, r32 rhs) { return std::abs(lhs - rhs) < zerof; };

    auto const r_max = equals(r, max);
    auto const r_min = equals(r, min);
    auto const g_max = equals(g, max);
    auto const g_min = equals(g, min);
    auto const b_max = equals(b, max);
    auto const b_min = equals(b, min);

    r32 h = 0.0f;
    r32 s = 0.0f;
    r32 v = max;

    if (equals(max, min))
    {
        h = -1.0f;
        s = 0.0f;

        return { h, s, v };
    }

    auto range = max - min;

    s = range / v;        

    if (r_max && g_min && b_min)
    {
        h = 0.0f;
        return { h, s, v };
    }

    if (r_min && g_max && b_min)
    {
        h = 120.0f / 360.0f;
        return { h, s, v };
    }

    if (r_min && g_min && b_max)
    {
        h = 240.0f / 360.0f;
        return { h, s, v };
    }

    u32 h_id =
        (r_max && b_min) ? 0 :
        (g_max && b_min) ? 1 :
        (g_max && r_min) ? 2 :
        (b_max && r_min) ? 3 :
        (b_max && g_min) ? 4 : 5;
        //(g_min && r_max) ? 5;

    auto h_360 = h_id * 60.0f;

    switch (h_id)
    {
    case 0:
        h_360 += 60.0f * (g - min) / range;
        break;
    case 1:
        h_360 += 60.0f * (max - r) / range;
        break;
    case 2:
        h_360 += 60.0f * (b - min) / range;
        break;
    case 3:
        h_360 += 60.0f * (max - g) / range;
        break;
    case 4:
        h_360 += 60.0f * (r - min) / range;
        break;
    case 5:
        h_360 += 60.0f * (max - b) / range;
        break;
    default:
        h_360 = -360.0f;
    }

    h = h_360 / 360.0f;

    return { h, s, v };
}



static bool conversion_test()
{
    printf("converstion_test\n");
    auto const not_equals = [](r32 lhs, r32 rhs) { return std::abs(lhs - rhs) > (1.5f / 255.0f); };

    

    for (u32 r = 0; r < 255; ++r)
    {
        auto red = r / 255.0f;

        for (u32 g = 0; g < 255; ++g)
        {
            auto green = g / 255.0f;

            for (u32 b = 0; b < 255; ++b)
            {
                auto blue = b / 255.0f;

                auto hsv = rgb_hsv(red, green, blue);
                auto rgb = hsv_rgb(hsv.hue, hsv.sat, hsv.val);

                if (not_equals(red, rgb.red) || not_equals(green, rgb.green) || not_equals(blue, rgb.blue))
                {
                    printf("FAIL\n");
                    return false;
                }
            }
        }
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
    //write_image(caddy, "caddy_1.bmp");

    auto vette_v = img::make_view(vette);
    auto caddy_v = img::make_view(caddy);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 3 * 2);

    auto hsv_vette = img::make_view_3(width, height, buffer);
    //auto hsv_caddy = img::make_view_3(width, height, buffer);

    img::map_rgb_hsv(vette_v, hsv_vette);
    //img::map_rgb_hsv(caddy_v, hsv_caddy);

    //img::map_hsv_rgb(hsv_caddy, vette_v);
    //write_image(vette, "vette_2.bmp");

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
        conversion_test() &&
        map_hsv_test() &&
        map_hsv_planar_test();

    if (result)
    {
        printf("map_rgb_hsv tests OK\n");
    }
    
    return result;
}