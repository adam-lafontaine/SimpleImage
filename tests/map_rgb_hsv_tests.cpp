#include "tests_include.hpp"

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


static HSVr32 rgb_hsv(r32 r, r32 g, r32 b)
{
    auto max = std::max(r, std::max(g, b));
    auto min = std::min(r, std::max(g, b));

    auto const equals = [](r32 lhs, r32 rhs) { return std::abs(lhs - rhs) < (1.5f / 255.0f); };

    auto c = max - min;
    auto value = max;

    auto sat = equals(max, 0.0f) ? 0.0f : (c / value);

    auto hue = 60.0f;

    if (equals(max, min))
    {
        hue = 0.0f;
    }
    else if (equals(max, r))
    {
        hue *= ((g - b) / c);
    }
    else if (equals(max, g))
    {
        hue *= ((b - r) / c + 2);
    }
    else
    {
        hue *= ((r - g) / c + 4);
    }

    hue /= 360.0f;

    return { hue, sat, value };
}


static RGBr32 hsv_rgb(r32 h, r32 s, r32 v)
{
    auto c = s * v;
    auto m = v - c;

    auto d = h * 6.0f; // 360.0f / 60.0f;

    auto x = c * (1.0f - std::abs(std::fmod(d, 2.0f) - 1.0f));

    auto r = m;
    auto g = m;
    auto b = m;

    switch (int(d))
    {
    case 0:
        r += c;
        g += x;
        break;
    case 1:
        r += x;
        g += c;
        break;
    case 2:
        g += c;
        b += x;
        break;
    case 3:
        g += x;
        b += c;
        break;
    case 4:
        r += x;
        b += c;
        break;
    default:
        r += c;
        b += x;
        break;
    }

    return { r, g, b };
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