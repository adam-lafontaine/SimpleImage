#include "tests_include.hpp"


void alpha_blend_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Buffer32 pixels;
    mb::create_buffer(pixels, width * height * 2);

    img::Image vette;
    img::Image caddy;

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, pixels);
    auto cur = img::make_view_resized_from_file(CADILLAC_PATH, caddy, width, height, pixels);

    img::for_each_pixel(src, [](img::Pixel& p) { p.rgba.alpha = 128; });

    img::alpha_blend(src, cur, out);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    mb::destroy_buffer(pixels);
}