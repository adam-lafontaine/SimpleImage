#include "../../tests_include.hpp"


void alpha_blend_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer32(width * height * 2);
    auto d_pixels = img::create_device_buffer32(width * height * 3);

    img::Image vette;
    img::Image caddy;

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, pixels);
    auto cur = img::make_view_resized_from_file(CADILLAC_PATH, caddy, width, height, pixels);

    img::for_each_pixel(src, [](img::Pixel& p) { p.rgba.alpha = 128; });

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_cur = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);
    img::copy_to_device(cur, d_cur);

    img::alpha_blend(d_src, d_cur, d_dst);

    img::copy_to_host(d_dst, out);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}