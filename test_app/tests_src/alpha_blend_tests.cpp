#include "tests_include.hpp"


void alpha_blend_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette;
    img::Image vette2;
    img::Image caddy;
    img::Image caddy2;

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, vette, vette2, width, height);
    auto cur = img::make_view_resized_from_file(CADILLAC_PATH, caddy, caddy2, width, height);

    img::for_each_pixel(src, [](img::Pixel& p) { p.rgba.alpha = 128; });

    img::alpha_blend(src, cur, out);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_image(vette2);
    img::destroy_image(caddy2);
}