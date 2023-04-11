#include "tests_include.hpp"


void alpha_blend_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

    img::Image caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);

    img::Image src;
    src.width = width;
    src.height = height;
    img::resize_image(vette, src);

    img::Image cur;
    cur.width = width;
    cur.height = height;
    img::resize_image(caddy, cur);

    img::for_each_pixel(img::make_view(src), [](img::Pixel& p) { p.rgba.alpha = 128; });

    img::alpha_blend(img::make_view(src), img::make_view(cur), out);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_image(src);
    img::destroy_image(cur);
}