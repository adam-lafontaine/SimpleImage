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

    img::for_each_pixel(src, [](img::Pixel& p) { p.rgba.alpha = 128; });
}