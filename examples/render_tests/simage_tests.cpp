#include "../../src/simage/simage.hpp"
#include "tests_include.hpp"

namespace img = simage;


void alpha_blend_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette_read;
	img::read_image_from_file(CORVETTE_PATH, vette_read);    

    img::Image caddy_read;
	img::read_image_from_file(CADILLAC_PATH, caddy_read);

    img::Image vette;
    img::create_image(vette, width, height);
    img::resize_image(vette_read, vette);

    img::Image caddy;
    img::create_image(caddy, width, height);
    img::resize_image(caddy_read, caddy);

    img::Buffer16 buffer;
    mb::create_buffer(buffer, width * height * 10);

    auto rgba_src = img::make_view_4(width, height, buffer);
    auto rgb_cur = img::make_view_3(width, height, buffer);
    auto rgb_dst = img::make_view_3(width, height, buffer);

    img::map_rgba(vette, rgba_src);
    img::map_rgb(caddy, rgb_cur);

    auto alpha_src = img::select_channel(rgba_src, img::RGBA::A);
    img::fill(alpha_src, 128);

    img::alpha_blend(rgba_src, rgb_cur, rgb_dst);

    img::map_rgb(rgb_dst, out);

    img::destroy_image(vette_read);
    img::destroy_image(vette);
    img::destroy_image(caddy_read);
    img::destroy_image(caddy);
    mb::destroy_buffer(buffer);
}