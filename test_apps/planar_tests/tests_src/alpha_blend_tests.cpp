#include "../../tests_include.hpp"


void alpha_blend_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    u32 n_channels = 2 + 4 + 2 * 3;

    auto buffer = img::create_buffer32(width * height * n_channels);

    img::Image img_vette;
    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, img_vette, width, height, buffer);

    img::Image img_caddy;
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, img_caddy, width, height, buffer);

    auto src = img::make_view_4(width, height, buffer);
    auto cur = img::make_view_3(width, height, buffer);
    auto dst = img::make_view_3(width, height, buffer);

    img::map_rgba(vette, src);
    img::map_rgb(caddy, cur);

    auto alpha = img::select_channel(src, img::RGBA::A);
    img::fill(alpha, 0.5f);

    img::alpha_blend(src, cur, dst);

    img::map_rgb(dst, out);

    img::destroy_buffer(buffer);
    img::destroy_image(img_vette);
    img::destroy_image(img_caddy);
}