#include "../../tests_include.hpp"


void copy_rgba_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    auto n_channels32 = 10;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    img::Image vette_img;
    img::Image caddy_img;

    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, vette_img, width, height, buffer32);
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, caddy_img, width, height, buffer32);

    auto vette_4 = img::make_view_4(width, height, buffer32);
    auto caddy_4 = img::make_view_4(width, height, buffer32);

    img::map_rgba(vette, vette_4);
    img::map_rgba(caddy, caddy_4);

    auto full = make_range(out);

    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    img::copy(img::sub_view(vette_4, mid), img::sub_view(caddy_4, mid));

    img::map_rgba(caddy_4, out);

    img::destroy_buffer(buffer32);
    img::destroy_image(vette_img);
    img::destroy_image(caddy_img);
}


void copy_rgb_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    auto n_channels32 = 8;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    img::Image vette_img;
    img::Image caddy_img;

    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, vette_img, width, height, buffer32);
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, caddy_img, width, height, buffer32);

    auto vette_3 = img::make_view_3(width, height, buffer32);
    auto caddy_3 = img::make_view_3(width, height, buffer32);

    img::map_rgb(vette, vette_3);
    img::map_rgb(caddy, caddy_3);

    auto full = make_range(out);

    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    img::copy(img::sub_view(caddy_3, mid), img::sub_view(vette_3, mid));

    img::map_rgba(vette_3, out);

    img::destroy_buffer(buffer32);
    img::destroy_image(vette_img);
    img::destroy_image(caddy_img);
}


void copy_gray_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    auto n_channels32 = 2;
    auto n_channels8 = 2;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    img::ImageGray vette_img;
    img::ImageGray caddy_img;

    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, vette_img, width, height, buffer8);
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, caddy_img, width, height, buffer8);

    auto vette_1 = img::make_view_1(width, height, buffer32);
    auto caddy_1 = img::make_view_1(width, height, buffer32);

    img::map_gray(vette, vette_1);
    img::map_gray(caddy, caddy_1);

    auto full = make_range(out);

    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    img::copy(img::sub_view(vette_1, mid), img::sub_view(caddy_1, mid));

    img::map_rgba(caddy_1, out);

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
    img::destroy_image(vette_img);
    img::destroy_image(caddy_img);
}