#include "../../tests_include.hpp"


void copy_rgba_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    auto n_channels32 = 5;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    img::Image vette_img;
    img::Image caddy_img;

    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, vette_img, width, height, buffer32);
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, caddy_img, width, height, buffer32);

    auto full = make_range(out);
    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    auto src_4 = img::make_view_4(mid, buffer32);
    auto dst_4 = img::make_view_4(mid, buffer32);

    img::map_rgba(img::sub_view(vette, mid), src_4);

    img::copy(src_4, dst_4);

    img::map_rgba(dst_4, img::sub_view(caddy, mid));    
    img::copy(caddy, out);

    img::destroy_buffer(buffer32);
    img::destroy_image(vette_img);
    img::destroy_image(caddy_img);
}


void copy_rgb_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;
    auto n_channels32 = 4;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    img::Image vette_img;
    img::Image caddy_img;

    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, vette_img, width, height, buffer32);
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, caddy_img, width, height, buffer32);

    auto full = make_range(out);
    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    auto src_3 = img::make_view_3(mid, buffer32);
    auto dst_3 = img::make_view_3(mid, buffer32);

    img::map_rgb(img::sub_view(caddy, mid), src_3);

    img::copy(src_3, dst_3);

    img::map_rgba(dst_3, img::sub_view(vette, mid));
    img::copy(vette, out);

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

    auto full = make_range(out);
    auto mid = full;
    mid.y_begin = height / 3;
    mid.y_end = mid.y_begin + height / 3;

    auto src_1 = img::make_view_1(mid, buffer32);
    auto dst_1 = img::make_view_1(mid, buffer32);

    img::map_gray(img::sub_view(vette, mid), src_1);

    img::copy(src_1, dst_1);

    img::map_gray(dst_1, img::sub_view(caddy, mid));

    img::map_rgba(caddy, out);

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
    img::destroy_image(vette_img);
    img::destroy_image(caddy_img);
}