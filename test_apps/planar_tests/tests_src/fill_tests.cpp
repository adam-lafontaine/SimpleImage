#include "../../tests_include.hpp"


void fill_rgba_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto full = make_range(out);

    auto top = make_range(width, height / 3);

    auto mid = full;
    mid.y_begin = top.y_end;
    mid.y_end = mid.y_begin + height / 3;

    auto btm = full;
    btm.y_begin = mid.y_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    auto buffer = img::create_buffer32(width * height * 4);

    auto red_4 = img::make_view_4(top, buffer);
    auto green_4 = img::make_view_4(mid, buffer);
    auto blue_4 = img::make_view_4(btm, buffer);

    img::fill(red_4, red);
    img::fill(green_4, green);
    img::fill(blue_4, blue);

    img::map_rgba(red_4, img::sub_view(out, top));
    img::map_rgba(green_4, img::sub_view(out, mid));
    img::map_rgba(blue_4, img::sub_view(out, btm));

    mb::destroy_buffer(buffer);
}


void fill_rgb_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto full = make_range(out);

    auto left = make_range(width / 3, height);

    auto mid = full;
    mid.x_begin = left.x_end;
    mid.x_end = mid.x_begin + width / 3;

    auto right = full;
    right.x_begin = mid.x_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    auto buffer = img::create_buffer32(width * height * 3);

    auto red_3 = img::make_view_3(left, buffer);
    auto green_3 = img::make_view_3(mid, buffer);
    auto blue_3 = img::make_view_3(left, buffer);

    img::fill(red_3, red);
    img::fill(green_3, green);
    img::fill(blue_3, blue);

    img::map_rgba(red_3, img::sub_view(out, left));
    img::map_rgba(green_3, img::sub_view(out, mid));
    img::map_rgba(blue_3, img::sub_view(out, left));

    mb::destroy_buffer(buffer);
}


void fill_gray_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto full = make_range(out);

    auto top = make_range(width, height / 3);

    auto mid = full;
    mid.y_begin = top.y_end;
    mid.y_end = mid.y_begin + height / 3;

    auto btm = full;
    btm.y_begin = mid.y_end;

    u8 const black = 0;
    u8 const gray = 128;
    u8 const white = 255;

    auto buffer = img::create_buffer32(width * height);

    auto black_1 = img::make_view_1(top, buffer);
    auto gray_1 = img::make_view_1(mid, buffer);
    auto white_1 = img::make_view_1(btm, buffer);

    img::fill(black_1, black);
    img::fill(gray_1, gray);
    img::fill(white_1, white);

    img::map_rgba(black_1, img::sub_view(out, top));
    img::map_rgba(gray_1, img::sub_view(out, mid));
    img::map_rgba(white_1, img::sub_view(out, btm));

    mb::destroy_buffer(buffer);
}

