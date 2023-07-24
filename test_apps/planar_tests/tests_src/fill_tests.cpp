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

    auto bottom = full;
    bottom.y_begin = mid.y_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    auto buffer = img::create_buffer32(width * height * 4);

    auto rgba = img::make_view_4(width, height, buffer);

    img::fill(img::sub_view(rgba, top), red);
    img::fill(img::sub_view(rgba, mid), green);
    img::fill(img::sub_view(rgba, bottom), blue);

    img::map_rgba(rgba, out);

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

    auto rgb = img::make_view_3(width, height, buffer);

    img::fill(img::sub_view(rgb, left), red);
    img::fill(img::sub_view(rgb, mid), green);
    img::fill(img::sub_view(rgb, right), blue);

    img::map_rgba(rgb, out);

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

    auto bottom = full;
    bottom.y_begin = mid.y_end;

    u8 const black = 0;
    u8 const gray = 128;
    u8 const white = 255;

    auto buffer = img::create_buffer32(width * height);

    auto view = img::make_view_1(width, height, buffer);

    img::fill(img::sub_view(view, top), black);
    img::fill(img::sub_view(view, mid), gray);
    img::fill(img::sub_view(view, bottom), white);

    img::map_rgba(view, out);

    mb::destroy_buffer(buffer);
}