#include "tests_include.hpp"


void fill_test(img::View const& out)
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

    img::fill(img::sub_view(out, top), red);
    img::fill(img::sub_view(out, mid), green);
    img::fill(img::sub_view(out, bottom), blue);
}


void fill_gray_test(img::View const& out)
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

    u8 const black = 0;
    u8 const gray = 128;
    u8 const white = 255;

    img::Buffer8 buffer;
	mb::create_buffer(buffer, width * height);

	auto dst = img::make_view_gray(width, height, buffer);

    img::fill(img::sub_view(dst, left), black);
    img::fill(img::sub_view(dst, mid), gray);
    img::fill(img::sub_view(dst, right), white);

    img::map_gray(dst, out);

    mb::destroy_buffer(buffer);
}