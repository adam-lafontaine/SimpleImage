#include "tests_include.hpp"


void fill_test(img::View const& out)
{
    auto full = make_range(out);

    auto top = make_range(out.width, out.height / 3);

    auto mid = full;
    mid.y_begin = top.y_end;
    mid.y_end = mid.y_begin + out.height / 3;

    auto bottom = full;
    bottom.y_begin = mid.y_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    img::fill(img::sub_view(out, top), red);
    img::fill(img::sub_view(out, mid), green);
    img::fill(img::sub_view(out, bottom), blue);
}