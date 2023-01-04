#include "tests_include.hpp"


void fill_platform_view_test(img::View const& out_view)
{
    auto full = make_range(out_view);

    auto top = make_range(out_view.width, out_view.height / 3);

    auto mid = full;
    mid.y_begin = top.y_end;
    mid.y_end = mid.y_begin + out_view.height / 3;

    auto bottom = full;
    bottom.y_begin = mid.y_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    img::fill(img::sub_view(out_view, top), red);
    img::fill(img::sub_view(out_view, mid), green);
    img::fill(img::sub_view(out_view, bottom), blue);
}