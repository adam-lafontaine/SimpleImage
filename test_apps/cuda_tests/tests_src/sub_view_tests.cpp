#include "../../tests_include.hpp"


void sub_view_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto r_top_left = make_range(width, height);
    r_top_left.x_end = width / 2;
    r_top_left.y_end = height / 2;

    auto r_top_right = make_range(width, height);
    r_top_right.x_begin = width / 2;
    r_top_right.y_end = height / 2;

    auto r_bottom_left = make_range(width, height);
    r_bottom_left.x_end = width / 2;
    r_bottom_left.y_begin = height / 2;

    auto r_bottom_right = make_range(width, height);
    r_bottom_right.x_begin = width / 2;
    r_bottom_right.y_begin = height / 2;

    img::Image image;

    auto h_buffer = img::create_buffer32(width * height);

    auto h_src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, h_buffer);

    auto d_buffer = img::create_device_buffer32(width * height);

    auto d_src = img::make_view(width, height, d_buffer);

    img::copy_to_device(h_src, d_src);

    img::copy_to_host(img::sub_view(d_src, r_top_left), img::sub_view(out, r_bottom_right));
    img::copy_to_host(img::sub_view(d_src, r_top_right), img::sub_view(out, r_bottom_left));
    img::copy_to_host(img::sub_view(d_src, r_bottom_left), img::sub_view(out, r_top_right));
    img::copy_to_host(img::sub_view(d_src, r_bottom_right), img::sub_view(out, r_top_left));

    img::destroy_image(image);
    img::destroy_buffer(h_buffer);
    img::destroy_buffer(d_buffer);
}