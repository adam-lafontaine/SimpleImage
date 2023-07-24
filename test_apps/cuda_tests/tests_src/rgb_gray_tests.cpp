#include "../../tests_include.hpp"


void rgb_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto h_pixels32 = img::create_buffer32(width * height);
    auto h_pixels8 = img::create_buffer8(width / 2 * height);

    auto d_pixels32 = img::create_device_buffer32(width * height / 2);
    auto d_pixels8 = img::create_device_buffer8(width * height / 2);

    img::Image vette;

    auto host_src = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, h_pixels32);
    auto host_dst = img::make_view(width / 2, height, h_pixels8);
    auto device32 = img::make_device_view(width / 2, height, d_pixels32);
    auto device8 = img::make_device_view(width / 2, height, d_pixels8);

    auto left = make_range(width / 2, height);
    auto right = make_range(width, height);
    right.x_begin = width / 2;

    auto host_left = img::sub_view(host_src, left);
    auto host_right = img::sub_view(host_src, right);

    auto out_left = img::sub_view(out, left);
    auto out_right = img::sub_view(out, right);

    img::copy_to_device(host_left, device32);
    img::map_rgb_gray(device32, device8);
    img::copy_to_host(device8, host_dst);

    img::map_rgba(host_dst, out_left);
    img::copy(host_right, out_right);

    img::destroy_image(vette);
    img::destroy_buffer(h_pixels32);
    img::destroy_buffer(h_pixels8);
    img::destroy_buffer(d_pixels32);
    img::destroy_buffer(d_pixels8);
}