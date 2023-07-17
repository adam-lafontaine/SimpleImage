#include "../../tests_include.hpp"


void copy_device_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer32(width * height);
    auto d_pixels = img::create_device_buffer32(width * height);

    img::Image vette;

    auto host = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, pixels);
    auto device = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(host, device);
    img::copy_to_host(device, out);

    img::destroy_image(vette);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void copy_device_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height);

    img::ImageGray caddy;

    auto host_src = img::make_view_resized_from_file(CADILLAC_PATH, caddy, width, height, pixels);
    auto host_dst = img::make_view(width, height, pixels);
    auto device = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(host_src, device);
    img::copy_to_host(device, host_dst);

    img::map_gray(host_dst, out);

    img::destroy_image(caddy);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void copy_device_sub_view_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer32(width * height * 2);
    auto d_pixels = img::create_device_buffer32(width * height / 2);

    img::Image vette;
    img::Image caddy;

    auto hostA = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, pixels);
    auto hostB = img::make_view_resized_from_file(CADILLAC_PATH, caddy, width, height, pixels);

    auto left = make_range(width / 2, height);
    auto right = make_range(width, height);
    right.x_begin = width / 2;

    auto host_left = img::sub_view(hostA, left);
    auto host_right = img::sub_view(hostB, right);

    auto out_left = img::sub_view(out, left);
    auto out_right = img::sub_view(out, right);

    auto device = img::make_device_view(width / 2, height, d_pixels);

    img::copy_to_device(host_left, device);
    img::copy_to_host(device, out_right);

    img::copy_to_device(host_right, device);
    img::copy_to_host(device, out_left);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void copy_device_sub_view_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer8(width * height * 3);
    auto d_pixels = img::create_device_buffer8(width * height / 2);

    img::ImageGray vette;
    img::ImageGray caddy;

    auto hostA = img::make_view_resized_from_file(CORVETTE_PATH, vette, width, height, pixels);
    auto hostB = img::make_view_resized_from_file(CADILLAC_PATH, caddy, width, height, pixels);
    auto host_dst = img::make_view(width, height, pixels);

    auto left = make_range(width / 2, height);
    auto right = make_range(width, height);
    right.x_begin = width / 2;

    auto host_left = img::sub_view(hostB, left);
    auto host_right = img::sub_view(hostA, right);

    auto dst_left = img::sub_view(host_dst, left);
    auto dst_right = img::sub_view(host_dst, right);

    auto device = img::make_device_view(width / 2, height, d_pixels);

    img::copy_to_device(host_left, device);
    img::copy_to_host(device, dst_right);

    img::copy_to_device(host_right, device);
    img::copy_to_host(device, dst_left);

    img::map_gray(host_dst, out);

    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}