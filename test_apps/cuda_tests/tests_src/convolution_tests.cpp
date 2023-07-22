#include "../../tests_include.hpp"


void blur_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::blur(d_src, d_dst);

    img::copy_to_host(d_dst, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void blur_rgb_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;
    
    auto pixels = img::create_buffer32(width * height * 2);
    auto d_pixels = img::create_device_buffer32(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::blur(d_src, d_dst);

    img::copy_to_host(d_dst, dst);

    img::copy(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void gradients_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height * 2);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CHESS_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::gradients(d_src, d_dst);

    img::copy_to_host(d_dst, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void gradients_xy_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height * 3);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CHESS_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst_x = img::make_device_view(width, height, d_pixels);
    auto d_dst_y = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::gradients_xy(d_src, d_dst_x, d_dst_y);

    auto full = make_range(width, height);

    auto left = full;
    left.x_end = width / 2;

    auto right = full;
    right.x_begin = left.x_end;

    img::copy_to_host(d_dst_x, dst);
    img::map_rgba(img::sub_view(dst, left), img::sub_view(out, left));

    img::copy_to_host(d_dst_y, dst);
    img::map_rgba(img::sub_view(dst, right), img::sub_view(out, right));

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}