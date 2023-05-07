#include "../../tests_include.hpp"


void split_channels_red_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto h_pixels = img::create_buffer32(width * height);
    auto h_src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, h_pixels);

    auto d_pixels = img::create_device_buffer32(width * height);
    auto d_src = img::make_view(width, height, d_pixels);
    
    auto d_buffer = img::create_device_buffer8(width * height * 3);

    auto d_red = img::make_view(width, height, d_buffer);
    auto d_green = img::make_view(width, height, d_buffer);
    auto d_blue = img::make_view(width, height, d_buffer);

    auto h_buffer = img::create_device_buffer8(width * height);
    auto h_dst = img::make_view(width, height, h_buffer);

    img::copy_to_device(h_src, d_src);

    img::split_rgb(d_src, d_red, d_green, d_blue);

    //img::copy_to_host(d_red, h_dst);    

    //img::map_gray(h_dst, out);

    img::destroy_image(image);
    img::destroy_buffer(h_pixels);
    img::destroy_buffer(d_pixels);
    img::destroy_buffer(d_buffer);
    img::destroy_buffer(h_buffer);
}


void split_channels_green_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto h_pixels = img::create_buffer32(width * height);
    auto h_src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, h_pixels);

    auto d_pixels = img::create_device_buffer32(width * height);
    auto d_src = img::make_view(width, height, d_pixels);
    
    auto d_buffer = img::create_device_buffer8(width * height * 3);

    auto d_red = img::make_view(width, height, d_buffer);
    auto d_green = img::make_view(width, height, d_buffer);
    auto d_blue = img::make_view(width, height, d_buffer);

    auto h_buffer = img::create_device_buffer8(width * height);
    auto h_dst = img::make_view(width, height, h_buffer);

    img::copy_to_device(h_src, d_src);

    img::split_rgb(d_src, d_red, d_green, d_blue);

    img::copy_to_host(d_green, h_dst);    

    img::map_gray(h_dst, out);

    img::destroy_image(image);
    img::destroy_buffer(h_pixels);
    img::destroy_buffer(d_pixels);
    img::destroy_buffer(d_buffer);
    img::destroy_buffer(h_buffer);
}


void split_channels_blue_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto h_pixels = img::create_buffer32(width * height);
    auto h_src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, h_pixels);

    auto d_pixels = img::create_device_buffer32(width * height);
    auto d_src = img::make_view(width, height, d_pixels);
    
    auto d_buffer = img::create_device_buffer8(width * height * 3);

    auto d_red = img::make_view(width, height, d_buffer);
    auto d_green = img::make_view(width, height, d_buffer);
    auto d_blue = img::make_view(width, height, d_buffer);

    auto h_buffer = img::create_device_buffer8(width * height);
    auto h_dst = img::make_view(width, height, h_buffer);

    img::copy_to_device(h_src, d_src);

    img::split_rgb(d_src, d_red, d_green, d_blue);

    img::copy_to_host(d_blue, h_dst);    

    img::map_gray(h_dst, out);

    img::destroy_image(image);
    img::destroy_buffer(h_pixels);
    img::destroy_buffer(d_pixels);
    img::destroy_buffer(d_buffer);
    img::destroy_buffer(h_buffer);
}