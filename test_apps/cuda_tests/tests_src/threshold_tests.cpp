#include "../../tests_include.hpp"


void threshold_min_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::threshold(d_src, d_dst, 75);

    img::copy_to_host(d_dst, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}


void threshold_min_max_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto pixels = img::create_buffer8(width * height * 2);
    auto d_pixels = img::create_device_buffer8(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    auto dst = img::make_view(width, height, pixels);

    auto d_src = img::make_device_view(width, height, d_pixels);
    auto d_dst = img::make_device_view(width, height, d_pixels);

    img::copy_to_device(src, d_src);

    img::threshold(d_src, d_dst, 30, 200);

    img::copy_to_host(d_dst, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(d_pixels);
}