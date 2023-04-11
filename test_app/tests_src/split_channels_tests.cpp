#include "tests_include.hpp"


void split_channels_red_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

    img::Image rgb;
    rgb.width = width;
    rgb.height = height;
    img::resize_image(vette, rgb);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto red = img::make_view_gray(width, height, buffer);
    auto green = img::make_view_gray(width, height, buffer);
    auto blue = img::make_view_gray(width, height, buffer);

    img::split_rgb(img::make_view(rgb), red, green, blue);

    img::map_gray(red, out);

    img::destroy_image(vette);
    img::destroy_image(rgb);
    mb::destroy_buffer(buffer);
}


void split_channels_green_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);    

    img::Image rgb;
    rgb.width = width;
    rgb.height = height;
    img::resize_image(vette, rgb);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto red = img::make_view_gray(width, height, buffer);
    auto green = img::make_view_gray(width, height, buffer);
    auto blue = img::make_view_gray(width, height, buffer);

    img::split_rgb(img::make_view(rgb), red, green, blue);

    img::map_gray(green, out);

    img::destroy_image(vette);
    img::destroy_image(rgb);
    mb::destroy_buffer(buffer);
}


void split_channels_blue_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);    

    img::Image rgb;
    rgb.width = width;
    rgb.height = height;
    img::resize_image(vette, rgb);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto red = img::make_view_gray(width, height, buffer);
    auto green = img::make_view_gray(width, height, buffer);
    auto blue = img::make_view_gray(width, height, buffer);

    img::split_rgb(img::make_view(rgb), red, green, blue);

    img::map_gray(blue, out);

    img::destroy_image(vette);
    img::destroy_image(rgb);
    mb::destroy_buffer(buffer);
}