#include "../../tests_include.hpp"


void split_channels_red_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto pixels = img::create_buffer32(width * height);

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    
    auto buffer = img::create_buffer8(width * height * 3);

    auto red = img::make_view(width, height, buffer);
    auto green = img::make_view(width, height, buffer);
    auto blue = img::make_view(width, height, buffer);

    img::split_rgb(src, red, green, blue);

    img::map_gray(red, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(buffer);
}


void split_channels_green_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    img::Buffer32 pixels;
    mb::create_buffer(pixels, width * height);

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    
    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto red = img::make_view(width, height, buffer);
    auto green = img::make_view(width, height, buffer);
    auto blue = img::make_view(width, height, buffer);

    img::split_rgb(src, red, green, blue);

    img::map_gray(green, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(buffer);
}


void split_channels_blue_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    img::Buffer32 pixels;
    mb::create_buffer(pixels, width * height);

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);
    
    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto red = img::make_view(width, height, buffer);
    auto green = img::make_view(width, height, buffer);
    auto blue = img::make_view(width, height, buffer);

    img::split_rgb(src, red, green, blue);

    img::map_gray(blue, out);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
    img::destroy_buffer(buffer);
}