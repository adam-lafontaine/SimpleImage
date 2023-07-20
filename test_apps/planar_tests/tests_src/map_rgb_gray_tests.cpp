#include "../../tests_include.hpp"


void map_rgba_tests(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer32(width * height * 5);

    img::Image image;

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer);
    auto dst = img::make_view_4(width, height, buffer);

    img::map_rgba(src, dst);

    img::map_rgba(dst, out);

    mb::destroy_buffer(buffer);
    img::destroy_image(image);
}


void map_rgb_tests(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer32(width * height * 4);

    img::Image image;

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view_3(width, height, buffer);

    img::map_rgb(src, dst);

    img::map_rgba(dst, out);

    mb::destroy_buffer(buffer);
    img::destroy_image(image);
}


void map_gray_tests(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer8 = img::create_buffer8(width * height);

    auto buffer32 = img::create_buffer32(width * height);

    img::ImageGray image;

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer8);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(src, dst);

    img::map_rgba(dst, out);

    mb::destroy_buffer(buffer8);
    mb::destroy_buffer(buffer32);
    img::destroy_image(image);
}