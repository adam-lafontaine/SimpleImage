#include "../../tests_include.hpp"


void rotate_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto pixels = img::create_buffer32(width * height);

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);

    auto radians = 0.6f * 2 * 3.14159f;
    Point2Du32 center = { width / 2, height / 2 };

    img::rotate(src, out, center, radians);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
}


void rotate_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;

    auto buffer = img::create_buffer8(width * height * 2);

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    auto radians = 0.9f * 3.14159f;
    Point2Du32 center = { width / 2, height / 2 };

    img::rotate(src, dst, center, radians);

    img::map_gray(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}