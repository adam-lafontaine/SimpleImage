#include "../../tests_include.hpp"


void rotate_rgb_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    u32 n_channels = 1 + 3 + 3;

    auto buffer = img::create_buffer32(width * height * n_channels);

    img::Image image;
    auto caddy = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);

    auto src = img::make_view_3(width, height, buffer);
    auto dst = img::make_view_3(width, height, buffer);

    img::map_rgb(caddy, src);

    auto radians = 0.6f * 2 * 3.14159f;
    Point2Du32 center = { width / 2, height / 2 };

    img::rotate(src, dst, center, radians);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer);
    img::destroy_image(image);
}


void rotate_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer8 = img::create_buffer8(width * height);
    auto buffer32 = img::create_buffer32(width * height * 2);

    img::ImageGray image;
    auto vette = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer8);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(vette, src);

    auto radians = 0.9f * 3.14159f;
    Point2Du32 center = { width / 2, height / 2 };

    img::rotate(src, dst, center, radians);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}