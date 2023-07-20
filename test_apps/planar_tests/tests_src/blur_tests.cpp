#include "../../tests_include.hpp"


void blur_gray_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);
    auto buffer32 = img::create_buffer32(width * height * 2);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer8);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(view, src);

    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}


void blur_rgb_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer = img::create_buffer32(width * height * 7);

    img::Image image;
    auto view = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer);

    auto src = img::make_view_3(width, height, buffer);
    auto dst = img::make_view_3(width, height, buffer);

    img::map_rgb(view, src);

    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);
    img::blur(dst, src);
    img::blur(src, dst);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer);
    img::destroy_image(image);
}