#include "../../tests_include.hpp"


void transform_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer8);

    auto buffer32 = img::create_buffer32(width * height * 2);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(view, src);

    auto const invert = [](f32 p) { return 1.0f - p; };

    img::transform(src, dst, invert);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}


void threshold_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer8);

    auto buffer32 = img::create_buffer32(width * height * 2);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(view, src);

    img::threshold(src, dst, 0.4f);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}


void binarize_test(img::View const& out)
{
    auto const width = out.width;
    auto const height = out.height;

    auto buffer8 = img::create_buffer8(width * height);

    img::ImageGray image;
    auto view = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer8);

    auto buffer32 = img::create_buffer32(width * height * 2);

    auto src = img::make_view_1(width, height, buffer32);
    auto dst = img::make_view_1(width, height, buffer32);

    img::map_gray(view, src);

    auto const pred = [](f32 p) { return p >= 0.4f; };

    img::binarize(src, dst, pred);

    img::map_rgba(dst, out);

    img::destroy_buffer(buffer8);
    img::destroy_buffer(buffer32);
    img::destroy_image(image);
}