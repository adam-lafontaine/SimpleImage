#include "tests_include.hpp"


void transform_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image caddy;
    img::Image caddy2;

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, caddy, caddy2, width, height);

    auto const invert = [](img::Pixel p)
    {
        p.rgba.red = 255 - p.rgba.red;
        p.rgba.green = 255 - p.rgba.green;
        p.rgba.blue = 255 - p.rgba.blue;

        return p;
    };

    img::transform(src, out, invert);

    img::destroy_image(caddy);
    img::destroy_image(caddy2);
}


void transform_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray caddy;
    img::ImageGray caddy2;

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, caddy, caddy2, width, height);

    auto const invert = [](u8 p) { return 255 - p; };

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view(width, height, buffer);

    img::transform(src, dst, invert);

    img::map_gray(dst, out);

    img::destroy_image(caddy);
    img::destroy_image(caddy2);
    mb::destroy_buffer(buffer);
}


void threshold_min_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray vette;
    img::ImageGray vette2;
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, vette, vette2, width, height);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view(width, height, buffer);

    img::threshold(src, dst, 75);

    img::map_gray(dst, out);

    img::destroy_image(vette);
    img::destroy_image(vette2);
    mb::destroy_buffer(buffer);
}


void threshold_min_max_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray vette;
    img::ImageGray vette2;
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, vette, vette2, width, height);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view(width, height, buffer);

    img::threshold(src, dst, 30, 200);

    img::map_gray(dst, out);

    img::destroy_image(vette);
    img::destroy_image(vette2);
    mb::destroy_buffer(buffer);
}


void binarize_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray weed;
    img::ImageGray weed2;
    img::read_image_from_file(WEED_PATH, weed);

    auto src = img::make_view_resized_from_file(WEED_PATH, weed, weed2, width, height);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view(width, height, buffer);

    img::binarize(src, dst, [](u8 p){ return p < 150; });

    img::map_gray(dst, out);

    img::destroy_image(weed);
    img::destroy_image(weed2);
    mb::destroy_buffer(buffer);
}


void binarize_rgb_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image weed;
    img::Image weed2;
    img::read_image_from_file(WEED_PATH, weed);

    auto src = img::make_view_resized_from_file(WEED_PATH, weed, weed2, width, height);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view(width, height, buffer);

    img::binarize(src, dst, [](img::Pixel p){ return p.rgba.red > 200; });

    img::map_gray(dst, out);

    img::destroy_image(weed);
    img::destroy_image(weed2);
    mb::destroy_buffer(buffer);
}