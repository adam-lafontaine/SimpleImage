#include "tests_include.hpp"


void transform_test(img::View const& out)
{
    img::Image caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);

    auto const invert = [](img::Pixel p)
    {
        p.rgba.red = 255 - p.rgba.red;
        p.rgba.green = 255 - p.rgba.green;
        p.rgba.blue = 255 - p.rgba.blue;

        return p;
    };

    auto width = out.width;
    auto height = out.height;

    img::Image image;
    image.width = width;
    image.height = height;
    img::resize_image(caddy, image);

    img::transform(img::make_view(image), out, invert);

    img::destroy_image(caddy);
    img::destroy_image(image);
}


void transform_gray_test(img::View const& out)
{
    img::ImageGray caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);

    auto const invert = [](u8 p) { return 255 - p; };

    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    image.width = width;
    image.height = height;
    img::resize_image(caddy, image);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view_gray(width, height, buffer);

    img::transform(img::make_view(image), dst, invert);

    img::map_gray(dst, out);

    img::destroy_image(caddy);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void threshold_min_test(img::View const& out)
{
    img::ImageGray vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    image.width = width;
    image.height = height;
    img::resize_image(vette, image);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view_gray(width, height, buffer);

    img::threshold(img::make_view(image), dst, 75);

    img::map_gray(dst, out);

    img::destroy_image(vette);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void threshold_min_max_test(img::View const& out)
{
    img::ImageGray vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    image.width = width;
    image.height = height;
    img::resize_image(vette, image);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view_gray(width, height, buffer);

    img::threshold(img::make_view(image), dst, 30, 200);

    img::map_gray(dst, out);

    img::destroy_image(vette);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void binarize_test(img::View const& out)
{
    img::ImageGray weed;
    img::read_image_from_file(WEED_PATH, weed);

    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    image.width = width;
    image.height = height;
    img::resize_image(weed, image);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view_gray(width, height, buffer);

    img::binarize(img::make_view(image), dst, [](u8 p){ return p < 150; });

    img::map_gray(dst, out);

    img::destroy_image(weed);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void binarize_rgb_test(img::View const& out)
{
    img::Image weed;
    img::read_image_from_file(WEED_PATH, weed);

    auto width = out.width;
    auto height = out.height;

    img::Image image;
    image.width = width;
    image.height = height;
    img::resize_image(weed, image);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto dst = img::make_view_gray(width, height, buffer);

    img::binarize(img::make_view(image), dst, [](img::Pixel p){ return p.rgba.red > 200; });

    img::map_gray(dst, out);

    img::destroy_image(weed);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}