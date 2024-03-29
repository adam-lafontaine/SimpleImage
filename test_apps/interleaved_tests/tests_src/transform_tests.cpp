#include "../../tests_include.hpp"


void transform_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto pixels = img::create_buffer32(width * height);

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, pixels);

    auto const invert = [](img::Pixel p)
    {
        p.rgba.red = 255 - p.rgba.red;
        p.rgba.green = 255 - p.rgba.green;
        p.rgba.blue = 255 - p.rgba.blue;

        return p;
    };

    img::transform(src, out, invert);

    img::destroy_image(image);
    img::destroy_buffer(pixels);
}


void transform_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto buffer = img::create_buffer8(width * height * 2);

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    auto const invert = [](u8 p) { return 255 - p; };    

    img::transform(src, dst, invert);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void threshold_min_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto buffer = img::create_buffer8(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::threshold(src, dst, 75);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void threshold_min_max_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;
    
    auto buffer = img::create_buffer8(width * height * 2);
	
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::threshold(src, dst, 30, 200);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void binarize_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray weed;

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 2);    

    auto src = img::make_view_resized_from_file(WEED_PATH, weed, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::binarize(src, dst, [](u8 p){ return p < 150; });

    img::map_rgba(dst, out);

    img::destroy_image(weed);
    img::destroy_buffer(buffer);
}


void binarize_rgb_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Image weed;

    img::Buffer32 pixels;
    mb::create_buffer(pixels, width * height);

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height);

    auto src = img::make_view_resized_from_file(WEED_PATH, weed, width, height, pixels);
    auto dst = img::make_view(width, height, buffer);

    img::binarize(src, dst, [](img::Pixel p){ return p.rgba.red > 200; });

    img::map_rgba(dst, out);

    img::destroy_image(weed);
    img::destroy_buffer(pixels);
    img::destroy_buffer(buffer);
}