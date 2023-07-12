#include "../../tests_include.hpp"


void copy_test(img::View const& out)
{
	auto width = out.width;
    auto height = out.height;

    img::Image image;

    auto pixels = img::create_buffer32(width * height);

    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, pixels);

	img::copy(src, out);

	img::destroy_image(image);
}


void copy_gray_test(img::View const& out)
{
	auto width = out.width;
    auto height = out.height;

    img::ImageGray image;

    auto buffer = img::create_buffer8(width * height * 2);

    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

	img::copy(src, dst);

	img::map_gray(dst, out);

	img::destroy_image(image);
	img::destroy_buffer(buffer);
}