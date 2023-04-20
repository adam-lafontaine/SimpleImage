#include "../../tests_include.hpp"


void resize_image_test(img::View const& out)
{
	img::Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	img::Image image;
	image.width = out.width;
	image.height = out.height;

	img::resize_image(vette, image);

	img::copy(img::make_view(image), out);

	img::destroy_image(vette);
	img::destroy_image(image);
}


void resize_gray_image_test(img::View const& out)
{
	img::ImageGray vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	img::ImageGray image;
	image.width = out.width;
	image.height = out.height;

	img::resize_image(vette, image);

	img::map_gray(img::make_view(image), out);

	img::destroy_image(vette);
	img::destroy_image(image);
}