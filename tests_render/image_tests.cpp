#include "tests_include.hpp"


void copy_image_test(img::View const& out)
{
	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	auto r = make_range(vette);
	if (vette.width > out.width)
	{
		r.x_end = out.width;
	}

	if (vette.height > out.height)
	{
		r.y_end = out.height;
	}

	img::copy(img::sub_view(vette, r), img::sub_view(out, r));

	img::destroy_image(vette);
}


void resize_image_test(img::View const& out)
{
	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	Image image;
	image.width = out.width;
	image.height = out.height;

	img::resize_image(vette, image);

	img::copy(img::make_view(image), out);

	img::destroy_image(vette);
	img::destroy_image(image);
}