#include "tests_include.hpp"


void copy_image_test(img::View const& out_view)
{
	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	auto r = make_range(vette);
	if (vette.width > out_view.width)
	{
		r.x_end = out_view.width;
	}

	if (vette.height > out_view.height)
	{
		r.y_end = out_view.height;
	}

	img::copy(img::sub_view(vette, r), img::sub_view(out_view, r));

	img::destroy_image(vette);
}


void resize_image_test(img::View const& out_view)
{
	Image vette;
	img::read_image_from_file(CORVETTE_PATH, vette);

	Image image;
	image.width = out_view.width;
	image.height = out_view.height;

	img::resize_image(vette, image);

	img::copy(img::make_view(image), out_view);

	img::destroy_image(vette);
	img::destroy_image(image);
}