#include "tests_include.hpp"


void copy_test(img::View const& out)
{
    img::Image vette;
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