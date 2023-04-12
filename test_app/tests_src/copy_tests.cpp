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


void copy_gray_test(img::View const& out)
{
    img::ImageGray vette;
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

	auto src = img::sub_view(vette, r);
	auto width = src.width;
	auto height = src.height;

	img::Buffer8 buffer;
	mb::create_buffer(buffer, width * height);

	auto dst = img::make_view(width, height, buffer);

	img::copy(src, dst);

	img::map_gray(dst, img::sub_view(out, r));

	img::destroy_image(vette);
	mb::destroy_buffer(buffer);
}