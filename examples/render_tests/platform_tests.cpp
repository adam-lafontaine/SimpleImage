#include "tests_include.hpp"


void fill_platform_view_test(img::View const& out)
{
    auto full = make_range(out);

    auto top = make_range(out.width, out.height / 3);

    auto mid = full;
    mid.y_begin = top.y_end;
    mid.y_end = mid.y_begin + out.height / 3;

    auto bottom = full;
    bottom.y_begin = mid.y_end;

    auto const red = img::to_pixel(255, 0, 0);
    auto const green = img::to_pixel(0, 255, 0);
    auto const blue = img::to_pixel(0, 0, 255);

    img::fill(img::sub_view(out, top), red);
    img::fill(img::sub_view(out, mid), green);
    img::fill(img::sub_view(out, bottom), blue);
}


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