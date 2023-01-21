#include "tests_include.hpp"


bool gradients_xy_test()
{
	auto title = "gradients_xy_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name)
	{ img::write_image(image, out_dir / name); };

	GrayImage image;
	img::read_image_from_file(CORVETTE_PATH, image);
	auto view = img::make_view(image);
	auto width = view.width;
	auto height = view.height;

	img::Buffer32 buffer;
	mb::create_buffer(buffer, width * height * 3);

	auto src = img::make_view_1(width, height, buffer);
	auto dst_xy = img::make_view_2(width, height, buffer);

	img::map(view, src);

	img::gradients_xy(src, dst_xy);

	auto dst_x = img::select_channel(dst_xy, img::XY::X);
	auto dst_y = img::select_channel(dst_xy, img::XY::Y);

	write_image(image, "vette.bmp");

	auto const to_abs = [](r32 p) { return p < 0.0f ? -p : p; };

	img::transform(dst_x, src, to_abs);
	img::map(src, view);
	write_image(image, "x_grad.bmp");

	img::transform(dst_y, src, to_abs);
	img::map(src, view);
	write_image(image, "y_grad.bmp");

	mb::destroy_buffer(buffer);
	img::destroy_image(image);

	return true;
}


bool gradients_tests()
{
	printf("\n*** gradients tests ***\n");

	auto result =
		gradients_xy_test();

	if (result)
	{
		printf("gradients tests OK\n");
	}
	return result;
}