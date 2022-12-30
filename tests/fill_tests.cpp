#include "tests_include.hpp"

bool fill_platform_view_test()
{
    auto title = "fill_platform_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    bool result = false;

    u32 width = 799;
	u32 height = 800;

    GrayImage gray;
    img::create_image(gray, width, height);
    
    img::fill(img::make_view(gray), 128);
    write_image(gray, "all_gray.bmp");
    
    Image image;
    img::create_image(image, width, height);
    auto view = img::make_view(image);

    img::fill(view, img::to_pixel(255, 0, 0));
    write_image(image, "all_red.bmp");

    img::fill(view, img::to_pixel(255, 255, 255, 128));
    write_image(image, "all_white_with_alpha.bmp");

    img::destroy_image(gray);
    img::destroy_image(image);

    return true;
}


bool fill_platform_sub_view_test()
{
    auto title = "fill_platform_sub_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    bool result = false;

    u32 width = 800;
	u32 height = 800;

    auto left = make_range(width / 2, height);
    auto right = make_range(width, height);
    right.x_begin = width / 2;

    GrayImage gray;
    img::create_image(gray, width, height);
    img::fill(img::sub_view(gray, left), 0);
    img::fill(img::sub_view(gray, right), 255);
    write_image(gray, "black_white.bmp");

    Image image;
    img::create_image(image, width, height);
    img::fill(img::sub_view(image, left), img::to_pixel(255, 0, 0));
    img::fill(img::sub_view(image, right), img::to_pixel(0, 0, 255));
    write_image(image, "red_blue.bmp");

    img::destroy_image(gray);
    img::destroy_image(image);

    return true;
}


bool fill_view_test()
{
    auto title = "fill_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    bool result = false;

    u32 width = 800;
	u32 height = 800;

    GrayImage gray;
    img::create_image(gray, width, height);

    Image image;
    img::create_image(image, width, height);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 4);

    auto view1 = img::make_view_1(width, height, buffer);
    img::fill(view1, 128);
    img::map(view1, img::make_view(gray));
    write_image(gray, "all_gray.bmp");
    mb::reset_buffer(buffer);

    auto view3 = img::make_view_3(width, height, buffer);
    img::fill(view3, img::to_pixel(255, 0, 0));
    img::map_rgb(view3, img::make_view(image));
    write_image(image, "all_red.bmp");
    mb::reset_buffer(buffer);

    auto view4 = img::make_view_4(width, height, buffer);
    img::fill(view4, img::to_pixel(255, 255, 255, 128));
    img::map_rgb(view4, img::make_view(image));
    write_image(image, "all_white_with_alpha.bmp");

    img::destroy_image(gray);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);

    return true;
}


bool fill_sub_view_test()
{
    auto title = "fill_sub_view_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    bool result = false;

    u32 width = 800;
	u32 height = 800;

    auto left = make_range(width / 2, height);
    auto right = make_range(width, height);
    right.x_begin = width / 2;

    GrayImage gray;
    img::create_image(gray, width, height);

    Image image;
    img::create_image(image, width, height);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height * 3);

    auto view1 = img::make_view_1(width, height, buffer);
    img::fill(img::sub_view(view1, left), 0);
    img::fill(img::sub_view(view1, right), 255);
    img::map(view1, img::make_view(gray));
    write_image(gray, "black_white.bmp");
    mb::reset_buffer(buffer);

    auto view3 = img::make_view_3(width, height, buffer);
    img::fill(img::sub_view(view3, left), img::to_pixel(255, 0, 0));
    img::fill(img::sub_view(view3, right), img::to_pixel(0, 0, 255));
    img::map_rgb(view3, img::make_view(image));
    write_image(image, "red_blue.bmp");

    img::destroy_image(gray);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);

    return true;
}


bool fill_tests()
{
    printf("\n*** fill tests ***\n");

    auto result = 
        fill_platform_view_test() &&
        fill_platform_sub_view_test() &&
        fill_view_test() &&
        fill_sub_view_test();

    if (result)
    {
        printf("fill tests OK\n");
    }
    return result;
}