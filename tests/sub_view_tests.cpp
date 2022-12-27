#include "tests_include.hpp"


template <typename T>
static bool is_valid_ptr(T* ptr)
{
    return static_cast<bool>(ptr);
}


static bool image_test()
{
    auto title = "sub_view_image_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    bool result = false;

    Image vette;    
    Image left_dst;
    Image bottom_dst;
    img::Buffer32 buffer;

    auto const cleanup = [&]()
    {
        img::destroy_image(vette);
        img::destroy_image(left_dst);
        img::destroy_image(bottom_dst);
        mb::destroy_buffer(buffer);
    };
    
    img::read_image_from_file(CORVETTE_PATH, vette);
    auto width = vette.width;
    auto height = vette.height;

    write_image(vette, "image.bmp");

    Range2Du32 left;
    left.x_begin = 0;
    left.x_end = width / 2;
    left.y_begin = 0;
    left.y_end = height;

    printf("sub_view from image\n");
    auto left_view = img::sub_view(vette, left);

    result = is_valid_ptr(left_view.image_data);
    result &= (left_view.image_data == vette.data_);
    result &= (left_view.width == width / 2);
    result &= (left_view.height == height);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    auto bottom = left;
    bottom.y_begin = height / 2;

    printf("sub_view from view\n");
    auto bottom_view = img::sub_view(left_view, bottom);

    result = is_valid_ptr(bottom_view.image_data);
    result &= (bottom_view.image_data == vette.data_);
    result &= (bottom_view.width == width / 2);
    result &= (bottom_view.height == height - bottom.y_begin);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    // silly copy
    
    img::create_image(left_dst, left_view.width, left_view.height);    
    img::create_image(bottom_dst, bottom_view.width, bottom_view.height);
    
    mb::create_buffer(buffer, 2 * width * height);    
    auto view = img::make_view_3(left_view.width, left_view.height, buffer);
    img::map_rgb(left_view, view);
    img::map_rgb(view, img::make_view(left_dst));

    mb::reset_buffer(buffer);

    view = img::make_view_3(bottom_view.width, bottom_view.height, buffer);
    img::map_rgb(bottom_view, view);
    img::map_rgb(view, img::make_view(bottom_dst));

    write_image(left_dst, "left.bmp");
    write_image(bottom_dst, "bottom.bmp");

    cleanup();

    return true;
}


static bool gray_test()
{
    auto title = "sub_view_gray_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) { img::write_image(image, out_dir / name); };

    bool result = false;

    GrayImage caddy;
    GrayImage bottom_dst;
    GrayImage left_dst;
    img::Buffer32 buffer;

    auto const cleanup = [&]()
    {
        img::destroy_image(caddy);
        img::destroy_image(left_dst);
        img::destroy_image(bottom_dst);
        mb::destroy_buffer(buffer);
    };

    img::read_image_from_file(CADILLAC_PATH, caddy);
    auto width = caddy.width;
    auto height = caddy.height;

    write_image(caddy, "gray.bmp");

    Range2Du32 left;
    left.x_begin = 0;
    left.x_end = width / 2;
    left.y_begin = 0;
    left.y_end = height;

    printf("sub_view from image\n");
    auto left_view = img::sub_view(caddy, left);

    result = is_valid_ptr(left_view.image_data);
    result &= (left_view.image_data == caddy.data_);
    result &= (left_view.width == width / 2);
    result &= (left_view.height == height);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    auto bottom = left;
    bottom.y_begin = height / 2;

    printf("sub_view from view\n");
    auto bottom_view = img::sub_view(left_view, bottom);

    result = is_valid_ptr(bottom_view.image_data);
    result &= (bottom_view.image_data == caddy.data_);
    result &= (bottom_view.width == width / 2);
    result &= (bottom_view.height == height - bottom.y_begin);
    if (!result)
    {
        printf("FAIL\n");
        cleanup();
        return false;
    }
    printf("OK\n");

    // silly copy    
    img::create_image(left_dst, left_view.width, left_view.height);    
    img::create_image(bottom_dst, bottom_view.width, bottom_view.height);
    
    mb::create_buffer(buffer, width * height);
    
    auto view = img::make_view_1(left_view.width, left_view.height, buffer);
    img::map(left_view, view);
    img::map(view, img::make_view(left_dst));

    mb::reset_buffer(buffer);

    view = img::make_view_1(bottom_view.width, bottom_view.height, buffer);
    img::map(bottom_view, view);
    img::map(view, img::make_view(bottom_dst));

    write_image(left_dst, "left.bmp");
    write_image(bottom_dst, "bottom.bmp");

    cleanup();

    return true;
}


bool sub_view_tests()
{
    printf("\n*** sub_view tests ***\n");

    auto result = 
        image_test() &&
        gray_test();

    if (result)
    {
        printf("sub_view tests OK\n");
    }
    
    return result;
}