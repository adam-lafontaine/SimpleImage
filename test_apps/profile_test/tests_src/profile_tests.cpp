#include "../../tests_include.hpp"
#include "../../../simage/src/util/profiler.cpp"


constexpr u32 WIDTH = 2000;
constexpr u32 HEIGHT = 2000;

constexpr u32 N_TEST_RUNS = 5;


typedef void (*TestFunc)();


static void run_test(TestFunc test, cstr label)
{
    perf::profile_clear();

    for (u32 i = 0; i < N_TEST_RUNS; ++i)
    {
        test();
    }

    perf::profile_report(label);
}


static void create_destroy_image()
{
    img::Image rgba;
    img::ImageGray gray;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto res = PROFILE(img::create_image(rgba, width, height));
    res = PROFILE(img::create_image(gray, width, height));
    PROFILE(img::destroy_image(rgba));
    PROFILE(img::destroy_image(gray));
}


static void create_destroy_buffer()
{   
    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = PROFILE(img::create_buffer32(width * height));
    auto buffer8 = PROFILE(img::create_buffer8(width * height));
    PROFILE(img::destroy_buffer(buffer32));
    PROFILE(img::destroy_buffer(buffer8));
}


static void read_image()
{
    img::Image rgba;
    img::ImageGray gray;

    auto path = CORVETTE_PATH;

    auto res = PROFILE(img::read_image_from_file(path, rgba));
    res = PROFILE(img::read_image_from_file(path, gray));

    img::destroy_image(rgba);
    img::destroy_image(gray);
}

static void resize_image()
{
    img::Image rgba_src;    
    img::ImageGray gray_src;    

    auto path = CORVETTE_PATH;

    auto res = img::read_image_from_file(path, rgba_src);
    res = img::read_image_from_file(path, gray_src);

    img::Image rgba_dst;
    rgba_dst.width = WIDTH;
    rgba_dst.height = HEIGHT;

    img::ImageGray gray_dst;
    gray_dst.width = WIDTH;
    gray_dst.height = HEIGHT;

    res = PROFILE(img::resize_image(rgba_src, rgba_dst));
    res = PROFILE(img::resize_image(gray_src, gray_dst));

    img::destroy_image(rgba_src);
    img::destroy_image(rgba_dst);
    img::destroy_image(gray_src);
    img::destroy_image(gray_dst);
}


static void make_view()
{
    img::Image rgba;
    img::ImageGray gray;

    auto path = CORVETTE_PATH;
    auto n_channels32 = 11;
    auto n_channels8 = 1;

    auto res = img::read_image_from_file(path, rgba);
    res = img::read_image_from_file(path, gray);

    auto v1 = PROFILE(img::make_view(rgba));
    auto v2 = PROFILE(img::make_view(gray));

    auto width = rgba.width;
    auto height = rgba.height;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto v3 = PROFILE(img::make_view(width, height, buffer32));
    auto v4 = PROFILE(img::make_view(width, height, buffer8));
    auto v5 = PROFILE(img::make_view_1(width, height, buffer32));
    auto v6 = PROFILE(img::make_view_2(width, height, buffer32));
    auto v7 = PROFILE(img::make_view_3(width, height, buffer32));
    auto v8 = PROFILE(img::make_view_4(width, height, buffer32));

    img::destroy_image(rgba);
    img::destroy_image(gray);
    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void sub_view()
{
    img::Image rgba;
    img::ImageGray gray;

    auto path = CORVETTE_PATH;
    auto n_channels32 = 11;
    auto n_channels8 = 1;

    auto res = img::read_image_from_file(path, rgba);
    res = img::read_image_from_file(path, gray);

    auto width = rgba.width;
    auto height = rgba.height;

    auto view_rgba = img::make_view(rgba);
    auto view_gray = img::make_view(gray);

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_1 = img::make_view_1(width, height, buffer32);
    auto view_2 = img::make_view_2(width, height, buffer32);
    auto view_3 = img::make_view_3(width, height, buffer32);
    auto view_4 = img::make_view_4(width, height, buffer32);

    auto range = make_range(width / 2, height / 2);

    auto v1 = PROFILE(img::sub_view(rgba, range));
    auto v2 = PROFILE(img::sub_view(gray, range));
    auto v3 = PROFILE(img::sub_view(view_rgba, range));
    auto v4 = PROFILE(img::sub_view(view_gray, range));
    auto v5 = PROFILE(img::sub_view(view_1, range));
    auto v6 = PROFILE(img::sub_view(view_2, range));
    auto v7 = PROFILE(img::sub_view(view_3, range));
    auto v8 = PROFILE(img::sub_view(view_4, range));

    img::destroy_image(rgba);
    img::destroy_image(gray);
    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void map_gray()
{
    img::Image rgba;

    auto path = CORVETTE_PATH;
    auto n_channels32 = 4;
    auto n_channels8 = 1;

    auto res = img::read_image_from_file(path, rgba);
    auto width = rgba.width;
    auto height = rgba.height;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_rgba = img::make_view(rgba);
    auto view_gray = img::make_view(width, height, buffer8);
    auto view_1 = img::make_view_1(width, height, buffer32);
    auto view_3 = img::make_view_3(width, height, buffer32);

    PROFILE(img::map_gray(view_rgba, view_gray));
    PROFILE(img::map_gray(view_rgba, view_1));
    PROFILE(img::map_gray(view_3, view_1));

    img::destroy_image(rgba);
    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}





void run_profile_tests()
{
    run_test(create_destroy_image, "create_destroy_image");
    run_test(create_destroy_buffer, "create_destroy_buffer");
    run_test(read_image, "read_image");
    run_test(resize_image, "resize_image");
    run_test(make_view, "make_view");
    run_test(sub_view, "sub_view");
    run_test(map_gray, "map_gray");
    


    
}