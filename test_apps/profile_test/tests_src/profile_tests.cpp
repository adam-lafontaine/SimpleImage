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


static void copy()
{
    auto n_channels32 = 2;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_rgba_s = img::make_view(width, height, buffer32);
    auto view_rgba_d = img::make_view(width, height, buffer32);
    auto view_gray_s = img::make_view(width, height, buffer8);
    auto view_gray_d = img::make_view(width, height, buffer8);

    PROFILE(img::copy(view_rgba_s, view_rgba_d));
    PROFILE(img::copy(view_gray_s, view_gray_d));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void fill()
{
    auto n_channels32 = 2;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_rgba_s = img::make_view(width, height, buffer32);
    auto view_rgba_d = img::make_view(width, height, buffer32);
    auto view_gray_s = img::make_view(width, height, buffer8);
    auto view_gray_d = img::make_view(width, height, buffer8);
}


static void map_gray_gray()
{
    auto n_channels32 = 5;
    auto n_channels8 = 1;
    
    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_rgba = img::make_view(width, height, buffer32);
    auto view_gray = img::make_view(width, height, buffer8);
    auto view_1 = img::make_view_1(width, height, buffer32);
    auto view_3 = img::make_view_3(width, height, buffer32);

    PROFILE(img::map_gray(view_gray, view_1));
    PROFILE(img::map_gray(view_1, view_gray));
    
    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void map_gray_rgba()
{
    auto n_channels32 = 2;
    auto n_channels8 = 1;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto view_rgba = img::make_view(width, height, buffer32);
    auto view_gray = img::make_view(width, height, buffer8);
    auto view_1 = img::make_view_1(width, height, buffer32);

    PROFILE(img::map_rgba(view_gray, view_rgba));
    PROFILE(img::map_rgba(view_1, view_rgba));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void map_rgba()
{
    auto n_channels32 = 8;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    auto view_rgba = img::make_view(width, height, buffer32);
    auto view_3 = img::make_view_3(width, height, buffer32);
    auto view_4 = img::make_view_4(width, height, buffer32);

    PROFILE(img::map_rgb(view_rgba, view_3));
    PROFILE(img::map_rgba(view_3, view_rgba));
    PROFILE(img::map_rgba(view_rgba, view_4));
    PROFILE(img::map_rgba(view_4, view_rgba));

    img::destroy_buffer(buffer32);
}


static void alpha_blend()
{
    auto n_channels32 = 13;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto cur_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_4 = img::make_view_4(width, height, buffer32);
    auto cur_3 = img::make_view_3(width, height, buffer32);
    auto dst_3 = img::make_view_3(width, height, buffer32);

    PROFILE(img::alpha_blend(src_rgba, cur_rgba, dst_rgba));
    PROFILE(img::alpha_blend(src_4, cur_3, dst_3));

    img::destroy_buffer(buffer32);
}


static void rotate()
{
    auto n_channels32 = 22;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);

    auto src_4 = img::make_view_4(width, height, buffer32);
    auto dst_4 = img::make_view_4(width, height, buffer32);

    auto src_3 = img::make_view_3(width, height, buffer32);
    auto dst_3 = img::make_view_3(width, height, buffer32);

    auto src_2 = img::make_view_2(width, height, buffer32);
    auto dst_2 = img::make_view_2(width, height, buffer32);

    auto src_1 = img::make_view_1(width, height, buffer32);
    auto dst_1 = img::make_view_1(width, height, buffer32);

    Point2Du32 origin = { width / 2, height / 2 };
    f32 angle_rad = 0.3f * 2 * 3.14159f;

    PROFILE(img::rotate(src_rgba, dst_rgba, origin, angle_rad));
    PROFILE(img::rotate(src_gray, dst_gray, origin, angle_rad));
    PROFILE(img::rotate(src_4, dst_4, origin, angle_rad));
    PROFILE(img::rotate(src_3, dst_3, origin, angle_rad));
    PROFILE(img::rotate(src_2, dst_2, origin, angle_rad));
    PROFILE(img::rotate(src_1, dst_1, origin, angle_rad));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void blur()
{
    auto n_channels32 = 10;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);

    auto src_3 = img::make_view_3(width, height, buffer32);
    auto dst_3 = img::make_view_3(width, height, buffer32);

    auto src_1 = img::make_view_1(width, height, buffer32);
    auto dst_1 = img::make_view_1(width, height, buffer32);

    PROFILE(img::blur(src_rgba, dst_rgba));
    PROFILE(img::blur(src_gray, dst_gray));
    PROFILE(img::blur(src_3, dst_3));
    PROFILE(img::blur(src_1, dst_1));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void gradients()
{
    auto n_channels32 = 4;
    auto n_channels8 = 4;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);
    auto dst_gray_x = img::make_view(width, height, buffer8);
    auto dst_gray_y = img::make_view(width, height, buffer8);

    auto src_1 = img::make_view_1(width, height, buffer32);
    auto dst_1 = img::make_view_1(width, height, buffer32);
    auto dst_2_xy = img::make_view_2(width, height, buffer32);

    PROFILE(img::gradients(src_gray, dst_gray));
    PROFILE(img::gradients_xy(src_gray, dst_gray_x, dst_gray_y));
    PROFILE(img::gradients(src_1, dst_1));
    PROFILE(img::gradients_xy(src_1, dst_2_xy));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void map_gray_interleaved(img::View const& src, img::ViewGray const& dst)
{
    img::map_gray(src, dst);
}


static void map_gray_planar(img::View const& src, img::ViewGray const& dst, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;
    
    auto d = img::make_view_1(w, h, buffer);

    img::map_gray(src, d);

    img::map_gray(d, dst);
}


static void alpha_blend_interleaved(img::View const& src, img::View const& cur, img::View const& dst)
{
    img::alpha_blend(src, cur, dst);
}


static void alpha_blend_planar(img::View const& src, img::View const& cur, img::View const& dst, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_4(w, h, buffer);
    auto c = img::make_view_3(w, h, buffer);
    auto d = img::make_view_3(w, h, buffer);

    img::map_rgba(src, s);
    img::map_rgb(cur, c);
    
    img::alpha_blend(s, c, d);

    img::map_rgba(d, dst);
}


static void rotate_rgb_interleaved(img::View const& src, img::View const& dst, Point2Du32 origin, f32 rad)
{
    img::rotate(src, dst, origin, rad);
}


static void rotate_gray_interleaved(img::ViewGray const& src, img::ViewGray const& dst, Point2Du32 origin, f32 rad)
{
    img::rotate(src, dst, origin, rad);
}


static void rotate_rgb_planar(img::View const& src, img::View const& dst, Point2Du32 origin, f32 rad, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_3(w, h, buffer);
    auto d = img::make_view_3(w, h, buffer);

    img::map_rgb(src, s);

    img::rotate(s, d, origin, rad);

    img::map_rgba(d, dst);
}


static void rotate_gray_planar(img::ViewGray const& src, img::ViewGray const& dst, Point2Du32 origin, f32 rad, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_1(w, h, buffer);
    auto d = img::make_view_1(w, h, buffer);

    img::map_gray(src, s);

    img::rotate(s, d, origin, rad);

    img::map_gray(d, dst);
}


static void blur_gray_interleaved(img::ViewGray const& src, img::ViewGray const& dst)
{
    img::blur(src, dst);
}


static void blur_rgb_interleaved(img::View const& src, img::View const& dst)
{
    img::blur(src, dst);
}


static void blur_gray_planar(img::ViewGray const& src, img::ViewGray const& dst, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_1(w, h, buffer);
    auto d = img::make_view_1(w, h, buffer);

    img::map_gray(src, s);

    img::blur(s, d);

    img::map_gray(d, dst);
}


static void blur_rgb_planar(img::View const& src, img::View const& dst, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_3(w, h, buffer);
    auto d = img::make_view_3(w, h, buffer);

    img::map_rgb(src, s);

    img::blur(s, d);

    img::map_rgba(d, dst);
}


static void gradients_u8(img::ViewGray const& src, img::ViewGray const& dst)
{
    img::gradients(src, dst);
}


static void gradients_xy_u8(img::ViewGray const& src, img::ViewGray const& dst_x, img::ViewGray const& dst_y)
{
    img::gradients_xy(src, dst_x, dst_y);
}


static void gradients_f32(img::ViewGray const& src, img::ViewGray const& dst, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_1(w, h, buffer);
    auto d = img::make_view_1(w, h, buffer);

    img::map_gray(src, s);

    img::gradients(s, d);

    
}


static void gradients_xy_f32(img::ViewGray const& src, img::ViewGray const& dst_x, img::ViewGray const& dst_y, img::Buffer32& buffer)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_view_1(w, h, buffer);
    auto d_xy = img::make_view_2(w, h, buffer);

    img::map_gray(src, s);

    img::gradients_xy(s, d_xy);

    img::map_gray(img::select_channel(d_xy, img::XY::X), dst_x);
    img::map_gray(img::select_channel(d_xy, img::XY::Y), dst_y);
}


static void compare_map_gray()
{
    auto n_channels32 = 2;
    auto n_channels8 = 1;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src = img::make_view(width, height, buffer32);
    auto dst = img::make_view(width, height, buffer8);

    PROFILE(map_gray_interleaved(src, dst));
    PROFILE(map_gray_planar(src, dst, buffer32));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_alpha_blend()
{
    auto n_channels32 = 13;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);

    auto src = img::make_view(width, height, buffer32);
    auto cur = img::make_view(width, height, buffer32);
    auto dst = img::make_view(width, height, buffer32);

    PROFILE(alpha_blend_interleaved(src, cur, dst));
    PROFILE(alpha_blend_planar(src, cur, dst, buffer32));

    img::destroy_buffer(buffer32);
}


static void compare_rotate()
{
    auto n_channels32 = 10;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);

    Point2Du32 origin = { width / 2, height / 2 };
    f32 angle_rad = 0.3f * 2 * 3.14159f;

    PROFILE(rotate_rgb_interleaved(src_rgba, dst_rgba, origin, angle_rad));
    PROFILE(rotate_gray_interleaved(src_gray, dst_gray, origin, angle_rad));
    PROFILE(rotate_rgb_planar(src_rgba, dst_rgba, origin, angle_rad, buffer32));
    PROFILE(rotate_gray_planar(src_gray, dst_gray, origin, angle_rad, buffer32));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_blur()
{
    auto n_channels32 = 10;
    auto n_channels8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);

    PROFILE(blur_rgb_interleaved(src_rgba, dst_rgba));
    PROFILE(blur_gray_interleaved(src_gray, dst_gray));
    PROFILE(blur_rgb_planar(src_rgba, dst_rgba, buffer32));
    PROFILE(blur_gray_planar(src_gray, dst_gray, buffer32));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_gradients()
{
    auto n_channels32 = 5;
    auto n_channels8 = 4;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);
    auto dst_x = img::make_view(width, height, buffer8);
    auto dst_y = img::make_view(width, height, buffer8);

    PROFILE(gradients_u8(src_gray, dst_gray));
    PROFILE(gradients_xy_u8(src_gray, dst_x, dst_y));
    PROFILE(gradients_f32(src_gray, dst_gray, buffer32));
    PROFILE(gradients_xy_f32(src_gray, dst_x, dst_y, buffer32));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


void run_profile_tests()
{
    //run_test(create_destroy_image, "create_destroy_image");
    //run_test(create_destroy_buffer, "create_destroy_buffer");
    //run_test(read_image, "read_image");
    //run_test(resize_image, "resize_image");
    //run_test(make_view, "make_view");
    //run_test(sub_view, "sub_view");
    run_test(copy, "copy");
    //run_test(map_gray_gray, "map_gray_gray");
    //run_test(map_gray_rgba, "map_gray_rgba");    
    //run_test(alpha_blend, "alpha_blend");
    //run_test(rotate, "rotate");
    //run_test(blur, "blur");
    //run_test(gradients, "gradients");

    /*run_test(compare_map_gray, "compare_map_gray");
    run_test(compare_alpha_blend, "compare_alpha_blend");
    run_test(compare_rotate, "compare_rotate");
    run_test(compare_blur, "compare_blur");
    run_test(compare_gradients, "compare_gradients");*/
}