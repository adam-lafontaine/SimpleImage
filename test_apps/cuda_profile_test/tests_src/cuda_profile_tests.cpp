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


static void create_destroy_device_buffer()
{   
    auto width = WIDTH;
    auto height = HEIGHT;

    auto d_buffer32 = PROFILE(img::create_device_buffer32(width * height));
    auto d_buffer8 = PROFILE(img::create_device_buffer8(width * height));

    auto u_buffer32 = PROFILE(img::create_unified_buffer32(width * height));
    auto u_buffer8 = PROFILE(img::create_unified_buffer8(width * height));

    PROFILE(img::destroy_buffer(d_buffer32));
    PROFILE(img::destroy_buffer(d_buffer8));

    PROFILE(img::destroy_buffer(u_buffer32));
    PROFILE(img::destroy_buffer(u_buffer8));
}


static void make_device_view()
{    
    auto n_channels32 = 1;
    auto n_channels8 = 1;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto d_buffer32 = img::create_device_buffer32(width * height * n_channels32);
    auto d_buffer8 = img::create_device_buffer8(width * height * n_channels8);

    auto v1 = PROFILE(img::make_device_view(width, height, d_buffer32));
    auto v2 = PROFILE(img::make_device_view(width, height, d_buffer8));
    
    img::destroy_buffer(d_buffer32);
    img::destroy_buffer(d_buffer8);
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


static void map_gray_cuda(img::View const& src, img::ViewGray const& dst, img::DeviceBuffer32& db32, img::DeviceBuffer8& db8)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db32);
    auto d = img::make_device_view(w, h, db8);

    img::copy_to_device(src, s);

    img::map_rgb_gray(s, d);

    img::copy_to_host(d, dst);
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


static void alpha_blend_cuda(img::View const& src, img::View const& cur, img::View const& dst, img::DeviceBuffer32& db32)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w ,h, db32);
    auto c = img::make_device_view(w, h, db32);
    auto d = img::make_device_view(w, h, db32);

    img::copy_to_device(src, s);
    img::copy_to_device(cur, c);

    img::alpha_blend(s, c, d);

    img::copy_to_host(d, dst);
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


static void rotate_rgb_cuda(img::View const& src, img::View const& dst, Point2Du32 origin, f32 rad, img::DeviceBuffer32& db32)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db32);
    auto d = img::make_device_view(w, h, db32);

    img::copy_to_device(src, s);

    img::rotate(s, d, origin, rad);

    img::copy_to_host(d, dst);
}


static void rotate_gray_cuda(img::ViewGray const& src, img::ViewGray const& dst, Point2Du32 origin, f32 rad, img::DeviceBuffer8& db8)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db8);
    auto d = img::make_device_view(w, h, db8);

    img::copy_to_device(src, s);

    img::rotate(s, d, origin, rad);

    img::copy_to_host(d, dst);
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


static void blur_gray_cuda(img::ViewGray const& src, img::ViewGray const& dst, img::DeviceBuffer8& db8)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db8);
    auto d = img::make_device_view(w, h, db8);

    img::copy_to_device(src, s);

    img::blur(s, d);

    img::copy_to_host(d, dst);
}


static void blur_rgb_cuda(img::View const& src, img::View const& dst, img::DeviceBuffer32& db32)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db32);
    auto d = img::make_device_view(w, h, db32);

    img::copy_to_device(src, s);

    img::blur(s, d);

    img::copy_to_host(d, dst);
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


static void gradients_cuda(img::ViewGray const& src, img::ViewGray const& dst, img::DeviceBuffer8& db8)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db8);
    auto d = img::make_device_view(w, h, db8);

    img::copy_to_device(src, s);

    img::gradients(s, d);

    img::copy_to_host(d, dst);
}


static void gradients_xy_cuda(img::ViewGray const& src, img::ViewGray const& dst_x, img::ViewGray const& dst_y, img::DeviceBuffer8& db8)
{
    auto w = src.width;
    auto h = src.height;

    auto s = img::make_device_view(w, h, db8);
    auto dx = img::make_device_view(w, h, db8);
    auto dy = img::make_device_view(w, h, db8);

    img::copy_to_device(src, s);

    img::gradients_xy(s, dx, dy);

    img::copy_to_host(dx, dst_x);
    img::copy_to_host(dy, dst_y);
}


static void compare_map_gray()
{
    auto n_channels32 = 2;
    auto n_channels8 = 1;
    auto n_d_ch32 = 1;
    auto n_d_ch8 = 1;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);
    auto d_buffer32 = img::create_device_buffer32(width * height * n_d_ch32);
    auto d_buffer8 = img::create_device_buffer8(width * height * n_d_ch8);

    auto src = img::make_view(width, height, buffer32);
    auto dst = img::make_view(width, height, buffer8);

    PROFILE(map_gray_interleaved(src, dst));
    PROFILE(map_gray_planar(src, dst, buffer32));
    PROFILE(map_gray_cuda(src, dst, d_buffer32, d_buffer8));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_alpha_blend()
{
    auto n_channels32 = 13;
    auto n_d_ch32 = 3;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto d_buffer32 = img::create_device_buffer32(width * height * n_d_ch32);

    auto src = img::make_view(width, height, buffer32);
    auto cur = img::make_view(width, height, buffer32);
    auto dst = img::make_view(width, height, buffer32);

    PROFILE(alpha_blend_interleaved(src, cur, dst));
    PROFILE(alpha_blend_planar(src, cur, dst, buffer32));
    PROFILE(alpha_blend_cuda(src, cur, dst, d_buffer32));

    img::destroy_buffer(buffer32);
}


static void compare_rotate()
{
    auto n_channels32 = 10;
    auto n_channels8 = 2;
    auto n_d_ch32 = 2;
    auto n_d_ch8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);
    auto d_buffer32 = img::create_device_buffer32(width * height * n_d_ch32);
    auto d_buffer8 = img::create_device_buffer8(width * height * n_d_ch8);

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
    PROFILE(rotate_rgb_cuda(src_rgba, dst_rgba, origin, angle_rad, d_buffer32));
    PROFILE(rotate_gray_cuda(src_gray, dst_gray, origin, angle_rad, d_buffer8));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_blur()
{
    auto n_channels32 = 10;
    auto n_channels8 = 2;
    auto n_d_ch32 = 2;
    auto n_d_ch8 = 2;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);
    auto d_buffer32 = img::create_device_buffer32(width * height * n_d_ch32);
    auto d_buffer8 = img::create_device_buffer8(width * height * n_d_ch8);

    auto src_rgba = img::make_view(width, height, buffer32);
    auto dst_rgba = img::make_view(width, height, buffer32);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);

    PROFILE(blur_rgb_interleaved(src_rgba, dst_rgba));
    PROFILE(blur_gray_interleaved(src_gray, dst_gray));
    PROFILE(blur_rgb_planar(src_rgba, dst_rgba, buffer32));
    PROFILE(blur_gray_planar(src_gray, dst_gray, buffer32));
    PROFILE(blur_rgb_cuda(src_rgba, dst_rgba, d_buffer32));
    PROFILE(blur_gray_cuda(src_gray, dst_gray, d_buffer8));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


static void compare_gradients()
{
    auto n_channels32 = 5;
    auto n_channels8 = 4;
    auto n_d_ch8 = 5;

    auto width = WIDTH;
    auto height = HEIGHT;

    auto buffer32 = img::create_buffer32(width * height * n_channels32);
    auto buffer8 = img::create_buffer8(width * height * n_channels8);
    auto d_buffer8 = img::create_device_buffer8(width * height * n_d_ch8);

    auto src_gray = img::make_view(width, height, buffer8);
    auto dst_gray = img::make_view(width, height, buffer8);
    auto dst_x = img::make_view(width, height, buffer8);
    auto dst_y = img::make_view(width, height, buffer8);

    PROFILE(gradients_u8(src_gray, dst_gray));
    PROFILE(gradients_xy_u8(src_gray, dst_x, dst_y));
    PROFILE(gradients_f32(src_gray, dst_gray, buffer32));
    PROFILE(gradients_xy_f32(src_gray, dst_x, dst_y, buffer32));
    PROFILE(gradients_cuda(src_gray, dst_gray, d_buffer8));
    PROFILE(gradients_xy_cuda(src_gray, dst_x, dst_y, d_buffer8));

    img::destroy_buffer(buffer32);
    img::destroy_buffer(buffer8);
}


void run_cuda_profile_tests()
{
    run_test(create_destroy_device_buffer, "create_destroy_device_buffer");
    run_test(make_device_view, "make_device_view");

    run_test(compare_map_gray, "compare_map_gray");
    run_test(compare_alpha_blend, "compare_alpha_blend");
    run_test(compare_rotate, "compare_rotate");
    run_test(compare_blur, "compare_blur");
    run_test(compare_gradients, "compare_gradients");
}