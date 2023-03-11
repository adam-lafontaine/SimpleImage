#include "tests_include.hpp"


bool map_hsv_test(img::View const& src, img::View const& dst)
{
    printf("map_hsv_test\n");
    img::copy(src, dst);

    auto width = src.width / 2;
    auto height = src.height;

    auto r = make_range(width, height);

    auto src_v = img::sub_view(src, r);
    auto dst_v = img::sub_view(dst, r);
    fill_green(dst_v);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto hsv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb_hsv(src_dv, hsv_d);
    img::map_hsv_rgb(hsv_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_hsv_red_test(img::View const& src, img::View const& dst)
{
    printf("map_hsv_red_test\n");    
    
    auto width = src.width / 2;
    auto height = src.height;

    auto left_r = make_range(width, height);
    auto right_r = left_r;
    right_r. x_end = src.width;
    right_r.x_begin = right_r.x_end - width;

    auto src_v = img::sub_view(src, left_r);
    
    auto dst_left_v = img::sub_view(dst, left_r);
    auto dst_right_v = img::sub_view(dst, right_r);

    fill_green(dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_left_dv = img::make_view(width, height, buffer32);
    auto dst_right_dv = img::make_view(width, height, buffer32);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 9);

    auto rgb_src_d = img::make_view_3(width, height, buffer16);
    auto rgb_dst_d = img::make_view_3(width, height, buffer16);    
    auto hsv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);
    img::map_rgb(src_dv, rgb_src_d);

    img::map_rgb_hsv(rgb_src_d, hsv_d);
    img::map_hsv_rgb(hsv_d, rgb_dst_d);

    // red channel
    auto left_d = img::select_channel(rgb_src_d, img::RGB::R);
    auto right_d = img::select_channel(rgb_dst_d, img::RGB::R);

    img::map_gray_rgb(left_d, dst_left_dv);
    img::map_gray_rgb(right_d, dst_right_dv);

    img::copy_to_host(dst_left_dv, dst_left_v);
    img::copy_to_host(dst_right_dv, dst_right_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_hsv_green_test(img::View const& src, img::View const& dst)
{
    printf("map_hsv_green_test\n");    
    
    auto width = src.width / 2;
    auto height = src.height;

    auto left_r = make_range(width, height);
    auto right_r = left_r;
    right_r. x_end = src.width;
    right_r.x_begin = right_r.x_end - width;

    auto src_v = img::sub_view(src, left_r);
    
    auto dst_left_v = img::sub_view(dst, left_r);
    auto dst_right_v = img::sub_view(dst, right_r);

    fill_green(dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_left_dv = img::make_view(width, height, buffer32);
    auto dst_right_dv = img::make_view(width, height, buffer32);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 9);

    auto rgb_src_d = img::make_view_3(width, height, buffer16);
    auto rgb_dst_d = img::make_view_3(width, height, buffer16);    
    auto hsv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);
    img::map_rgb(src_dv, rgb_src_d);

    img::map_rgb_hsv(rgb_src_d, hsv_d);
    img::map_hsv_rgb(hsv_d, rgb_dst_d);

    // green channel
    auto left_d = img::select_channel(rgb_src_d, img::RGB::G);
    auto right_d = img::select_channel(rgb_dst_d, img::RGB::G);

    img::map_gray_rgb(left_d, dst_left_dv);
    img::map_gray_rgb(right_d, dst_right_dv);

    img::copy_to_host(dst_left_dv, dst_left_v);
    img::copy_to_host(dst_right_dv, dst_right_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_hsv_blue_test(img::View const& src, img::View const& dst)
{
    printf("map_hsv_blue_test\n");    
    
    auto width = src.width / 2;
    auto height = src.height;

    auto left_r = make_range(width, height);
    auto right_r = left_r;
    right_r. x_end = src.width;
    right_r.x_begin = right_r.x_end - width;

    auto src_v = img::sub_view(src, left_r);
    
    auto dst_left_v = img::sub_view(dst, left_r);
    auto dst_right_v = img::sub_view(dst, right_r);

    fill_green(dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_left_dv = img::make_view(width, height, buffer32);
    auto dst_right_dv = img::make_view(width, height, buffer32);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 9);

    auto rgb_src_d = img::make_view_3(width, height, buffer16);
    auto rgb_dst_d = img::make_view_3(width, height, buffer16);    
    auto hsv_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);
    img::map_rgb(src_dv, rgb_src_d);

    img::map_rgb_hsv(rgb_src_d, hsv_d);
    img::map_hsv_rgb(hsv_d, rgb_dst_d);

    // blue channel
    auto left_d = img::select_channel(rgb_src_d, img::RGB::B);
    auto right_d = img::select_channel(rgb_dst_d, img::RGB::B);

    img::map_gray_rgb(left_d, dst_left_dv);
    img::map_gray_rgb(right_d, dst_right_dv);

    img::copy_to_host(dst_left_dv, dst_left_v);
    img::copy_to_host(dst_right_dv, dst_right_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}