#include "tests_include.hpp"


bool map_rgba_test(img::Image const& src, img::View const& dst)
{
    printf("map_rgba_test\n");

    auto width = src.width / 2;
    auto height = src.height;    

    auto r_src = make_range(width, height);
    auto r_dst = make_range(src.width, src.height);
    r_dst.x_end = src.width;
    r_dst.x_begin = r_dst.x_end - width;

    auto src_v = img::sub_view(src, r_src);
    auto dst_v = img::sub_view(dst, r_dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 4);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto dst_d = img::make_view_4(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgba(src_dv, dst_d);
    img::map_rgba(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_rgb_test(img::Image const& src, img::View const& dst)
{
    printf("map_rgb_test\n");

    auto width = src.width;
    auto height = src.height / 2;    

    auto r_src = make_range(width, height);
    auto r_dst = make_range(src.width, src.height);
    r_dst.y_end = src.height;
    r_dst.y_begin = r_dst.y_end - height;

    auto src_v = img::sub_view(src, r_src);
    auto dst_v = img::sub_view(dst, r_dst);

    img::DeviceBuffer32 buffer32;
    cuda::create_device_buffer(buffer32, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height * 3);

    auto src_dv = img::make_view(width, height, buffer32);
    auto dst_dv = img::make_view(width, height, buffer32);
    auto dst_d = img::make_view_3(width, height, buffer16);

    img::copy_to_device(src_v, src_dv);

    img::map_rgb(src_dv, dst_d);
    img::map_rgb(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_v);

    cuda::destroy_buffer(buffer32);
    cuda::destroy_buffer(buffer16);

    return true;
}


bool map_gray_test(img::Image const& src, img::View const& dst)
{
    printf("map_gray_test\n");

    auto width = src.width / 2;
    auto height = src.height;    

    auto r_dst = make_range(width, height);
    auto r_src = make_range(src.width, src.height);
    r_src.x_end = src.width;
    r_src.x_begin = r_src.x_end - width;

    auto src_v = img::sub_view(src, r_src);
    auto dst_v = img::sub_view(dst, r_dst);

    img::ImageGray src_gray;
    img::ImageGray dst_gray;
    img::create_image(src_gray, width, height);
    img::create_image(dst_gray, width, height);

    auto src_gray_v = img::make_view(src_gray);
    auto dst_gray_v = img::make_view(dst_gray);

    img::map_gray(src_v, src_gray_v);

    img::DeviceBuffer8 buffer8;
    cuda::create_device_buffer(buffer8, width * height * 2);

    img::DeviceBuffer16 buffer16;
    cuda::create_device_buffer(buffer16, width * height);

    auto src_dv = img::make_view(width, height, buffer8);
    auto dst_dv = img::make_view(width, height, buffer8);
    auto dst_d = img::make_view_1(width, height, buffer16);

    img::copy_to_device(src_gray_v, src_dv);

    img::map_gray(src_dv, dst_d);
    img::map_gray(dst_d, dst_dv);

    img::copy_to_host(dst_dv, dst_gray_v);

    img::map_gray(dst_gray_v, dst_v);

    img::destroy_image(src_gray);
    img::destroy_image(dst_gray);
    cuda::destroy_buffer(buffer8);
    cuda::destroy_buffer(buffer16);

    return true;
}