#include "tests_include.hpp"


bool copy_image_test(img::Image const& src, img::View const& dst)
{
    printf("copy_image_test\n");

    auto width = src.width;
    auto height = src.height;

    img::DeviceBuffer32 buffer;
    cuda::create_device_buffer(buffer, width * height);

    auto d_view = img::make_view(width, height, buffer);

    img::copy_to_device(img::make_view(src), d_view);
    img::copy_to_host(d_view, dst);

    cuda::destroy_buffer(buffer);

    return true;
}


bool copy_view_test(img::Image const& src, img::View const& dst)
{
    printf("copy_view_test\n");

    auto width = src.width / 2;
    auto height = src.height / 2;   

    auto src_r = make_range(width, height);
    
    auto src_v = img::sub_view(src, src_r);

    Range2Du32 dst_r{};
    dst_r.x_begin = width / 2;
    dst_r.x_end = dst_r.x_begin + width;
    dst_r.y_begin = height / 2;
    dst_r.y_end = dst_r.y_begin + height;

    auto dst_v = img::sub_view(dst, dst_r);

    img::DeviceBuffer32 buffer;
    cuda::create_device_buffer(buffer, width * height);

    auto d_view = img::make_view(width, height, buffer);

    img::copy_to_device(src_v, d_view);
    img::copy_to_host(d_view, dst_v);

    cuda::destroy_buffer(buffer);

    return true;
}


bool copy_gray_image_test(img::Image const& src, img::View const& dst)
{
    printf("copy_gray_image_test\n");

    auto width = src.width;
    auto height = src.height;

    img::ImageGray gray_src;
    img::create_image(gray_src, width, height);
    auto src_v = img::make_view(gray_src);

    img::ImageGray gray_dst;
    img::create_image(gray_dst, width, height);
    auto dst_v = img::make_view(gray_dst);

    img::map(img::make_view(src), src_v);

    img::DeviceBuffer8 buffer;
    cuda::create_device_buffer(buffer, width * height);

    auto d_view = img::make_view(width, height, buffer);

    img::copy_to_device(src_v, d_view);
    img::copy_to_host(d_view, dst_v);

    img::map(dst_v, dst);

    img::destroy_image(gray_src);
    img::destroy_image(gray_dst);
    cuda::destroy_buffer(buffer);

    return true;
}


bool copy_gray_view_test(img::Image const& src, img::View const& dst)
{
    printf("copy_view_test\n");

    img::ImageGray gray_src;
    img::create_image(gray_src, src.width, src.height);

    img::ImageGray gray_dst;
    img::create_image(gray_dst, dst.width, dst.height);

    auto width = src.width / 2;
    auto height = src.height / 2;   

    auto src_r = make_range(width, height);       
    auto src_v = img::sub_view(gray_src, src_r);

    Range2Du32 dst_r{};
    dst_r.x_begin = width / 2;
    dst_r.x_end = dst_r.x_begin + width;
    dst_r.y_begin = height / 2;
    dst_r.y_end = dst_r.y_begin + height;

    auto dst_v = img::sub_view(gray_dst, dst_r);

    img::DeviceBuffer8 buffer;
    cuda::create_device_buffer(buffer, width * height);

    auto d_view = img::make_view(width, height, buffer);

    img::copy_to_device(src_v, d_view);
    img::copy_to_host(d_view, dst_v);

    img::map(dst_v, img::sub_view(dst, dst_r));

    img::destroy_image(gray_src);
    img::destroy_image(gray_dst);
    cuda::destroy_buffer(buffer);

    return true;
}