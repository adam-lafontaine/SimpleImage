#include "tests_include.hpp"


void blur_test(img::View const& out) // broken
{
    auto width = out.width;
    auto height = out.height;

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 2);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::blur(src, dst);

    img::map_gray(dst, out);

    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void gradients_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 2);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CHESS_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::gradients(src, dst);

    img::map_gray(dst, out);

    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}


void gradients_xy_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 3);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CHESS_PATH, image, width, height, buffer);
    auto dst_x = img::make_view(width, height, buffer);
    auto dst_y = img::make_view(width, height, buffer);

    auto full = make_range(width, height);

    auto left = full;
    left.x_end = width / 2;

    auto right = full;
    right.x_begin = left.x_end; 
    

    img::gradients_xy(src, dst_x, dst_y);

    img::map_gray(img::sub_view(dst_x, left), img::sub_view(out, left));
    img::map_gray(img::sub_view(dst_y, right), img::sub_view(out, right));

    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}