#include "../../tests_include.hpp"


void blur_gray_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer8(width * height * 2);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CADILLAC_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::blur(src, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void blur_rgb_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer32(width * height * 2);

    img::Image image;
    auto src = img::make_view_resized_from_file(CORVETTE_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::blur(src, dst);

    img::copy(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void gradients_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer8(width * height * 2);

    img::ImageGray image;
    auto src = img::make_view_resized_from_file(CHESS_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::gradients(src, dst);

    img::map_rgba(dst, out);

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}


void gradients_xy_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    auto buffer = img::create_buffer8(width * height * 3);

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

    img::map_rgba(img::sub_view(dst_x, left), img::sub_view(out, left));
    img::map_rgba(img::sub_view(dst_y, right), img::sub_view(out, right));

    img::destroy_image(image);
    img::destroy_buffer(buffer);
}