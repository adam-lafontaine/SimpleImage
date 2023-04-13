#include "../../tests_include.hpp"


void centroid_test(img::View const& out)
{
    auto width = out.width;
    auto height = out.height;

    img::ImageGray image;

    img::Buffer8 buffer;
    mb::create_buffer(buffer, width * height * 2);    

    auto src = img::make_view_resized_from_file(WEED_PATH, image, width, height, buffer);
    auto dst = img::make_view(width, height, buffer);

    img::binarize(src, dst, [](u8 p){ return p < 150; });

    auto pt = img::centroid(dst);

    // region around centroid
	Range2Du32 c{};
	c.x_begin = pt.x - 10;
	c.x_end = pt.x + 10;
	c.y_begin = pt.y - 10;
	c.y_end = pt.y + 10;

    img::fill(img::sub_view(dst, c), 0);

    img::map_gray(dst, out);

    img::destroy_image(image);
    mb::destroy_buffer(buffer);
}