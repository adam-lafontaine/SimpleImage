#include "tests_include.hpp"

#include <array>


static void fill_to_top(img::View1r32 const& view, r32 value, u8 color)
{
    assert(value >= 0.0f);
    assert(value <= 1.0f);

    auto top = view.height * (1.0f - value);

    if (top >= 1.0f)
    {
        return;
    }

    auto r = make_range(view);

    r.y_begin = top <= 0.0f ? 0u : (u32)top;

    img::fill(img::sub_view(view, r), color);
}


static void draw_histogram(const r32* values, u32 n_bins, img::View1r32 const& dst)
{
    u32 space_px = 1;
    auto width = (dst.width - 2 * space_px) / n_bins - space_px;

    img::fill(dst, 255);

    auto r = make_range(width, dst.height);

    for (u32 i = 0; i < n_bins; ++i)
    {
        r.x_begin += space_px;
        r.x_end += space_px;

        fill_to_top(img::sub_view(dst, r), values[i], 0);

        r.x_begin += width;
        r.x_end += width;
    }
}


static void draw(img::HistRGB const& rgb, img::HistHSV const& hsv, img::HistYUV const& yuv, img::View1r32 const& dst)
{
    u32 space_px = 5;
    auto height = dst.height / 9 - space_px;

    std::array<const r32*, 9> hists = 
    {
        rgb.R,
        rgb.G,
        rgb.B,
        hsv.H,
        hsv.S,
        hsv.V,
        yuv.Y,
        yuv.U,
        yuv.V
    };

    auto r = make_range(dst.width, height);

    for (auto hist : hists)
    {
        r.y_begin += space_px;
        r.y_end += space_px;

        draw_histogram(hist, 256, img::sub_view(dst, r));

        r.y_begin += height;
        r.y_end += height;
    }
}


static bool histogram_draw_test()
{
    auto title = "histogram_draw_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    u32 block_width = 5;
    u32 block_space = 1;
    u32 block_height = 105;

    u32 width = 256 * (block_width + block_space) + block_space;
    u32 height = 9 * block_height;

    GrayImage image;
    img::create_image(image, width, height);
    auto view = img::make_view(image);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height);

    auto hist_view = img::make_view_1(width, height, buffer);

    img::HistRGB h_rgb = { 0 };
    img::HistHSV h_hsv = { 0 };
    img::HistYUV h_yuv = { 0 };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);

    Image caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);

    Image chess;
    img::read_image_from_file(CHESS_PATH, chess);

    img::histograms(img::make_view(vette), h_rgb, h_hsv, h_yuv);
    draw(h_rgb, h_hsv, h_yuv, hist_view);
    img::map(hist_view, view);
    write_image(image, "vette.bmp");


    img::destroy_image(image);
    mb::destroy_buffer(buffer);
    img::destroy_image(vette);
    img::destroy_image(caddy);
    img::destroy_image(chess);

    return true;
}


bool histogram_tests()
{
    printf("\n*** histogram tests ***\n");

    auto result = 
        histogram_draw_test();

    if (result)
    {
        printf("histogram tests OK\n");
    }
    
    return result;
}