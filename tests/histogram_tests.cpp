#include "tests_include.hpp"

#include <array>
#include <algorithm>


constexpr u32 BIN_WIDTH = 5;
constexpr u32 BIN_SPACE = 1;
constexpr u32 HIST_HEIGHT = 100;
constexpr u32 HIST_SPACE = 5;
constexpr auto N_BINS = 64;


static void fill_to_top(img::View1r32 const& view, r32 value, u8 color)
{
    assert(value >= 0.0f);
    assert(value <= 1.0f);

    int y_begin = (int)(view.height * (1.0f - value));

    auto r = make_range(view);

    if (y_begin < 0)
    {
        r.y_begin = 0;
    }
    else if ((u32)y_begin >= view.height)
    {
        return;
    }
    else
    {
        r.y_begin = (u32)y_begin;
    }    

    img::fill(img::sub_view(view, r), color);
}


static void draw_histogram(const r32* values, u32 n_bins, img::View1r32 const& dst)
{
    u32 space_px = BIN_SPACE;
    auto width = BIN_WIDTH;

    auto max = *std::max_element(values, values + n_bins);

    img::fill(dst, 128);

    auto r = make_range(width, dst.height);

    for (u32 i = 0; i < n_bins; ++i)
    {
        auto val = max == 0.0f ? 0.0f : values[i] / max;

        fill_to_top(img::sub_view(dst, r), val, 0);

        r.x_begin += (width + space_px);
        r.x_end += (width + space_px);
    }
}


static void draw(img::Histogram9r32& hists, img::View1r32 const& dst)
{
    img::fill(dst, 255);

    u32 space_px = HIST_SPACE;
    auto height = HIST_HEIGHT;

    std::array<const r32*, 9> list = 
    {
        hists.rgb.R,
        hists.rgb.G,
        hists.rgb.B,
        hists.hsv.H,
        hists.hsv.S,
        hists.hsv.V,
        hists.yuv.Y,
        hists.yuv.U,
        hists.yuv.V
    };

    auto r = make_range(dst.width, height);
    r.x_begin = space_px;
    r.x_end -= space_px;

    for (auto hist : list)
    {
        r.y_begin += space_px;
        r.y_end += space_px;

        draw_histogram(hist, N_BINS, img::sub_view(dst, r));

        r.y_begin += height;
        r.y_end += height;
    }
}


static bool histogram_fill_test()
{
    auto title = "histogram_fill_test";
    printf("\n%s:\n", title);
    auto out_dir = IMAGE_OUT_PATH / title;
    empty_dir(out_dir);
    auto const write_image = [&out_dir](auto const& image, const char* name)
    { img::write_image(image, out_dir / name); };

    u32 width = N_BINS * (BIN_WIDTH + BIN_SPACE) - BIN_SPACE + 2 * HIST_SPACE;
    u32 height = 9 * (HIST_HEIGHT + HIST_SPACE) + HIST_SPACE;

    GrayImage hist_image;
    img::create_image(hist_image, width, height);
    auto dst = img::make_view(hist_image);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height);

    auto hist_view = img::make_view_1(width, height, buffer);

    img::Histogram9r32 hists;
    hists.n_bins = N_BINS;

    Image image;
    img::create_image(image, 256, 64);
    auto view = img::make_view(image);

    auto const do_hist = [&](const char* filename) 
    {
        img::histograms(view, hists);
        draw(hists, hist_view);
        img::map(hist_view, dst);
        write_image(hist_image, filename);
    };

    auto r = make_range(view);

    for (u32 i = 0; i < 256; ++i)
    {
        r.x_begin = i;
        r.x_end = i + 1;
        img::fill(img::sub_view(view, r), img::to_pixel(i, i, i));
    }
    do_hist("all.bmp");

    img::fill(view, img::to_pixel(255, 0, 0));
    do_hist("red.bmp");

    img::fill(view, img::to_pixel(0, 255, 0));
    do_hist("green.bmp");

    img::fill(view, img::to_pixel(0, 0, 255));
    do_hist("blue.bmp");

    img::fill(view, img::to_pixel(255, 255, 255));
    do_hist("white.bmp");    

    img::destroy_image(hist_image);
    img::destroy_image(image);
    mb::destroy_buffer(buffer);

    return true;
}


static bool histogram_images_test()
{
    auto title = "histogram_images_test";
	printf("\n%s:\n", title);
	auto out_dir = IMAGE_OUT_PATH / title;
	empty_dir(out_dir);
	auto const write_image = [&out_dir](auto const& image, const char* name) 
        { img::write_image(image, out_dir / name); };

    u32 width = N_BINS * (BIN_WIDTH + BIN_SPACE) - BIN_SPACE + 2 * HIST_SPACE;
    u32 height = 9 * (HIST_HEIGHT + HIST_SPACE) + HIST_SPACE;

    GrayImage hist_image;
    img::create_image(hist_image, width, height);
    auto dst = img::make_view(hist_image);

    img::Buffer32 buffer;
    mb::create_buffer(buffer, width * height);

    auto hist_view = img::make_view_1(width, height, buffer);

    img::Histogram9r32 hists;
    hists.n_bins = N_BINS;

    auto const do_hist = [&](Image const& image, const char* filename)
    {
        img::histograms(img::make_view(image), hists);
        draw(hists, hist_view);
        img::map(hist_view, dst);
        write_image(hist_image, filename);
    };

    Image vette;
    img::read_image_from_file(CORVETTE_PATH, vette);

    Image caddy;
    img::read_image_from_file(CADILLAC_PATH, caddy);

    Image chess;
    img::read_image_from_file(CHESS_PATH, chess);

    do_hist(vette, "vette.bmp");
    do_hist(caddy, "caddy.bmp");
    do_hist(chess, "chess.bmp");


    img::destroy_image(hist_image);
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
        histogram_fill_test() &&
        histogram_images_test();

    if (result)
    {
        printf("histogram tests OK\n");
    }
    
    return result;
}