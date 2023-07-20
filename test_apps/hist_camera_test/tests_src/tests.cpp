#include "../../tests_include.hpp"

#include <array>
#include <algorithm>

constexpr u32 BIN_SPACE = 1;
constexpr u32 HIST_SPACE = 5;
constexpr auto N_BINS = 64;

static img::Buffer8 buffer8;
static img::ViewGray hist_view;


bool init_histogram_memory(u32 width, u32 height)
{
    buffer8 = img::create_buffer8(width * height);

    hist_view = img::make_view(width, height, buffer8);

    return buffer8.data_;
}


void destroy_histogram_memory()
{
    img::destroy_buffer(buffer8);
}


class HistParams
{
public:
    u32 n_bins;
    u32 bin_width;
    u32 bin_space;
    u32 hist_height;
    u32 hist_space;
};


static void fill_to_top(img::View1u8 const& view, f32 value, u8 color)
{
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


static void draw_histogram(const f32* values, img::View1u8 const& dst, HistParams const& props)
{
    u32 space_px = props.bin_space;
    auto width = props.bin_width;

    auto max = *std::max_element(values, values + props.n_bins);

    img::fill(dst, 128);

    auto r = make_range(width, dst.height);

    for (u32 i = 0; i < props.n_bins; ++i)
    {
        auto val = max == 0.0f ? 0.0f : values[i] / max;

        fill_to_top(img::sub_view(dst, r), val, 0);

        r.x_begin += (width + space_px);
        r.x_end += (width + space_px);
    }
}


static void draw(img::hist::Histogram12f32& hists, img::View1u8 const& dst, HistParams const& props)
{
    img::fill(dst, 255);

    u32 space_px = props.hist_space;
    auto height = props.hist_height;

    auto r = make_range(dst.width, height);
    r.x_begin = space_px;
    r.x_end -= space_px;

    for (u32 i = 0; i < 12; ++i)
    {
        r.y_begin += space_px;
        r.y_end += space_px;

        draw_histogram(hists.list[i], img::sub_view(dst, r), props);

        r.y_begin += height;
        r.y_end += height;
    }
}


void generate_histograms(img::View const& src, img::View const& dst)
{
    u32 width = src.width;
    u32 height = src.height;

    HistParams params{};
    params.n_bins = N_BINS;
    params.bin_space = BIN_SPACE;
    params.hist_space = HIST_SPACE;
    params.bin_width = (width + BIN_SPACE - 2 * HIST_SPACE) / N_BINS - BIN_SPACE;
    params.hist_height = (height - HIST_SPACE) / 12 - HIST_SPACE;

    img::hist::Histogram12f32 hists;
    hists.n_bins = N_BINS;

    img::hist::make_histograms(src, hists);
    draw(hists, hist_view, params);

    img::map_rgba(hist_view, dst);
}