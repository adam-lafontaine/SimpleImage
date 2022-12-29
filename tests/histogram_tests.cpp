#include "tests_include.hpp"


static void fill_to_top(img::View1r32 const& view, r32 value, r32 color)
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