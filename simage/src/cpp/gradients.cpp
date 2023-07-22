/* gradients static */

namespace simage
{
    template <typename T>
	static f32 gradient_x_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = std::min(y, h - y - 1);
        auto c = std::min(x, w - x - 1);
        auto rc = std::min(r, c);

        if (rc == 0)
        {
            return 0.0f;
        }

        switch(c)
        {
        case 1:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_X_3x3.data(), 3, 3);

        case 2:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_X_3x5.data(), 5, 3);
        
        case 3:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_X_3x7.data(), 7, 3);

        case 4:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_X_3x9.data(), 9, 3);
        
        default:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_X_3x11.data(), 11, 3);
        }
    }


    template <typename T>
	static f32 gradient_y_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = std::min(y, h - y - 1);
        auto c = std::min(x, w - x - 1);
        auto rc = std::min(r, c);

        if (rc == 0)
        {
            return 0.0f;
        }

        switch(r)
        {
        case 1:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_Y_3x3.data(), 3, 3);

        case 2:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_Y_3x5.data(), 3, 5);
        
        case 3:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_Y_3x7.data(), 3, 7);

        case 4:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_Y_3x9.data(), 3, 9);
        
        default:
            return convolve_at_xy_f32(view, x, y, (f32*)GRAD_Y_3x11.data(), 3, 11);
        }
    }
}


/* gradients */

namespace simage
{
    void gradients(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.height; ++y)
        {
            auto d = row_begin(dst, y);
            for (u32 x = 0; x < src.width; ++x)
            {
                auto grad_x = gradient_x_at_xy(src, x, y);
                auto grad_y = gradient_y_at_xy(src, x, y);
                d[x] = round_to_u8(std::hypotf(grad_x, grad_y));
            }
        }
    }


    void gradients_xy(ViewGray const& src, ViewGray const& dst_x, ViewGray const& dst_y)
    {
        assert(verify(src, dst_x));
        assert(verify(src, dst_y));

        for (u32 y = 0; y < src.height; ++y)
        {
            auto d_x = row_begin(dst_x, y);
            auto d_y = row_begin(dst_y, y);
            for (u32 x = 0; x < src.width; ++x)
            {
                auto grad_x = gradient_x_at_xy(src, x, y);
                auto grad_y = gradient_y_at_xy(src, x, y);
                d_x[x] = round_to_u8(std::abs(grad_x));
                d_y[x] = round_to_u8(std::abs(grad_y));
            }
        }
    }
}


/* gradients */

namespace simage
{
    void gradients(View1f32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        for (u32 y = 0; y < src.height; ++y)
        {
            auto d = row_begin(dst, y);
            for (u32 x = 0; x < src.width; ++x)
            {
                auto grad_x = gradient_x_at_xy(src, x, y);
                auto grad_y = gradient_y_at_xy(src, x, y);
                d[x] = std::hypotf(grad_x, grad_y);
            }
        }
    }


	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst)
	{
		auto dst_x = select_channel(xy_dst, XY::X);
		auto dst_y = select_channel(xy_dst, XY::Y);

		assert(verify(src, dst_x));
		assert(verify(src, dst_y));

        for (u32 y = 0; y < src.height; ++y)
        {
            auto d_x = row_begin(dst_x, y);
            auto d_y = row_begin(dst_y, y);
            for (u32 x = 0; x < src.width; ++x)
            {
                d_x[x] = gradient_x_at_xy(src, x, y);
                d_y[x] = gradient_y_at_xy(src, x, y);
            }
        }
	}
}
