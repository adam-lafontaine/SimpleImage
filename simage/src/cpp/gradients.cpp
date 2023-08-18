/* gradients static */

namespace simage
{
    template <typename T>
	static f32 gradient_x_outer_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = std::min(y, h - y - 1);
        auto c = std::min(x, w - x - 1);
        auto rc = std::min(r, c);

        c *= (rc != 0);

        constexpr auto grad_3x3 = GRAD_X_3x3.data();
        constexpr auto grad_3x5 = GRAD_X_3x5.data();
        constexpr auto grad_3x7 = GRAD_X_3x7.data();
        constexpr auto grad_3x9 = GRAD_X_3x9.data();
        constexpr auto grad_3x11 = GRAD_X_3x11.data();

        switch(c)
        { 
        case 0:
            return 0.0f;      
        case 1:
            return convolve_at_xy_f32<T, 3, 3>(view, x, y, (f32*)grad_3x3);

        case 2:
            return convolve_at_xy_f32<T, 5, 3>(view, x, y, (f32*)grad_3x5);
        
        case 3:
            return convolve_at_xy_f32<T, 7, 3>(view, x, y, (f32*)grad_3x7);

        case 4:
            return convolve_at_xy_f32<T, 9, 3>(view, x, y, (f32*)grad_3x9);
        
        default:
            return convolve_at_xy_f32<T, 11, 3>(view, x, y, (f32*)grad_3x11);
        }
    }


    template <typename T>
	static f32 gradient_y_outer_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        u32 const w = view.width;
		u32 const h = view.height;

        auto r = std::min(y, h - y - 1);
        auto c = std::min(x, w - x - 1);
        auto rc = std::min(r, c);

        r *= (rc != 0);

        constexpr auto grad_3x3 = GRAD_Y_3x3.data();
        constexpr auto grad_3x5 = GRAD_Y_3x5.data();
        constexpr auto grad_3x7 = GRAD_Y_3x7.data();
        constexpr auto grad_3x9 = GRAD_Y_3x9.data();
        constexpr auto grad_3x11 = GRAD_Y_3x11.data();

        switch(r)
        {
        case 0:
            return 0.0f;
        case 1:
            return convolve_at_xy_f32<T, 3, 3>(view, x, y, (f32*)grad_3x3);

        case 2:
            return convolve_at_xy_f32<T, 3, 5>(view, x, y, (f32*)grad_3x5);
        
        case 3:
            return convolve_at_xy_f32<T, 3, 7>(view, x, y, (f32*)grad_3x7);

        case 4:
            return convolve_at_xy_f32<T, 3, 9>(view, x, y, (f32*)grad_3x9);
        
        default:
            return convolve_at_xy_f32<T, 3, 11>(view, x, y, (f32*)grad_3x11);
        }
    }


    template <typename T>
	static f32 gradient_x_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        constexpr auto grad_3x11 = GRAD_X_3x11.data();
        return convolve_at_xy_f32<T, 11, 3>(view, x, y, (f32*)grad_3x11);
    }


    template <typename T>
	static f32 gradient_y_at_xy(View1<T> const& view, u32 x, u32 y)
    {
        constexpr auto grad_3x11 = GRAD_Y_3x11.data();
        return convolve_at_xy_f32<T, 3, 11>(view, x, y, (f32*)grad_3x11);
    }


    template <typename T, class convert_to_T>
    static void gradients_top_bottom(View1<T> const& src, View1<T> const& dst, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        T* d = 0;
        T* d2 = 0;

        u32 y = 0;
        u32 y2 = 0;
        u32 x = 0;

        y2 = height - 1;
        for (y = 0; y < 5; ++y)
        {
            d = row_begin(dst, y);
            d2 = row_begin(dst, y2);
            for (x = 0; x < width; ++x)
            {
                grad_x = gradient_x_outer_at_xy(src, x, y);
                grad_y = gradient_y_outer_at_xy(src, x, y);
                d[x] = convert(grad_x, grad_y);

                grad_x = gradient_x_outer_at_xy(src, x, y2);
                grad_y = gradient_y_outer_at_xy(src, x, y2);
                d2[x] = convert(grad_x, grad_y);
            }

            --y2;
        }
    }


    template <typename T, class convert_to_T>
    static void gradients_left_right(View1<T> const& src, View1<T> const& dst, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        for (u32 y = 5; y < height - 5; ++y)
        {
            auto d = row_begin(dst, y);

            u32 x2 = width - 1;
            for (u32 x = 0; x < 5; ++x)
            {
                grad_x = gradient_x_outer_at_xy(src, x, y);
                grad_y = gradient_y_outer_at_xy(src, x, y);
                d[x] = convert(grad_x, grad_y);

                grad_x = gradient_x_outer_at_xy(src, x2, y);
                grad_y = gradient_y_outer_at_xy(src, x2, y);
                d[x2] = convert(grad_x, grad_y);

                --x2;
            }
        }
    }


    template <typename T, class convert_to_T>
    static void gradients_middle(View1<T> const& src, View1<T> const& dst, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        for (u32 y = 5; y < height - 5; ++y)
        {
            auto d = row_begin(dst, y);
            for (u32 x = 5; x < width - 5; ++x)
            {
                grad_x = gradient_x_at_xy(src, x, y);
                grad_y = gradient_y_at_xy(src, x, y);
                d[x] = convert(grad_x, grad_y);
            }
        }
    }


    template <typename T, class convert_to_T>
    static void gradients_xy_top_bottom(View1<T> const& src, View1<T> const& dst_x, View1<T> const& dst_y, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        u32 y2 = height - 1;
        for (u32 y = 0; y < 5; ++y)
        {
            auto d_x = row_begin(dst_x, y);
            auto d_y = row_begin(dst_y, y);
            auto d_x2 = row_begin(dst_x, y2);
            auto d_y2 = row_begin(dst_y, y2);
            for (u32 x = 0; x < width; ++x)
            {
                grad_x = gradient_x_outer_at_xy(src, x, y);
                grad_y = gradient_y_outer_at_xy(src, x, y);
                d_x[x] = convert(grad_x);
                d_y[x] = convert(grad_y);

                grad_x = gradient_x_outer_at_xy(src, x, y2);
                grad_y = gradient_y_outer_at_xy(src, x, y2);
                d_x2[x] = convert(grad_x);
                d_y2[x] = convert(grad_y);
            }

            --y2;
        }
    }


    template <typename T, class convert_to_T>
    static void gradients_xy_left_right(View1<T> const& src, View1<T> const& dst_x, View1<T> const& dst_y, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        for (u32 y = 5; y < height - 5; ++y)
        {
            auto d_x = row_begin(dst_x, y);
            auto d_y = row_begin(dst_y, y);

            u32 x2 = width - 1;
            for (u32 x = 0; x < 5; ++x)
            {
                grad_x = gradient_x_outer_at_xy(src, x, y);
                grad_y = gradient_y_outer_at_xy(src, x, y);
                d_x[x] = convert(grad_x);
                d_y[x] = convert(grad_y);

                grad_x = gradient_x_outer_at_xy(src, x2, y);
                grad_y = gradient_y_outer_at_xy(src, x2, y);
                d_x[x2] = convert(grad_x);
                d_y[x2] = convert(grad_y);
            }

            --x2;
        }
    }


    template <typename T, class convert_to_T>
    static void gradients_xy_middle(View1<T> const& src, View1<T> const& dst_x, View1<T> const& dst_y, convert_to_T const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        f32 grad_x = 0.0f;
        f32 grad_y = 0.0f;

        for (u32 y = 5; y < height - 5; ++y)
        {
            auto d_x = row_begin(dst_x, y);
            auto d_y = row_begin(dst_y, y);
            for (u32 x = 5; x < width - 5; ++x)
            {
                grad_x = gradient_x_at_xy(src, x, y);
                grad_y = gradient_y_at_xy(src, x, y);
                d_x[x] = convert(grad_x);
                d_y[x] = convert(grad_y);
            }
        }
    }


    static void gradients_xy_middle(View1<f32> const& src, View1<f32> const& dst_x, View1<f32> const& dst_y)
    {
        auto const width = src.width;
        auto const height = src.height;

        constexpr auto grad_x_3x11 = GRAD_X_3x11.data();
        constexpr auto grad_y_3x11 = GRAD_Y_3x11.data();

        u32 x_begin = 5;
		u32 x_end = width - 5;

        for (u32 y = 5; y < height - 5; ++y)
        {
            convolve_span<11, 3>(src, dst_x, x_begin, x_end, y, (f32*)grad_x_3x11);
            convolve_span<3, 11>(src, dst_y, x_begin, x_end, y, (f32*)grad_y_3x11);
        }
    }
}


/* gradients */

namespace simage
{
    void gradients(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));
        
        gradients_top_bottom(src, dst, hypot_to_u8);
        gradients_left_right(src, dst, hypot_to_u8);
        gradients_middle(src, dst, hypot_to_u8);
    }


    void gradients_xy(ViewGray const& src, ViewGray const& dst_x, ViewGray const& dst_y)
    {
        assert(verify(src, dst_x));
        assert(verify(src, dst_y));
        
        gradients_xy_top_bottom(src, dst_x, dst_y, abs_to_u8);
        gradients_xy_left_right(src, dst_x, dst_y, abs_to_u8);
        gradients_xy_middle(src, dst_x, dst_y, abs_to_u8);
    }
    

    void gradients(View1f32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        gradients_top_bottom(src, dst, std::hypotf);
        gradients_left_right(src, dst, std::hypotf);
        gradients_middle(src, dst, std::hypotf);
        
    }


	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst)
	{
		auto dst_x = select_channel(xy_dst, XY::X);
		auto dst_y = select_channel(xy_dst, XY::Y);

		assert(verify(src, dst_x));
		assert(verify(src, dst_y));

        auto const f = [](f32 a){ return a; };

        gradients_xy_top_bottom(src, dst_x, dst_y, f);
        gradients_xy_left_right(src, dst_x, dst_y, f);
        gradients_xy_middle(src, dst_x, dst_y);
	}
}
