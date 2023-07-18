/* gradients static */

namespace simage
{
    template <typename T>
    static f32 gradient_x_11(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 3; ++v)
        {
            auto s = row_begin(view, y - 1 + v);
            for (u32 u = 0; u < 11; ++u)
            {
                total += s[x - 5 + u] * GRAD_X_3x11[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_y_11(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 11; ++v)
        {
            auto s = row_begin(view, y - 5 + v);
            for (u32 u = 0; u < 3; ++u)
            {
                total += s[x - 1 + u] * GRAD_Y_3x11[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_x_5(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 3; ++v)
        {
            auto s = row_begin(view, y - 1 + v);
            for (u32 u = 0; u < 5; ++u)
            {
                total += s[x - 2 + u] * GRAD_X_3x5[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_y_5(View1<T> const& view, u32 x, u32 y)
    {
        f32 total = 0.0f;
        u32 w = 0;

        for (u32 v = 0; v < 5; ++v)
        {
            auto s = row_begin(view, y - 2 + v);
            for (u32 u = 0; u < 3; ++u)
            {
                total += s[x - 1 + u] * GRAD_Y_3x5[w++];
            }
        }

        return total;
    }


    template <typename T>
    static f32 gradient_x_3(View1<T> const& view, u32 x, u32 y)
    {
		return convolve_at_xy(view, x, y, GRAD_X_3x3);
    }


    template <typename T>
    static f32 gradient_y_3(View1<T> const& view, u32 x, u32 y)
    {
		return convolve_at_xy(view, x, y, GRAD_X_3x3);
    }


    template <typename T>
    static T gradient_xy_11(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_11(view, x, y);
        auto grad_y = gradient_y_11(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static T gradient_xy_5(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_5(view, x, y);
        auto grad_y = gradient_y_5(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static T gradient_xy_3(View1<T> const& view, u32 x, u32 y)
    {
        auto grad_x = gradient_x_3(view, x, y);
        auto grad_y = gradient_y_3(view, x, y);

        return (T)std::hypotf(grad_x, grad_y);
    }


    template <typename T>
    static void gradients_row(View1<T> const& src, View1<T> const& dst, u32 y)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y >= 5 && y < height - 5)
		{
			d[0] = d[width - 1] = 0;

            d[1] = gradient_xy_3(src, 1, y);
            d[width - 2] = gradient_xy_3(src, width - 2, y);

            for (u32 x = 2; x < 5; ++x)
            {
                d[x] = gradient_xy_5(src, x, y);
                d[width - x - 1] = gradient_xy_5(src, width - x - 1, y);
            }

            for (u32 x = 5; x < width - 5; ++x)
            {
                d[x] = gradient_xy_11(src, x, y);
            }

			return;
		}
		
		if (y >= 2 && y < 5 || y >= height - 5 && y <= height - 3)
		{
			d[0] = d[width - 1] = 0;

            d[1] = gradient_xy_3(src, 1, y);
            d[width - 2] = gradient_xy_3(src, width - 2, y);

            for (u32 x = 2; x < width - 3; ++x)
            {
                d[x] = gradient_xy_5(src, x, y);
            }
            return;
		}

		if (y == 1 || y == height - 2)
		{
			 d[0] = d[width - 1] = 0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = gradient_xy_3(src, x, y);
            }
            return;
		}

		// y == 0 || y == height - 1
		for (u32 x = 0; x < width ; ++x)
		{
			d[x] = (T)0;
		}
    }


    template <typename T, class CONV>
    static void gradients_x_row(View1<T> const& src, View1<T> const& dst, u32 y, CONV const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y > 0 && y < height - 1)
		{
			d[0] = d[width - 1] = (T)0;

            d[1] = convert(gradient_x_3(src, 1, y));
            d[width - 2] = convert(gradient_x_3(src, width - 2, y));

            for (u32 x = 2; x < 5; ++x)
            {
                d[x] = convert(gradient_x_5(src, x, y));
                d[width - x - 1] = convert(gradient_x_5(src, width - x - 1, y));
            }

            for (u32 x = 5; x < width - 5; ++x)
            {
                d[x] = convert(gradient_x_11(src, x, y));
            }

			return;
		}

		// y == 0 || y == height - 1
		for (u32 x = 0; x < width ; ++x)
		{
			d[x] = (T)0;
		}
    }


    template <typename T, class CONV>
    static void gradients_y_row(View1<T> const& src, View1<T> const& dst, u32 y, CONV const& convert)
    {
        auto const width = src.width;
        auto const height = src.height;

        auto d = row_begin(dst, y);

		if (y >= 5 && y < height - 5)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_11(src, x, y));
            }

			return;
		}
		
		if (y >= 2 && y < 5 || y >= height - 5 && y <= height - 3)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_5(src, x, y));
            }
            return;
		}

		if (y == 1 || y == height - 2)
		{
			d[0] = d[width - 1] = (T)0;

            for (u32 x = 1; x < width - 1; ++x)
            {
                d[x] = convert(gradient_y_3(src, x, y));
            }
            return;
		}

		if (y == 0 || y == height - 1)
		{
			for (u32 x = 0; x < width ; ++x)
            {
                d[x] = (T)0;
            }
            return;
		}
    }


    template <typename T>
    static void gradients_1(View1<T> const& src, View1<T> const& dst)
    {   
        for (u32 y = 0; y < src.height; ++y)
        {
            gradients_row(src, dst, y);
        }
    }
	
	
	template <typename T>
    static void gradients_xy_1(View1<T> const& src, View1<T> const& x_dst, View1<T> const& y_dst)
    {
		auto const convert = [](f32 grad){ return (T)grad; };

        for (u32 y = 0; y < src.height; ++y)
        {
            gradients_x_row(src, x_dst, y, convert);
            gradients_y_row(src, y_dst, y, convert);
        }
    }


    template <typename T>
    static void gradients_unsigned_xy_1(View1<T> const& src, View1<T> const& x_dst, View1<T> const& y_dst)
    {
		auto const convert = [](f32 grad){ return (T)std::abs(grad); };

        for (u32 y = 0; y < src.height; ++y)
        {
            gradients_x_row(src, x_dst, y, convert);
            gradients_y_row(src, y_dst, y, convert);
        }
    }
}


/* gradients */

namespace simage
{
    void gradients(ViewGray const& src, ViewGray const& dst)
    {
        assert(verify(src, dst));

        gradients_1(src, dst);
    }


    void gradients_xy(ViewGray const& src, ViewGray const& dst_x, ViewGray const& dst_y)
    {
        assert(verify(src, dst_x));
        assert(verify(src, dst_y));

        gradients_unsigned_xy_1(src, dst_x, dst_y);
    }
}


/* gradients */

namespace simage
{
    void gradients(View1f32 const& src, View1f32 const& dst)
    {
        assert(verify(src, dst));

        gradients_1(src, dst);
    }


	void gradients_xy(View1f32 const& src, View2f32 const& xy_dst)
	{
		auto dst_x = select_channel(xy_dst, XY::X);
		auto dst_y = select_channel(xy_dst, XY::Y);

		assert(verify(src, dst_x));
		assert(verify(src, dst_y));

		gradients_xy_1(src, dst_x, dst_y);
	}
}
