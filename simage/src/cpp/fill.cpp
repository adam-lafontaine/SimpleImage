/* fill span */

namespace simage
{
	template <typename T>
	static void fill_channel_no_simd(View1<T> const& view, T value)
	{
		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			for (u32 i = 0; i < view.width; ++i)
			{
				d[i] = value;
			}
		}
	}


#ifdef SIMAGE_NO_SIMD


	template <typename T>
	static void fill_channel(View1<T> const& view, T value)
	{
		fill_channel_no_simd(view, value);
	}

#else


	static inline void fill_span(f32* dst, simd::vecf32 const& v_val, u32 len)
	{
		constexpr auto step = simd::LEN;		

		u32 i = 0;
        for (; i <= (len - step); i += step)
		{
			simd::store_f32(v_val, dst + i);
		}

		i = len - step;
		simd::store_f32(v_val, dst + i);
	}


	template <typename T>
	static void fill_channel(View1<T> const& view, T value)
	{
		fill_channel_no_simd(view, value);
	}


	static void fill_channel(View1<f32> const& view, f32 value)
	{
		if (view.width < simd::LEN)
		{
			fill_channel_no_simd(view, value);
			return;
		}

		simd::vecf32 v_val{};
		simd::load_f32_broadcast(value, v_val);

		for (u32 y = 0; y < view.height; ++y)
		{
			auto d = row_begin(view, y);
			fill_span(d, v_val, view.width);
		}
	}

#endif // SIMAGE_NO_SIMD
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color)
	{
		assert(verify(view));

		fill_channel(view, color);
	}


	void fill(ViewGray const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}
}


/* fill */

namespace simage
{

	void fill(View4f32 const& view, Pixel color)
	{
		assert(verify(view));

		f32 colors[4] = 
		{
			cs::to_channel_f32(color.channels[0]),
			cs::to_channel_f32(color.channels[1]),
			cs::to_channel_f32(color.channels[2]),
			cs::to_channel_f32(color.channels[3])
		};

		std::array<std::function<void()>, 4> f_list
		{
			[&](){ fill_channel(select_channel(view, 0), colors[0]); },
			[&](){ fill_channel(select_channel(view, 1), colors[1]); },
			[&](){ fill_channel(select_channel(view, 2), colors[2]); },
			[&](){ fill_channel(select_channel(view, 3), colors[3]); },
		};

    	execute(f_list);
	}


	void fill(View3f32 const& view, Pixel color)
	{
		assert(verify(view));

		f32 colors[3] = 
		{
			cs::to_channel_f32(color.channels[0]),
			cs::to_channel_f32(color.channels[1]),
			cs::to_channel_f32(color.channels[2]),
		};

		std::array<std::function<void()>, 3> f_list
		{
			[&](){ fill_channel(select_channel(view, 0), colors[0]); },
			[&](){ fill_channel(select_channel(view, 1), colors[1]); },
			[&](){ fill_channel(select_channel(view, 2), colors[2]); },
		};

    	execute(f_list);
	}


	void fill(View1f32 const& view, f32 gray)
	{
		assert(verify(view));

		fill_channel(view, gray);
	}


	void fill(View1f32 const& view, u8 gray)
	{
		assert(verify(view));

		fill_channel(view, cs::to_channel_f32(gray));
	}
}
