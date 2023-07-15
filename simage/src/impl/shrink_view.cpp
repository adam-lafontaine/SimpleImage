/* shrink_view */
#if 0
namespace simage
{
	static f32 average(View1f32 const& view)
	{
		f32 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += s[x];
			}
		}

		return total / (view.width * view.height);
	}


	static f32 average(ViewGray const& view)
	{
		f32 total = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				total += cs::to_channel_f32(s[x]);
			}
		}

		return total / (view.width * view.height);
	}


	template <size_t N>
	static std::array<f32, N> average(ViewCHf32<N> const& view)
	{
		std::array<f32, N> results = { 0 };
		for (u32 i = 0; i < N; ++i) { results[i] = 0.0f; }

		for (u32 y = 0; y < view.height; ++y)
		{
			for (u32 i = 0; i < N; ++i)
			{
				auto s = channel_row_begin(view, y, i);

				for (u32 x = 0; x < view.width; ++x)
				{
					results[i] += s[x];
				}
			}
		}

		for (u32 i = 0; i < N; ++i)
		{
			results[i] /= (view.width * view.height);
		}

		return results;
	}
	

	static cs::RGBf32 average(View const& view)
	{	
		f32 red = 0.0f;
		f32 green = 0.0f;
		f32 blue = 0.0f;

		for (u32 y = 0; y < view.height; ++y)
		{
			auto s = row_begin(view, y);
			for (u32 x = 0; x < view.width; ++x)
			{
				auto p = s[x].rgba;
				red += cs::to_channel_f32(p.red);
				green += cs::to_channel_f32(p.green);
				blue += cs::to_channel_f32(p.blue);
			}
		}

		red /= (view.width * view.height);
		green /= (view.width * view.height);
		blue /= (view.width * view.height);

		return { red, green, blue };
	}


	template <class VIEW>
	static void do_shrink_1D(VIEW const& src, View1f32 const& dst)
	{
		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;
				
				d[x] = average(sub_view(src, r));
			}
		}
	}


	void shrink(View1f32 const& src, View1f32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View3f32 const& src, View3f32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d0 = channel_row_begin(dst, y, 0);
			auto d1 = channel_row_begin(dst, y, 1);
			auto d2 = channel_row_begin(dst, y, 2);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;

				auto avg = average(sub_view(src, r));

				d0[x] = avg[0];
				d1[x] = avg[1];
				d2[x] = avg[2];
			}
		}
	}


	void shrink(ViewGray const& src, View1f32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		do_shrink_1D(src, dst);
	}


	void shrink(View const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src));
		assert(verify(dst));
		assert(dst.width <= src.width);
		assert(dst.height <= src.height);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = rgb_row_begin(dst, y);

			Range2Du32 r{};
			r.y_begin = y * src.height / dst.height;
			r.y_end = r.y_begin + src.height / dst.height;
			for (u32 x = 0; x < dst.width; ++x)
			{
				r.x_begin = x * src.width / dst.width;
				r.x_end = r.x_begin + src.width / dst.width;

				auto avg = average(sub_view(src, r));
				d.R[x] = avg.red;
				d.G[x] = avg.green;
				d.B[x] = avg.blue;				
			}
		}
	}
}
#endif
