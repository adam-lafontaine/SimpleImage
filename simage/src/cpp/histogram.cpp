/* make_histograms */

namespace simage
{
namespace hist
{	
	namespace lut
    {
		template <size_t N>
		static constexpr std::array<u8, 256> make_hist_bins()
		{
			static_assert(N <= MAX_HIST_BINS);
			static_assert(MAX_HIST_BINS % N == 0);

			std::array<u8, 256> lut = {};

            for (u32 i = 0; i < 256u; ++i)
            {
                lut[i] = (i * (u32)N) >> 8;
            }

            return lut;
		}
		

        static constexpr auto hist_bins_2 = make_hist_bins<2>();
		static constexpr auto hist_bins_4 = make_hist_bins<4>();
		static constexpr auto hist_bins_8 = make_hist_bins<8>();
		static constexpr auto hist_bins_16 = make_hist_bins<16>();
		static constexpr auto hist_bins_32 = make_hist_bins<32>();
		static constexpr auto hist_bins_64 = make_hist_bins<64>();
		static constexpr auto hist_bins_128 = make_hist_bins<128>();
    }


	static inline constexpr u8 to_hist_bin_u8(u8 val, u32 n_bins)
	{
		switch (n_bins)
		{
		case 2: return lut::hist_bins_2[val];
		case 4: return lut::hist_bins_4[val];
		case 8: return lut::hist_bins_8[val];
		case 16: return lut::hist_bins_16[val];
		case 32: return lut::hist_bins_32[val];
		case 64: return lut::hist_bins_64[val];
		case 128: return lut::hist_bins_128[val];
		case 256: return val;

		default: return 0;
		}
	}
	

	static void update_counts(u8 red, u8 green, u8 blue, HistRGBf32& dst, u32 n_bins)
	{
		dst.R[to_hist_bin_u8(red, n_bins)]++;
		dst.G[to_hist_bin_u8(green, n_bins)]++;
		dst.B[to_hist_bin_u8(blue, n_bins)]++;
	}


	static void update_counts(u8 yuv_y, u8 yuv_u, u8 yuv_v, HistYUVf32& dst, u32 n_bins)
	{
		dst.Y[to_hist_bin_u8(yuv_y, n_bins)]++;
		dst.U[to_hist_bin_u8(yuv_u, n_bins)]++;
		dst.V[to_hist_bin_u8(yuv_v, n_bins)]++;
	}


	static void update_counts(u8 hue, u8 sat, u8 val, HistHSVf32& dst, u32 n_bins)
	{
		if (sat)
		{
			dst.H[to_hist_bin_u8(hue, n_bins)]++;
		}

		dst.S[to_hist_bin_u8(sat, n_bins)]++;
		dst.V[to_hist_bin_u8(val, n_bins)]++;
	}


	static void update_counts(u8 light, u8 chroma, u8 hue, HistLCHf32& dst, u32 n_bins)
	{
		dst.L[to_hist_bin_u8(light, n_bins)]++;
		dst.C[to_hist_bin_u8(chroma, n_bins)]++;
		dst.H[to_hist_bin_u8(hue, n_bins)]++;
	}	


	static void update_from_rgb(u8 red, u8 green, u8 blue, Histogram12f32& dst, u32 n_bins)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;

		auto hsv = hsv::u8_from_rgb_u8(red, green, blue);
		auto lch = lch::u8_from_rgb_u8(red, green, blue);
		auto yuv = yuv::u8_from_rgb_u8(red, green, blue);

		update_counts(red, green, blue, dst.rgb, n_bins);
		update_counts(hsv.hue, hsv.sat, hsv.val, dst.hsv, n_bins);
		update_counts(lch.light, lch.chroma, lch.hue, dst.lch, n_bins);
		update_counts(yuv.y, yuv.u, yuv.v, dst.yuv, n_bins);
	}


	static void update_from_yuv(u8 yuv_y, u8 yuv_u, u8 yuv_v, Histogram12f32& dst, u32 n_bins)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;

		auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
		auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
		auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

		update_counts(rgba.red, rgba.green, rgba.blue, dst.rgb, n_bins);
		update_counts(hsv.hue, hsv.sat, hsv.val, dst.hsv, n_bins);
		update_counts(lch.light, lch.chroma, lch.hue, dst.lch, n_bins);
		update_counts(yuv_y, yuv_u, yuv_v, dst.yuv, n_bins);
	}


	static void normalize_counts(Histogram12f32& dst, u32 n_bins, u32 count_total)
	{
		auto total = (f32)count_total;

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_view_rgba(View const& src, Histogram12f32& dst, u32 n_bins)
	{
		u32 len = src.width * src.height;

		auto s = row_begin(src, 0);

		for (u32 i = 0; i < len; ++i)
		{
			auto rgba = s[i].rgba;

			update_from_rgb(rgba.red, rgba.green, rgba.blue, dst, n_bins);
		}

		normalize_counts(dst, n_bins, len);
	}


	static void make_histograms_from_view_yuv(ViewYUV const& src, Histogram12f32& dst, u32 n_bins)
	{
		u32 len = src.width * src.height;

		auto src422 = (YUYVu8*)row_begin(src, 0);

        for (u32 i422 = 0; i422 < len / 2; ++i422)
		{
			auto yuv = src422[i422];

			update_from_yuv(yuv.y1, yuv.u, yuv.v, dst, n_bins);
			update_from_yuv(yuv.y2, yuv.u, yuv.v, dst, n_bins);
		}

		normalize_counts(dst, n_bins, len);
	}


	static void make_histograms_from_view_bgr(ViewBGR const& src, Histogram12f32& dst, u32 n_bins)
	{
		u32 len = src.width * src.height;

		auto s = row_begin(src, 0);

		for (u32 i = 0; i < len; ++i)
		{
			auto bgr = s[i];

			update_from_rgb(bgr.red, bgr.green, bgr.blue, dst, n_bins);
		}

		normalize_counts(dst, n_bins, len);
	}
	
	
	void make_histograms(View const& src, Histogram12f32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		if (is_1d(src))
		{
			make_histograms_from_view_rgba(src, dst, n_bins);
		}
		else
		{
			// TODO
		}
	}


	void make_histograms(ViewYUV const& src, Histogram12f32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		if (is_1d(src))
		{
			make_histograms_from_view_yuv(src, dst, n_bins);
		}
		else
		{
			// TODO
		}
	}


	void make_histograms(ViewBGR const& src, Histogram12f32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		if (is_1d(src))
		{
			make_histograms_from_view_bgr(src, dst, n_bins);
		}
		else
		{
			// TODO
		}
	}


	void make_histograms(View const& src, HistRGBf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto s = row_begin(src, 0);
			for (u32 i = 0; i < len; ++i)
			{
				auto rgba = s[i].rgba;
				update_counts(rgba.red, rgba.green, rgba.blue, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.R[bin] /= total;
			dst.G[bin] /= total;
			dst.B[bin] /= total;
		}
	}


	void make_histograms(View const& src, HistHSVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto s = row_begin(src, 0);
			for (u32 i = 0; i < len; ++i)
			{
				auto rgba = s[i].rgba;
				auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				update_counts(hsv.hue, hsv.sat, hsv.val, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.H[bin] /= total;
			dst.S[bin] /= total;
			dst.V[bin] /= total;
		}
	}


	void make_histograms(View const& src, HistLCHf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto s = row_begin(src, 0);
			for (u32 i = 0; i < len; ++i)
			{
				auto rgba = s[i].rgba;
				auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				update_counts(lch.light, lch.chroma, lch.hue, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.L[bin] /= total;
			dst.C[bin] /= total;
			dst.H[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistYUVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto src422 = (YUYVu8*)row_begin(src, 0);

			for (u32 i422 = 0; i422 < len / 2; ++i422)
			{
				auto yuv = src422[i422];
				update_counts(yuv.y1, yuv.u, yuv.v, dst, n_bins);
				update_counts(yuv.y2, yuv.u, yuv.v, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.Y[bin] /= total;
			dst.U[bin] /= total;
			dst.V[bin] /= total;
		}
	}
	
	
	void make_histograms(ViewYUV const& src, HistRGBf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto src422 = (YUYVu8*)row_begin(src, 0);

			for (u32 i422 = 0; i422 < len / 2; ++i422)
			{
				auto yuv = src422[i422];
				auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
				update_counts(rgb.red, rgb.green, rgb.blue, dst, n_bins);			
				
				rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
				update_counts(rgb.red, rgb.green, rgb.blue, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.R[bin] /= total;
			dst.G[bin] /= total;
			dst.B[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistHSVf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto src422 = (YUYVu8*)row_begin(src, 0);

			for (u32 i422 = 0; i422 < len / 2; ++i422)
			{
				auto yuv = src422[i422];
				auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
				auto hsv = hsv::u8_from_rgb_u8(rgb.red, rgb.green, rgb.blue);
				update_counts(hsv.hue, hsv.sat, hsv.val, dst, n_bins);

				rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
				hsv = hsv::u8_from_rgb_u8(rgb.red, rgb.green, rgb.blue);
				update_counts(hsv.hue, hsv.sat, hsv.val, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.H[bin] /= total;
			dst.S[bin] /= total;
			dst.V[bin] /= total;
		}
	}


	void make_histograms(ViewYUV const& src, HistLCHf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(to_hist_bin_u8(n_bins - 1, n_bins) != 0);

		dst = { 0 };

		u32 len = src.width * src.height;

		if (is_1d(src))
		{
			auto src422 = (YUYVu8*)row_begin(src, 0);

			for (u32 i422 = 0; i422 < len / 2; ++i422)
			{
				auto yuv = src422[i422];
				auto rgb = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
				auto lch = lch::u8_from_rgb_u8(rgb.red, rgb.green, rgb.blue);
				update_counts(lch.light, lch.chroma, lch.hue, dst, n_bins);

				rgb = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
				lch = lch::u8_from_rgb_u8(rgb.red, rgb.green, rgb.blue);
				update_counts(lch.light, lch.chroma, lch.hue, dst, n_bins);
			}
		}
		else
		{
			// TODO
		}

		auto total = (f32)len;

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.L[bin] /= total;
			dst.C[bin] /= total;
			dst.H[bin] /= total;
		}
	}
}
}

