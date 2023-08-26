/* make_histograms */

namespace simage
{
namespace hist
{	
	/*inline constexpr u8 to_hist_bin_u8(u8 val, u32 n_bins)
	{
		//return val * n_bins / 256;
		return (val * n_bins) >> 8;
	}*/


	namespace lut
    {
		template <size_t N>
		static constexpr std::array<u8, 256> make_hist_bins()
		{
			static_assert(N <= MAX_HIST_BINS);
			static_assert(MAX_HIST_BINS % N == 0);

			std::array<u8, 256> lut = {};

            for (u32 i = 0; i < 256; ++i)
            {
                lut[i] = (i * N) >> 8;
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
		static constexpr auto hist_bins_256 = make_hist_bins<256>();
    }


	static inline constexpr u8 to_hist_bin_u8(u8 val)
	{
		return lut::hist_bins_256[val];
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
		case 256: return lut::hist_bins_256[val];

		default: return 0;
		}
	}


	static void for_each_rgb(View const& src, std::function<void(u8, u8, u8)> const& rgb_func)
	{
		constexpr u32 PIXEL_STEP = 1;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += PIXEL_STEP)
			{
				auto& rgba = s[x].rgba;

				rgb_func(rgba.red, rgba.green, rgba.blue);
			}
		}
	}


	static void for_each_yuv(ViewYUV const& src, std::function<void(u8, u8, u8)> const& yuv_func)
	{
		constexpr u32 PIXEL_STEP = 1;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				yuv_func(yuv.y1, yuv.u, yuv.v);
				yuv_func(yuv.y2, yuv.u, yuv.v);
			}
		}
	}


	static void for_each_bgr(ViewBGR const& src, std::function<void(u8, u8, u8)> const& rgb_func)
	{
		constexpr u32 PIXEL_STEP = 1;

		for (u32 y = 0; y < src.height; y += PIXEL_STEP)
		{
			auto s = row_begin(src, y);
			for (u32 x = 0; x < src.width; x += PIXEL_STEP)
			{
				auto& bgr = s[x];

				rgb_func(bgr.red, bgr.green, bgr.blue);
			}
		}
	}


	static void make_histograms_from_rgb(View const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;

		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);
			auto lch = lch::u8_from_rgb_u8(red, green, blue);
			auto yuv = yuv::u8_from_rgb_u8(red, green, blue);

			h_rgb.R[to_hist_bin_u8(red)]++;
			h_rgb.G[to_hist_bin_u8(green)]++;
			h_rgb.B[to_hist_bin_u8(blue)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val)]++;

			h_lch.L[to_hist_bin_u8(lch.light)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma)]++;
			h_lch.H[to_hist_bin_u8(lch.hue)]++;

			h_yuv.Y[to_hist_bin_u8(yuv.y)]++;
			h_yuv.U[to_hist_bin_u8(yuv.u)]++;
			h_yuv.V[to_hist_bin_u8(yuv.v)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

		u32 n_bins = MAX_HIST_BINS;

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_yuv(ViewYUV const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			h_rgb.R[to_hist_bin_u8(rgba.red)]++;
			h_rgb.G[to_hist_bin_u8(rgba.green)]++;
			h_rgb.B[to_hist_bin_u8(rgba.blue)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val)]++;

			h_lch.L[to_hist_bin_u8(lch.light)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma)]++;
			h_lch.H[to_hist_bin_u8(lch.hue)]++;

			h_yuv.Y[to_hist_bin_u8(yuv_y)]++;
			h_yuv.U[to_hist_bin_u8(yuv_u)]++;
			h_yuv.V[to_hist_bin_u8(yuv_v)]++;
		};

		f32 total = 0.0f;

		for_each_yuv(src, update_bins);

		u32 n_bins = MAX_HIST_BINS;

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}


	static void make_histograms_from_bgr(ViewBGR const& src, Histogram12f32& dst)
	{
		auto& h_rgb = dst.rgb;
		auto& h_hsv = dst.hsv;
		auto& h_lch = dst.lch;
		auto& h_yuv = dst.yuv;

		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue)
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);
			auto lch = lch::u8_from_rgb_u8(red, green, blue);
			auto yuv = yuv::u8_from_rgb_u8(red, green, blue);

			h_rgb.R[to_hist_bin_u8(red)]++;
			h_rgb.G[to_hist_bin_u8(green)]++;
			h_rgb.B[to_hist_bin_u8(blue)]++;

			if (hsv.sat)
			{
				h_hsv.H[to_hist_bin_u8(hsv.hue)]++;
			}

			h_hsv.S[to_hist_bin_u8(hsv.sat)]++;
			h_hsv.V[to_hist_bin_u8(hsv.val)]++;

			h_lch.L[to_hist_bin_u8(lch.light)]++;
			h_lch.C[to_hist_bin_u8(lch.chroma)]++;
			h_lch.H[to_hist_bin_u8(lch.hue)]++;

			h_yuv.Y[to_hist_bin_u8(yuv.y)]++;
			h_yuv.U[to_hist_bin_u8(yuv.u)]++;
			h_yuv.V[to_hist_bin_u8(yuv.v)]++;

			total++;
		};

		for_each_bgr(src, update_bins);

		u32 n_bins = MAX_HIST_BINS;

		for (u32 i = 0; i < 12; ++i)
		{
			for (u32 bin = 0; bin < n_bins; ++bin)
			{
				dst.list[i][bin] /= total;
			}
		}
	}
	
	
	void make_histograms(View const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		static_assert(N_BINS <= MAX_HIST_BINS);
		static_assert(MAX_HIST_BINS % N_BINS == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_rgb(src, dst);
	}


	void make_histograms(ViewYUV const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		static_assert(N_BINS <= MAX_HIST_BINS);
		static_assert(MAX_HIST_BINS % N_BINS == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_yuv(src, dst);
	}


	void make_histograms(ViewBGR const& src, Histogram12f32& dst)
	{
		static_assert(MAX_HIST_BINS == 256);
		static_assert(N_BINS <= MAX_HIST_BINS);
		static_assert(MAX_HIST_BINS % N_BINS == 0);

		dst.rgb = { 0 };
		dst.hsv = { 0 };
		dst.lch = { 0 };
		dst.yuv = { 0 };

		make_histograms_from_bgr(src, dst);
	}


	void make_histograms(View const& src, HistRGBf32& dst, u32 n_bins)
	{
		static_assert(MAX_HIST_BINS == 256);
		assert(n_bins <= MAX_HIST_BINS);
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			dst.R[to_hist_bin_u8(red, n_bins)]++;
			dst.G[to_hist_bin_u8(green, n_bins)]++;
			dst.B[to_hist_bin_u8(blue, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto hsv = hsv::u8_from_rgb_u8(red, green, blue);

			if (hsv.sat)
			{
				dst.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			dst.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			dst.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };
		f32 total = 0.0f;

		auto const update_bins = [&](u8 red, u8 green, u8 blue) 
		{
			auto lch = lch::u8_from_rgb_u8(red, green, blue);

			dst.L[to_hist_bin_u8(lch.light, n_bins)]++;
			dst.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			dst.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			total++;
		};

		for_each_rgb(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			dst.Y[to_hist_bin_u8(yuv_y, n_bins)]++;
			dst.U[to_hist_bin_u8(yuv_u, n_bins)]++;
			dst.V[to_hist_bin_u8(yuv_v, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);

			dst.R[to_hist_bin_u8(rgba.red, n_bins)]++;
			dst.G[to_hist_bin_u8(rgba.green, n_bins)]++;
			dst.B[to_hist_bin_u8(rgba.blue, n_bins)]++;

			total++;
		};		

		for_each_yuv(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto hsv = hsv::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			if (hsv.sat)
			{
				dst.H[to_hist_bin_u8(hsv.hue, n_bins)]++;
			}

			dst.S[to_hist_bin_u8(hsv.sat, n_bins)]++;
			dst.V[to_hist_bin_u8(hsv.val, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

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
		assert(MAX_HIST_BINS % n_bins == 0);

		dst = { 0 };

		f32 total = 0.0f;

		auto const update_bins = [&](u8 yuv_y, u8 yuv_u, u8 yuv_v)
		{
			auto rgba = yuv::u8_to_rgb_u8(yuv_y, yuv_u, yuv_v);
			auto lch = lch::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);

			dst.L[to_hist_bin_u8(lch.light, n_bins)]++;
			dst.C[to_hist_bin_u8(lch.chroma, n_bins)]++;
			dst.H[to_hist_bin_u8(lch.hue, n_bins)]++;

			total++;
		};

		for_each_yuv(src, update_bins);

		for (u32 bin = 0; bin < n_bins; ++bin)
		{
			dst.L[bin] /= total;
			dst.C[bin] /= total;
			dst.H[bin] /= total;
		}
	}
}
}

