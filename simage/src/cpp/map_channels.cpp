/* map */

namespace simage
{
	void map_rgba(ViewGray const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				RGBAu8 gray = { s[x], s[x], s[x], 255 };

				d[x].rgba = gray;
			}
		}
	}


	void map_rgba(ViewYUV const& src, View const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s2 = row_begin(src, y);
			auto s422 = (YUV422u8*)s2;
			auto d = row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgba = yuv::u8_to_rgb_u8(yuv.y1, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;

				++x;
				rgba = yuv::u8_to_rgb_u8(yuv.y2, yuv.u, yuv.v);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.red = 255;
			}
		}
	}


	void map_rgba(ViewBGR const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto& rgba = d[x].rgba;
				rgba.red = s[x].red;
				rgba.green = s[x].green;
				rgba.blue = s[x].blue;
				rgba.alpha = 255;
			}
		}
	}	


	void map_rgba(ViewRGB const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto& rgba = d[x].rgba;
				rgba.red = s[x].red;
				rgba.green = s[x].green;
				rgba.blue = s[x].blue;
				rgba.alpha = 255;
			}
		}
	}


	void map_gray(View const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				d[x] = gray::u8_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
			}
		}
	}


	void map_gray(ViewYUV const& src, ViewGray const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = s[x].y;
			}
		}
	}
}


/* map */

namespace simage
{
	static void map_channel_row_u8_to_f32(u8* src, f32* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_f32(src[x]);
		}
	}


	static void map_channel_row_f32_to_u8(f32* src, u8* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x] = cs::to_channel_u8(src[x]);
		}
	}


	void map_gray(View1u8 const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_u8_to_f32(s, d, src.width);
		}
	}
	

	void map_gray(View1f32 const& src, View1u8 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);
			map_channel_row_f32_to_u8(s, d, src.width);
		}
	}


	void map_gray(ViewYUV const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = cs::to_channel_f32(s[x].y);
			}
		}
	}


	void map_gray(View1f32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto s = row_begin(src, y);			

			for (u32 x = 0; x < src.width; ++x)
			{
				auto const gray = cs::to_channel_u8(s[x]);

				d[x].rgba.red = gray;
				d[x].rgba.green = gray;
				d[x].rgba.blue = gray;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_gray(View const& src, View1f32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto d = row_begin(dst, y);
			auto& s = row_begin(src, y)->rgba;			

			for (u32 x = 0; x < src.width; ++x)
			{
				d[x] = gray::f32_from_rgb_f32(s.red, s.green, s.blue);
			}
		}
	}
}


/* map_rgb */

namespace simage
{	
	static void map_rgba_row_u8_to_f32(Pixel* src, RGBAf32p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_f32(src[x].rgba.red);
			dst.G[x] = cs::to_channel_f32(src[x].rgba.green);
			dst.B[x] = cs::to_channel_f32(src[x].rgba.blue);
			dst.A[x] = cs::to_channel_f32(src[x].rgba.alpha);
		}
	}


	static void map_rgba_row_f32_to_u8(RGBAf32p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = cs::to_channel_u8(src.A[x]);
		}
	}


	static void map_rgb_row_u8_to_f32(Pixel* src, RGBf32p const& dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst.R[x] = cs::to_channel_f32(src[x].rgba.red);
			dst.G[x] = cs::to_channel_f32(src[x].rgba.green);
			dst.B[x] = cs::to_channel_f32(src[x].rgba.blue);
		}
	}


	static void map_rgb_row_f32_to_u8(RGBf32p const& src, Pixel* dst, u32 width)
	{
		for (u32 x = 0; x < width; ++x)
		{
			dst[x].rgba.red = cs::to_channel_u8(src.R[x]);
			dst[x].rgba.green = cs::to_channel_u8(src.G[x]);
			dst[x].rgba.blue = cs::to_channel_u8(src.B[x]);
			dst[x].rgba.alpha = 255;
		}
	}


	void map_rgba(View const& src, ViewRGBAf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgba_row_begin(dst, y);

			map_rgba_row_u8_to_f32(s, d, src.width);
		}
	}


	void map_rgba(ViewRGBAf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgba_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgba_row_f32_to_u8(s, d, src.width);
		}
	}

	
	void map_rgb(View const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			map_rgb_row_u8_to_f32(s, d, src.width);
		}
	}


	void map_rgb(ViewRGBf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = row_begin(dst, y);

			map_rgb_row_f32_to_u8(s, d, src.width);
		}
	}
}


/* map_hsv */

namespace simage
{	
	void map_rgb_hsv(View const& src, ViewHSVf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto hsv = hsv::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		}
	}


	void map_hsv_rgb(ViewHSVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y) 
		{
			auto s = hsv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::f32_to_rgb_u8(s.H[x], s.S[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_rgb_hsv(ViewRGBf32 const& src, ViewHSVf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = hsv_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto hsv = hsv::f32_from_rgb_f32(r, g, b);
				d.H[x] = hsv.hue;
				d.S[x] = hsv.sat;
				d.V[x] = hsv.val;
			}
		}
	}


	void map_hsv_rgb(ViewHSVf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = hsv_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = hsv::f32_to_rgb_f32(s.H[x], s.S[x], s.V[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}
}


/* map_lch */

namespace simage
{
	void map_rgb_lch(View const& src, ViewLCHf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = s[x].rgba;
				auto lch = lch::f32_from_rgb_u8(rgba.red, rgba.green, rgba.blue);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		}
	}


	void map_lch_rgb(ViewLCHf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = lch_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgba = lch::f32_to_rgb_u8(s.L[x], s.C[x], s.H[x]);

				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_rgb_lch(ViewRGBf32 const& src, ViewLCHf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = rgb_row_begin(src, y);
			auto d = lch_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto r = s.R[x];
				auto g = s.G[x];
				auto b = s.B[x];

				auto lch = lch::f32_from_rgb_f32(r, g, b);
				d.L[x] = lch.light;
				d.C[x] = lch.chroma;
				d.H[x] = lch.hue;
			}
		}
	}


	void map_lch_rgb(ViewLCHf32 const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = lch_row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = lch::f32_to_rgb_f32(s.L[x], s.C[x], s.H[x]);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}
}


/* map_yuv */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{
		assert(verify(src, dst));
		assert(src.width % 2 == 0);
		static_assert(sizeof(YUV2u8) == 2);

		for (u32 y = 0; y < src.height; ++y)
		{
			auto s422 = (YUV422u8*)row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x422 = 0; x422 < src.width / 2; ++x422)
			{
				auto yuv = s422[x422];

				auto x = 2 * x422;
				auto rgb = yuv::u8_to_rgb_f32(yuv.y1, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;

				++x;
				rgb = rgb = yuv::u8_to_rgb_f32(yuv.y2, yuv.u, yuv.v);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}

/*
	void mipmap_yuv_rgb(ViewYUV const& src, ViewRGBf32 const& dst)
	{		
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d) 
		{
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		for (u32 y = 0; y < src.height; ++y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgb = yuv::u8_to_rgb_f32(y_avg, u_avg, v_avg);
				d.R[x] = rgb.red;
				d.G[x] = rgb.green;
				d.B[x] = rgb.blue;
			}
		}
	}*/


	void map_yuv_rgb2(ViewYUV const& src, View const& dst)
	{
		static_assert(sizeof(YUV2u8) == 2);
		assert(verify(src));
		assert(verify(dst));
		assert(src.width % 2 == 0);
		assert(dst.width == src.width / 2);
		assert(dst.height == src.height / 2);

		constexpr auto avg4 = [](u8 a, u8 b, u8 c, u8 d)
		{
			auto val = 0.25f * ((f32)a + b + c + d);
			return (u8)(u32)(val + 0.5f);
		};

		constexpr auto avg2 = [](u8 a, u8 b)
		{
			auto val = 0.5f * ((f32)a + b);
			return (u8)(u32)(val + 0.5f);
		};

		for (u32 y = 0; y < src.height; ++y)
		{
			auto src_y1 = y * 2;
			auto src_y2 = src_y1 + 1;

			auto s1 = (YUV422u8*)row_begin(src, src_y1);
			auto s2 = (YUV422u8*)row_begin(src, src_y2);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < dst.width; ++x)
			{
				auto yuv1 = s1[x];
				auto yuv2 = s2[x];
				u8 y_avg = avg4(yuv1.y1, yuv1.y2, yuv2.y1, yuv2.y2);
				u8 u_avg = avg2(yuv1.u, yuv2.u);
				u8 v_avg = avg2(yuv1.v, yuv2.v);

				auto rgba = yuv::u8_to_rgb_u8(y_avg, u_avg, v_avg);
				d[x].rgba.red = rgba.red;
				d[x].rgba.green = rgba.green;
				d[x].rgba.blue = rgba.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}


	void map_yuv_rgb(ViewYUVf32 const& src, View const& dst)
	{
		assert(verify(src, dst));

		for (u32 y = 0; y < src.height; ++y) 
		{
			auto s = yuv_row_begin(src, y);
			auto d = row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				auto rgb = yuv::f32_to_rgb_u8(s.Y[x], s.U[x], s.V[x]);

				d[x].rgba.red = rgb.red;
				d[x].rgba.green = rgb.green;
				d[x].rgba.blue = rgb.blue;
				d[x].rgba.alpha = 255;
			}
		}
	}
}


/* map bgr */

namespace simage
{
	void map_bgr_rgb(ViewBGR const& src, ViewRGBf32 const& dst)
	{		
		for (u32 y = 0; y < src.height; ++y)
		{
			auto s = row_begin(src, y);
			auto d = rgb_row_begin(dst, y);

			for (u32 x = 0; x < src.width; ++x)
			{
				d.R[x] = cs::to_channel_f32(s[x].red);
				d.G[x] = cs::to_channel_f32(s[x].green);
				d.B[x] = cs::to_channel_f32(s[x].blue);
			}
		}
	}
}