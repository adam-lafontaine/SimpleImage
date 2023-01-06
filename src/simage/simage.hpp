#pragma once

#include "simage_platform.hpp"
#include "../util/memory_buffer.hpp"

#include <array>

namespace mb = memory_buffer;


namespace simage
{
	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
	};


	enum class HSV : int
	{
		H = 0, S = 1, V = 2
	};


	enum class YUV : int
	{
		Y = 0, U = 1, V = 2
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}

}


/* view */

namespace simage
{
	template <size_t N>
	class ViewCHr32
	{
	public:

		u32 image_width = 0;

		r32* image_channel_data[N] = {};

		union
		{
			Range2Du32 range = {};

			struct
			{
				u32 x_begin;
				u32 x_end;
				u32 y_begin;
				u32 y_end;
			};
		};

		u32 width = 0;
		u32 height = 0;
	};


    template <size_t N>
	class PixelCHr32
	{
	public:

		static constexpr u32 n_channels = N;

		r32* channels[N] = {};
	};


	using View1r32 = MatrixView<r32>;

    using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;
	using View2r32 = ViewCHr32<2>;

    using Pixel4r32 = PixelCHr32<4>;
	using Pixel3r32 = PixelCHr32<3>;
	using Pixel2r32 = PixelCHr32<2>;
}


/* make_view */

namespace simage
{
	using Buffer32 = MemoryBuffer<r32>;

	using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;
	using ViewHSVr32 = View3r32;


	View1r32 make_view_1(u32 width, u32 height, Buffer32& buffer);

	View2r32 make_view_2(u32 width, u32 height, Buffer32& buffer);

	View3r32 make_view_3(u32 width, u32 height, Buffer32& buffer);

	View4r32 make_view_4(u32 width, u32 height, Buffer32& buffer);
}


/* map */

namespace simage
{
	void map(ViewGray const& src, View1r32 const& dst);

	void map(View1r32 const& src, ViewGray const& dst);


	void map(ViewGray const& src, View const& dst);
}


/* map_rgb */

namespace simage
{	
	void map_rgba(View const& src, ViewRGBAr32 const& dst);

	void map_rgba(ViewRGBAr32 const& src, View const& dst);
	
	void map_rgb(View const& src, ViewRGBr32 const& dst);

	void map_rgb(ViewRGBr32 const& src, View const& dst);

	void map_rgb(View1r32 const& src, View const& dst);


	/*inline ViewRGBAr32 make_rgba_view(View const& src, Buffer32& buffer)
	{
		auto view = make_view_4(src.width, src.height, buffer);
		map_rgb(src, view);

		return view;
	}


	inline ViewRGBr32 make_rgb_view(View const& src, Buffer32& buffer)
	{
		auto view = make_view_3(src.width, src.height, buffer);
		map_rgb(src, view);

		return view;
	}*/
}


/* map_rgb_hsv */

namespace simage
{
	void map_rgb_hsv(View const& src, ViewHSVr32 const& dst);

	void map_hsv_rgb(ViewHSVr32 const& src, View const& dst);


	void map_rgb_hsv(ViewRGBr32 const& src, ViewHSVr32 const& dst);	

	void map_hsv_rgb(ViewHSVr32 const& src, ViewRGBr32 const& dst);
}


/* map_yuv_rgb */

namespace simage
{
	void map_yuv_rgb(ViewYUV const& src, ViewRGBr32 const& dst);

	void map_yuv_rgb(ViewYUV const& src, View const& dst);
}


/* sub_view */

namespace simage
{
	View4r32 sub_view(View4r32 const& view, Range2Du32 const& range);

	View3r32 sub_view(View3r32 const& view, Range2Du32 const& range);

	View2r32 sub_view(View2r32 const& view, Range2Du32 const& range);

	View1r32 sub_view(View1r32 const& view, Range2Du32 const& range);
}


/* select_channel */

namespace simage
{
	View1r32 select_channel(ViewRGBAr32 const& view, RGBA channel);

	View1r32 select_channel(ViewRGBr32 const& view, RGB channel);

	View1r32 select_channel(ViewHSVr32 const& view, HSV channel);

	View1r32 select_channel(View2r32 const& view, GA channel);

	View1r32 select_channel(View2r32 const& view, XY channel);


	ViewRGBr32 select_rgb(ViewRGBAr32 const& view);
}


/* fill */

namespace simage
{
	void fill(View const& view, Pixel color);

	void fill(ViewGray const& view, u8 gray);

	void fill(View4r32 const& view, Pixel color);

	void fill(View3r32 const& view, Pixel color);

	void fill(View1r32 const& view, u8 gray);
}


/* shrink */

namespace simage
{
	void shrink(View1r32 const& src, View1r32 const& dst);

	void shrink(View3r32 const& src, View3r32 const& dst);

	void shrink(ViewGray const& src, View1r32 const& dst);

	void shrink(View const& src, ViewRGBr32 const& dst);
}


/* histogram */

namespace simage
{
	constexpr u32 MAX_HIST_BINS = 256;


	class HistRGBr32
	{
	public:
		r32 R[MAX_HIST_BINS];
		r32 G[MAX_HIST_BINS];
		r32 B[MAX_HIST_BINS];
	};	


	class HistHSVr32
	{
	public:
		r32 H[MAX_HIST_BINS];
		r32 S[MAX_HIST_BINS];
		r32 V[MAX_HIST_BINS];
	};


	class HistYUVr32
	{
	public:
		r32 Y[MAX_HIST_BINS];
		r32 U[MAX_HIST_BINS];
		r32 V[MAX_HIST_BINS];
	};


	class Histogram9r32
	{
	public:
		
		union
		{
			struct
			{
				HistRGBr32 rgb;
				HistHSVr32 hsv;
				HistYUVr32 yuv;
			};

			r32 list[9][MAX_HIST_BINS] = { 0 };
		};

		u32 n_bins = MAX_HIST_BINS;
	};


	void make_histograms(View const& src, Histogram9r32& dst);

	void make_histograms(ViewYUV const& src, Histogram9r32& dst);
}


/* stb_simage.cpp */

namespace simage
{
	bool read_image_from_file(const char* img_path_src, Image& image_dst);

	bool read_image_from_file(const char* file_path_src, ImageGray& image_dst);


#ifndef SIMAGE_NO_WRITE

	bool write_image(Image const& image_src, const char* file_path_dst);

	bool write_image(ImageGray const& image_src, const char* file_path_dst);

#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE

	bool resize_image(Image const& image_src, Image& image_dst);

	bool resize_image(ImageGray const& image_src, ImageGray& image_dst);

#endif // !SIMAGE_NO_RESIZE
}


#ifndef SIMAGE_NO_FILESYSTEM

#include <filesystem>


namespace simage
{
	using path_t = std::filesystem::path;


	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}


	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}


	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.string().c_str());
	}

#endif // !SIMAGE_NO_WRITE
	
}

#else

#include <string>

namespace simage
{
	using path_t = std::string;

	inline bool read_image_from_file(path_t const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}


	inline bool read_image_from_file(path_t const& img_path_src, ImageGray& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline bool write_image(Image const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}


	inline bool write_image(ImageGray const& image_src, path_t const& file_path_dst)
	{
		return write_image(image_src, file_path_dst.c_str());
	}

#endif // !SIMAGE_NO_WRITE
	
}

#endif // !SIMAGE_NO_FILESYSTEM
