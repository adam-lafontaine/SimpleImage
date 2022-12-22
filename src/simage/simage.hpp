#pragma once

#include "simage_platform.hpp"


namespace simage
{
	void make_image(Image& image, u32 width, u32 height);

	void destroy_image(Image& image);

	void make_image(gray::Image& image, u32 width, u32 height);

	void destroy_image(gray::Image& image);
}


/* stb_simage.cpp */

namespace simage
{
	void read_image_from_file(const char* img_path_src, Image& image_dst);

	void read_image_from_file(const char* file_path_src, gray::Image& image_dst);


#ifndef SIMAGE_NO_WRITE

	void write_image(Image const& image_src, const char* file_path_dst);

	void write_image(gray::Image const& image_src, const char* file_path_dst);

#endif // !SIMAGE_NO_WRITE


#ifndef SIMAGE_NO_RESIZE

	void resize_image(Image const& image_src, Image& image_dst);

	void resize_image(gray::Image const& image_src, gray::Image& image_dst);

#endif // !SIMAGE_NO_RESIZE
}


#ifndef SIMAGE_NO_FILESYSTEM

#include <filesystem>

namespace fs = std::filesystem;


namespace simage
{

	inline void read_image_from_file(fs::path const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}


	inline void read_image_from_file(fs::path const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.string().c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}


	inline void write_image(gray::Image const& image_src, fs::path const& file_path_dst)
	{
		write_image(image_src, file_path_dst.string().c_str());
	}

#endif // !SIMAGE_NO_WRITE
	
}

#else

#include <string>

namespace simage
{

	inline void read_image_from_file(std::string const& img_path_src, Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}


	inline void read_image_from_file(std::string const& img_path_src, gray::Image& image_dst)
	{
		return read_image_from_file(img_path_src.c_str(), image_dst);
	}

#ifndef SIMAGE_NO_WRITE

	inline void write_image(Image const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}


	inline void write_image(gray::Image const& image_src, std::string const& file_path_dst)
	{
		write_image(image_src, file_path_dst.c_str());
	}

#endif // !SIMAGE_NO_WRITE
	
}

#endif // !SIMAGE_NO_FILESYSTEM


/* constants, enums */

namespace simage
{
	// platform dependent, endianness
	class RGBAr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
		r32* A;
	};


	class RGBr32p
	{
	public:
		r32* R;
		r32* G;
		r32* B;
	};


	class HSVr32p
	{
	public:
		r32* H;
		r32* S;
		r32* V;
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class HSV : int
	{
		H = 0, S = 1, V = 2
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};
}





/* camera */

namespace simage
{
    class YUV422
	{
	public:
		u8 u;
		u8 y1;
		u8 v;
		u8 y2;
	};


	class YUV2
	{
	public:
		u8 uv;
		u8 y;
	};


	class ImageYUV
	{
	public:
		u32 width;
		u32 height;

		YUV2* data;
	};
}


/* planar */

namespace simage
{
    class View1r32
	{
	public:

		r32* image_data = 0;
		u32 image_width = 0;

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


	class PixelRGBAr32
	{
	public:

		static constexpr u32 n_channels = 4;

		union 
		{
			RGBAr32p rgba;

			r32* channels[4] = {};
		};

		// for_each_xy
		/*r32& red() { return *rgba.R; }
		r32& green() { return *rgba.G; }
		r32& blue() { return *rgba.B; }
		r32& alpha() { return *rgba.A; }*/
	};


	class PixelRGBr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union 
		{
			RGBr32p rgb;

			r32* channels[3] = {};
		};

		// for_each_xy
		/*r32& red() { return *rgb.R; }
		r32& green() { return *rgb.G; }
		r32& blue() { return *rgb.B; }*/
	};


	class PixelHSVr32
	{
	public:

		static constexpr u32 n_channels = 3;

		union
		{
			HSVr32p hsv;

			r32* channels[3] = {};
		};

		/*r32& hue() { return *hsv.H; }
		r32& sat() { return *hsv.S; }
		r32& val() { return *hsv.V; }*/
	};
}


namespace simage
{
	using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;
	using View2r32 = ViewCHr32<2>;

    using Pixel4r32 = PixelCHr32<4>;
	using Pixel3r32 = PixelCHr32<3>;
	using Pixel2r32 = PixelCHr32<2>;

    using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;

	using ViewHSVr32 = View3r32;
}


