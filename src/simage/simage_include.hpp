#pragma once

#include "../defines.hpp"


/* constants, enums */

namespace simage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

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


	class RGBAu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};


	enum class RGB : int
	{
		R = 0, G = 1, B = 2
	};


	enum class RGBA : int
	{
		R = 0, G = 1, B = 2, A = 3
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


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}
}


/* platform image */

namespace simage
{
    typedef union pixel_t
	{
		u8 channels[4] = {};

		u32 value;

		RGBAu8 rgba;

	} Pixel;


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		Pixel p{};
		p.channels[id_cast(RGBA::R)] = r;
		p.channels[id_cast(RGBA::G)] = g;
		p.channels[id_cast(RGBA::B)] = b;
		p.channels[id_cast(RGBA::A)] = a;

		return p;
	}


	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b)
	{
		return to_pixel(r, g, b, 255);
	}


	constexpr inline Pixel to_pixel(u8 value)
	{
		return to_pixel(value, value, value, 255);
	}


	class Image
	{
	public:

		u32 width = 0;
		u32 height = 0;

		Pixel* data = nullptr;
	};


    class ViewRGBAu8
	{
	public:

		Pixel* image_data = 0;
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


    namespace gray
    {
        class Image
		{
		public:

			u32 width;
			u32 height;

			u8* data = nullptr;
		};


		class View
		{
		public:

			u8* image_data = 0;
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
    }
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


	using View4r32 = ViewCHr32<4>;
	using View3r32 = ViewCHr32<3>;
	using View2r32 = ViewCHr32<2>;

	using ViewRGBAr32 = View4r32;
	using ViewRGBr32 = View3r32;

	using ViewHSVr32 = View3r32;


    template <size_t N>
	class PixelCHr32
	{
	public:

		static constexpr u32 n_channels = N;

		r32* channels[N] = {};
	};


	using Pixel4r32 = PixelCHr32<4>;
	using Pixel3r32 = PixelCHr32<3>;
	using Pixel2r32 = PixelCHr32<2>;


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
		r32& red() { return *rgba.R; }
		r32& green() { return *rgba.G; }
		r32& blue() { return *rgba.B; }
		r32& alpha() { return *rgba.A; }
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