#pragma once

#include "defines.hpp"



namespace simage
{
	constexpr auto RGB_CHANNELS = 3u;
	constexpr auto RGBA_CHANNELS = 4u;

	class RGBAu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
		u8 alpha;
	};
    
}


/* platform (interleaved) image */

namespace simage
{
    typedef union pixel_t
	{
		u8 channels[4];

		u32 value;

		RGBAu8 rgba = {};

	} Pixel;


	template <typename T>
	class MatrixView2D
	{
	public:
		T* data = nullptr;
		u32 width = 0;
		u32 height = 0;
	};


	template <typename T>
    class SubMatrixView2D
	{
	public:

		T* matrix_data_ = 0;
		u32 matrix_width = 0;

		u32 width = 0;
		u32 height = 0;

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
	};


	using Image = Matrix2D<Pixel>;
	using View = MatrixView2D<Pixel>;
	using SubView = SubMatrixView2D<Pixel>;

	using ImageGray = Matrix2D<u8>;
	using ViewGray = MatrixView2D<u8>;
	using SubViewGray = SubMatrixView2D<u8>;
}


/* channel view */

namespace simage
{
	template <typename T, size_t N>
	class ChannelMatrix2D
	{
	public:

		T* channel_data[N] = {};		

		u32 width = 0;
		u32 height = 0;
	};


	template <typename T>
	using View4 = ChannelMatrix2D<T, 4>;

	template <typename T>
	using View3 = ChannelMatrix2D<T, 3>;

	template <typename T>
	using View2 = ChannelMatrix2D<T, 2>;

	template <typename T>
	using View1 = MatrixView2D<T>;

	using View1u8 = ViewGray;

	template <typename T>
	using SubView1 = SubMatrixView2D<T>;

	using SubView1u8 = SubViewGray;


	using View4f32 = View4<f32>;
	using View3f32 = View3<f32>;
	using View2f32 = View2<f32>;
	using View1f32 = View1<f32>;	

	using ViewRGBAf32 = View4f32;
	using ViewRGBf32 = View3f32;
	using ViewHSVf32 = View3f32;
	using ViewLCHf32 = View3f32;

	using SubView1f32 = SubView1<f32>;
}


/* camera types */

namespace simage
{
	class YUYVu8
	{
	public:		
		u8 y1;
		u8 u;		
		u8 y2;
		u8 v;
	};


	class YUV2u8
	{
	public:		
		u8 y;
		u8 uv;
	};


	class UYVYu8
	{
	public:
		u8 u;
		u8 y1;
		u8 v;
		u8 y2;
	};


	class UVY2u8
	{
	public:
		u8 uv;
		u8 y;
	};


	class BGRu8
	{
	public:
		u8 blue;
		u8 green;
		u8 red;
	};


	class RGBu8
	{
	public:
		u8 red;
		u8 green;
		u8 blue;
	};

#if IS_LITTLE_ENDIAN

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


	enum class LCH : int
	{
		L = 0, C = 1, H = 2
	};


	enum class GA : int
	{
		G = 0, A = 1
	};


	enum class XY : int
	{
		X = 0, Y = 1
	};

#else

	enum class RGB : int
	{
		R = 2, G = 1, B = 0
	};


	enum class RGBA : int
	{
		R = 3, G = 2, B = 1, A = 0
	};


	enum class HSV : int
	{
		H = 2, S = 1, V = 0
	};


	enum class LCH : int
	{
		L = 2, C = 1, H = 0
	};


	enum class GA : int
	{
		G = 1, A = 0
	};


	enum class XY : int
	{
		X = 1, Y = 0
	};

#endif	


	template <typename T>
	constexpr inline int id_cast(T channel)
	{
		return static_cast<int>(channel);
	}


	using ImageYUV = Matrix2D<YUV2u8>;
	using ViewYUV = MatrixView2D<YUV2u8>;
	using SubViewYUV = SubMatrixView2D<YUV2u8>;

	using ImageUVY = Matrix2D<UVY2u8>;
	using ViewUVY = MatrixView2D<UVY2u8>;
	using SubViewUVY = SubMatrixView2D<UVY2u8>;

	using ImageBGR = Matrix2D<BGRu8>;
	using ViewBGR = MatrixView2D<BGRu8>;
	using SubViewBGR = SubMatrixView2D<BGRu8>;

	using ImageRGB = Matrix2D<RGBu8>;
	using ViewRGB = MatrixView2D<RGBu8>;
	using SubViewRGB = SubMatrixView2D<RGBu8>;
}


/* to_pixel */

namespace simage
{
	constexpr inline Pixel to_pixel(u8 r, u8 g, u8 b, u8 a)
	{
		Pixel p{};
		p.rgba.red = r;
		p.rgba.green = g;
		p.rgba.blue = b;
		p.rgba.alpha = a;

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
}


#ifndef SIMAGE_NO_CUDA

/* device view */

namespace simage
{
	template <typename T>
	class DeviceMatrix2D
	{
	public:
		T* data = nullptr;
		u32 width = 0;
		u32 height = 0;
	};


	template <typename T>
	using DeviceView1 = DeviceMatrix2D<T>;


	using DeviceView = DeviceMatrix2D<Pixel>;
	using DeviceViewGray = DeviceMatrix2D<u8>;
	using DeviceViewYUV = DeviceMatrix2D<YUV2u8>;
	using DeviceViewBGR = DeviceMatrix2D<BGRu8>;
}

#endif // SIMAGE_NO_CUDA