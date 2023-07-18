/* channel pixels */

namespace simage
{
	template <typename T>
	class RGBp
	{
	public:
		T* R;
		T* G;
		T* B;
	};


	template <typename T>
	class RGBAp
	{
	public:
		T* R;
		T* G;
		T* B;
		T* A;
	};


	template <typename T>
	class HSVp
	{
	public:
		T* H;
		T* S;
		T* V;
	};


	template <typename T>
	class LCHp
	{
	public:
		T* L;
		T* C;
		T* H;
	};


	template <typename T>
	class YUVp
	{
	public:
		T* Y;
		T* U;
		T* V;
	};


	using RGBf32p = RGBp<f32>;
	using RGBAf32p = RGBAp<f32>;

	using HSVf32p = HSVp<f32>;
	using YUVf32p = YUVp<f32>;
	using LCHf32p = LCHp<f32>;
}