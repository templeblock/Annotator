#pragma once

#include "Color8.h"
#include "Vec4.h"

namespace imqs {
namespace gfx {

class Color8;

class ColorHSVA {
public:
	uint8_t h = 0;
	uint8_t s = 0;
	uint8_t v = 0;
	uint8_t a = 1;

	ColorHSVA() {}
	ColorHSVA(uint8_t h, uint8_t s, uint8_t v, uint8_t a) : h(h), s(s), v(v), a(a) {}

	// Input values are clamped between 0..1
	static ColorHSVA FromFloat(float h, float s, float v, float a);

	Color8      ToRGBA() const;
	static void ToRGBA(const Vec4f& hsva, Vec4f& rgba);

	float Hf() const { return (float) h / 255.0f; } // Returns Hue 0..1
	float Sf() const { return (float) s / 255.0f; } // Returns Sat 0..1
	float Vf() const { return (float) v / 255.0f; } // Returns Val 0..1
	float Af() const { return (float) a / 255.0f; } // Returns Alpha 0..1

	static ColorHSVA White() { return ColorHSVA(0, 0, 255, 255); }
	static ColorHSVA Black() { return ColorHSVA(0, 0, 0, 255); }
	static ColorHSVA Transparent() { return ColorHSVA(0, 0, 0, 0); }
};
} // namespace gfx
} // namespace imqs
