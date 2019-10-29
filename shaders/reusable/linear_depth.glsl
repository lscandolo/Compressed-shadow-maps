#ifndef LINEARDEPTH_GLSL
#define LINEARDEPTH_GLSL

// Non linear [0,1] to linear [near,far]
float linearizeDepthVS(float d, float near, float far)
{
	d = d * 2.0 - 1.0;
	return 2.0 * near * far / (far + near - d * (far - near));
}

// linear [near,far] to non linear [0,1]
float unlinearizeDepthVS(float d, float near, float far)
{
	d = ((2.0*near*far) / d - (far + near)) / (near - far);
	d = (d * 0.5 + 0.5);
	return d;
}

// Non linear [0,1] to linear [0,1]
float linearizeDepth(float d, float near, float far)
{
	d = linearizeDepthVS(d, near, far);
	return (d - near) / (far - near);
}

// linear [0,1] to non linear [0,1]
float unlinearizeDepth(float d, float near, float far)
{
	d = near + d * (far - near);
	return unlinearizeDepthVS(d, near, far);
}

#endif // LINEARDEPTH_GLSL
