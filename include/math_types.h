#pragma once

#include <glm/glm.hpp>
#include <array>


/////////////////////////////////////////////
//////// Bounding box
/////////////////////////////////////////////

template<int N, typename F, typename T>
struct _bbox_base
{
	typedef _bbox_base<N,F,T> this_type;
	typedef T container_type;
	typedef F value_type;

	T max;
	T min;

	_bbox_base<N,F,T>()
	{
		clear();
	}

	_bbox_base<N, F, T>(const T& tmin, const T& tmax)
	: min(tmin)
	, max(tmax)
	{}

	_bbox_base<N, F, T>(const T& tsingle)
	: min(tsingle)
	, max(tsingle)
	{}

	this_type& clear()
	{
		max = T(value_type(INT_MIN));
		min = T(value_type(INT_MAX));
		return (*this);
	}

	this_type& extend(const T& t)
	{
		for (int i = 0; i < N; ++i) 
		{
			max[i] = std::max(max[i], t[i]);
			min[i] = std::min(min[i], t[i]);
		}
		return (*this);
	}

	container_type center() const
	{
		T t;
		for (int i = 0; i < N; ++i) t[i] = (max[i] + min[i]) / F(2.0);

		return t;
	}

	container_type extent() const
	{
		T e;
		for (int i = 0; i < N; ++i) e[i] = max[i] - min[i];
		return e;
	}

	value_type diagonal() const
	{ 
		T d = extent();
		double diagonalSqr = 0.0;
		for (int i = 0; i < N; ++i) diagonalSqr += double(d[i]) * double(d[i]);
		return value_type(sqrt(diagonalSqr));
	}

	this_type operator*(float rhs)
	{
		this_type lhs = *this;
		for (int i = 0; i < N; ++i) lhs.max[i] *= rhs;
		for (int i = 0; i < N; ++i) lhs.min[i] *= rhs;

		return lhs;
	}

	this_type& operator*=(float rhs)
	{
		(*this) = (*this) * rhs;
		return (*this);
	}

	this_type operator/(float rhs)
	{
		this_type lhs = *this;
		for (int i = 0; i < N; ++i) lhs.max[i] /= rhs;
		for (int i = 0; i < N; ++i) lhs.min[i] /= rhs;

		return lhs;
	}

	this_type& operator/=(float rhs)
	{
		(*this) = (*this) / rhs;
		return (*this);
	}

};

template<int N, typename F>
using _bbox_vec = _bbox_base < N, F, glm::vec<N, F> >;

template<int N, typename F>
using _bbox_array = _bbox_base < N, F, std::array<F, N> >;

using bbox4 = _bbox_vec<int(4), float>;
using bbox3 = _bbox_vec<int(3), float>;
using bbox2 = _bbox_vec<int(2), float>;


/////////////////////////////////////////////
//////// Bounding sphere
/////////////////////////////////////////////

template<int N, typename F, typename T>
struct _bsphere_base
{
	typedef _bsphere_base<N, F, T> this_type;
	typedef T container_type;
	typedef F value_type;

	T center;
	F radius;

	_bsphere_base<N, F, T>()
	{
		clear();
	}

	_bsphere_base<N, F, T>(const T& _center, const F& _radius)
		: center(_center)
		, radius(_radius)
	{}

	this_type& clear()
	{
		center = T(0,0,0);
		radius = F(-1.f);
		return (*this);
	}

	container_type extent() const
	{
		return F(2.0)*radius;
	}

	template< typename BF, typename BT>
	this_type& from_bbox(const _bbox_base<N, BF, BT>& bbox)
	{
		BT c = bbox.center();
		for (int i = 0; i < N; ++i) { center[i] = c[i]; }
		radius = F(bbox.diagonal()) / F(2.0);
		return *this;
	}

	this_type& extend(const T& t)
	{
		if (radius < 0) 
		{
			center = t;
			radius = F(0);
			return *this;
		}

		T diff;
		for (int i = 0; i < N; ++i) diff[i] = center[i] - t[i];

		F dist = 0;
		for (int i = 0; i < N; ++i) dist += F(T[i]*T[i]);
		F = sqrtf(F);


		if (dist > 0 && dist > radius)
		{
			T opposite = center + diff*(radius/dist);
			center = (t + opposite) / 2.0;
			radius = F( (dist+radius) / 2.0 );
		}

		return *this;

	}

	this_type& combine(const this_type& rhs)
	{
		T diff;
		for (int i = 0; i < N; ++i) diff[i] = center[i] - rhs.center[i];

		F dist = 0;
		for (int i = 0; i < N; ++i) dist += F(T[i] * T[i]);
		F = sqrtf(F);

		if (dist == F(0)) return *this;
		
		diff = diff / dist;

		return extend(rhs.center - diff*(rhs.radius/dist));

	}

};

template<int N, typename F>
using _bsphere_vec = _bsphere_base < N, F, glm::vec<N, F> >;

template<int N, typename F>
using _bsphere_array = _bsphere_base < N, F, std::array<F, N> >;

using bsphere2 = _bsphere_vec<int(2), float>;
using bsphere3 = _bsphere_vec<int(3), float>;
