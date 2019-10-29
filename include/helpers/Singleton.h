#pragma once

template <class T>
class Singleton
{
public:
	static T& instance()
	{
		static T instance;
		return instance;
	}
private:
	Singleton() {}

public:
	Singleton(T const&) = delete;
	void operator=(T const&) = delete;
};