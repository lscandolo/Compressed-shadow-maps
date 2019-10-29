#pragma once

#include <cstdint>

#define NODE_NONEXIST	(0)
#define NODE_LEAF		(1)
#define NODE_EMPTY		(2)
#define NODE_INNER		(3)

#define LEAF_BIT  (0x1)
#define EMPTY_BIT (0x2)

static
inline uint32_t getnodemem(const uint8_t flags)
{
	switch (flags)
	{
		case NODE_INNER: return 9;
		case NODE_EMPTY: return 5;
		case NODE_LEAF:  return 4;
	};
	return 0;
}

static
inline uint32_t getnodemem16(const uint8_t flags)
{
	switch (flags)
	{
		case NODE_INNER: return 7;
		case NODE_EMPTY: return 3;
		case NODE_LEAF:  return 4;
	};
	return 0;
}

static
inline uint32_t getnodemem8(const uint8_t flags)
{
	switch (flags)
	{
		case NODE_INNER: return 3;
		case NODE_EMPTY: return 1;
		case NODE_LEAF:  return 2;
	};
	return 0;
}

static
inline bool is_emptynode(const uint8_t flags)
{
	return (flags == NODE_EMPTY);
}

static
inline bool is_leafnode(const uint8_t flags)
{
	return (flags == NODE_LEAF);
}