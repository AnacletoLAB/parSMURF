// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include "folds.h"
#include "testtraindivider.h"

class Partition {
public:
	Partition( const Folds * const folds, const TestTrainDivider * const ttd,
		const uint32_t nPart, const uint32_t fp, const uint32_t ratio, const uint32_t m, const uint32_t n );
	~Partition();

	void setFold( const uint32_t currentFold );

	const Folds 			* const		folds;
	const TestTrainDivider	* const 	ttd;
	uint32_t							nPart;
	uint32_t							fp;
	uint32_t							ratio;
	const uint32_t						m;
	const uint32_t						n;

	// Once upon a time, this was declared as const
	uint32_t							maxSize;		// maximum number of samples on the training matrix
														// I have some notes somewhere about it...

	uint32_t							currentFold;
	uint32_t							trngPosNum;		// copied from ttd
	uint32_t							trngNegNum;		//		""
	uint32_t							testPosNum;		//		""
	uint32_t							testNegNum;		//		""
	uint32_t				*			testPosIdx;		//		""
	uint32_t				*			testNegIdx;		//		""
	uint32_t				*			trngPosIdx;		//		""
	uint32_t				*			trngNegIdx;		//		""
};
