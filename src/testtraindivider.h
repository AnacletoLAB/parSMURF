// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <cstring>
#include "HyperSMURFUtils.h"
#include "folds.h"

class TestTrainDivider {
public:
	TestTrainDivider( const Folds * const folds, const uint8_t wmode );
	TestTrainDivider( const Folds * const folds, const uint8_t wmode, uint32_t foldToJump );
	~TestTrainDivider();

	void verbose();
	void verbose( uint32_t i );

	const Folds * const 	folds;
	const uint8_t			wmode;

	uint32_t				nFolds;

	uint32_t	**			testPosIdx;
	uint32_t	**			testNegIdx;
	uint32_t	*			testPosNum;
	uint32_t	*			testNegNum;

	uint32_t	**			trngPosIdx;
	uint32_t	**			trngNegIdx;
	uint32_t	*			trngPosNum;
	uint32_t	*			trngNegNum;
};
