// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <iomanip>
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "sampler.h"

class SamplerSimple : public Sampler {
public:
	SamplerSimple( const Partition * const part, const uint8_t wmode, const double * const x, const uint32_t m, const uint32_t n );
	~SamplerSimple();

	void sample();
	void verbose();

private:
	void setup();
	void minOversample();
	void majUndersample();

	double		*	tempBufA;
	double		*	tempBufB;
	uint32_t		negToBeGenerated;
	uint32_t		posToBeGenerated;
	uint32_t		lineLen;

};
