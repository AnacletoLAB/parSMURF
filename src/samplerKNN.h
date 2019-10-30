// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <iomanip>
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "sampler.h"

typedef double data_t;

class SamplerKNN : public Sampler {
public:
	SamplerKNN( const Partition * const part, const uint8_t wmode, const double * const x, const uint32_t m, const uint32_t n, uint32_t k )
			: Sampler( part, wmode, x, m, n ), k{k} {}
	~SamplerKNN() {};

	void sample();
	void verbose();

	// TEMP:
	uint32_t		lineLen;

private:
	void setup();
	void minOversample();
	void majUndersample();

	uint32_t		negToBeGenerated;
	uint32_t		posToBeGenerated;
	// uint32_t		lineLen;
	uint32_t 		k; 			// num neighbhors for KNN

};
