// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <vector>
#include <iomanip>
#include <complex>		// Why?
#include "ANN.h"


class SamplerKNNMPI {
public:
	SamplerKNNMPI( double * const x, const uint32_t m, const uint32_t n, const uint32_t lineLen,
		const uint32_t k, const uint32_t fp, const uint32_t posToBeGenerated,  const uint32_t posForOverSmpl )
			: x{x}, m{m}, n{n}, lineLen{lineLen}, k{k}, fp{fp}, posToBeGenerated{posToBeGenerated}, posForOverSmpl{posForOverSmpl} {};
	~SamplerKNNMPI() {};

	void sample();
	void verbose();

	void accumulateTempProb( const std::vector<std::vector<std::vector<double>>>& predictions, const uint32_t testSize,
			double * const class1, double * const class2 );

private:
	void minOversample();
	void getSample( const uint32_t numSamp, double * const sample );
	void setSample( const uint32_t numCol, const double * const sample );

	double * const		x;
	const uint32_t		m;
	const uint32_t		n;
	const uint32_t		lineLen;
	const uint32_t 		k; 			// num neighbhors for KNN
	const uint32_t		fp;
	const uint32_t		posToBeGenerated;
	const uint32_t		posForOverSmpl;
};
