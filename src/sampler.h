// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <vector>
#include <cstring>
#include "HyperSMURFUtils.h"
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"

class Sampler {
public:
	Sampler( const Partition * const part, const uint8_t wmode, const double * const x, const uint32_t m, const uint32_t n );
	~Sampler();

	const Partition			* const part;
	const TestTrainDivider	* const ttd;
	const Folds 			* const	folds;
	const double			* const x;
	const uint32_t					m;
	const uint32_t					n;
	const uint8_t					wmode;

	uint32_t						currentPartition;
	uint32_t						currentFold;

	// Input to over/undersampler
	uint32_t						posForOverSmpl;
	uint32_t						negForUndrSmpl;
	const uint32_t			*		pos;
	const uint32_t			*		neg;

	// Output from over/undersampler
	mutable double			*		trngData;
	mutable uint32_t				trngPos;
	mutable uint32_t				trngNeg;
	mutable uint32_t				trngSize;

	// Filled by Sampler::copySample()
	double					*		testData;
	uint32_t						testSize;

	virtual void sample();
	void setPartition( const uint32_t currentPart );
	void copyTestSet( const double * const x );
	// void accumulateResInProbVect( const std::vector<std::vector<std::vector<double>>>& predictions,
	// 		double * const class1, double * const class2 );
	void accumulateAndDivideResInProbVect( const std::vector<std::vector<std::vector<double>>>& predictions,
			double * const class1, double * const class2, const uint32_t nPart );

	// Functions that mess with data and training matrices
	// They provide a minimal handy interface for the oversampler and undersampler
	// sample is a r/w vector of m + 1 elements
	// lineLen, all in all, is equal to the total number of samples that will be contained in the training set
	//		( see SimpleSampler::setup() )

	// TODO: inline them all!
	// Gets the sample nSam from the data matrix and copy it in the sample buffer
	void getSample( const uint32_t nSam, double * const sample );
	// Writes the sample buffer in the nCol of the training matrix
	void setSample( const uint32_t nCol, double * const sample, const uint32_t lineLen );
	// Gets the sample nSam from the data matrix and writes it in the nCol of the training matrix
	void copySample( const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen );
	// Gets the sample (without label) nSam from the data matrix and writes it in the nCol of the training matrix
	void copySampleNoLabel( const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen );
	// Used in copying the test set...
	void copySample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen );

private:
	virtual void minOversample();
	virtual void majUndersample();

};
