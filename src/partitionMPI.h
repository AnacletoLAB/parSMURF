// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include "folds.h"
#include "testtraindivider.h"

class PartitionMPI : public Partition {
public:
	PartitionMPI( const Folds * const folds, const TestTrainDivider * const ttd,
		const uint32_t nPart, const uint32_t fp, const uint32_t ratio, const uint32_t m, const uint32_t n,
		const uint32_t currentPart, const uint32_t currentFold, const uint32_t totalChunks );
	~PartitionMPI();

	void		verbose();
	void		fillTrngData( const double * const x, double * const trngData );
	void		fillTestData( const double * const x, double * const testData );

	uint32_t				currentFold;
	uint32_t				currentPartition;
	uint32_t				assignedToChunk;

	// Input to over/undersampler
	uint32_t				posForOverSmpl;
	uint32_t				negForUndrSmpl;
	uint32_t			*	pos;
	uint32_t			*	neg;

	// Output from over/undersampler
	uint32_t				trngPos;
	uint32_t				trngNeg;
	uint32_t				trngSize;

	uint32_t				posToBeGenerated;
	uint32_t				negToBeGenerated;
	uint32_t				lineLen;

	// Filled by Sampler::copySample()
	uint32_t				testSize;
	uint32_t			*	testPos;
	uint32_t			*	testNeg;

private:
	void copyTestSample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen, double * const testData );
	void copyTrngSample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen, double * const trngData );
};
