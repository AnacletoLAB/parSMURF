// parSMURF
// Alessandro Petrini, 2018-2019
#include "partition.h"
#include <cmath>

// Check notes for maxSize formula
Partition::Partition( const Folds * const folds, const TestTrainDivider * const ttd,
	const uint32_t nPart, const uint32_t fp, const uint32_t ratio, const uint32_t m, const uint32_t n ) :
		folds( folds ), ttd( ttd ), nPart( nPart ), fp( fp ), ratio( ratio ), m( m ), n( n ),
		maxSize( ratio == 0 ?
				( (fp + 1) * (folds->maxPos) + folds->maxNeg ) * (folds->nFolds - 1) : // fix per ratio = 0
				(fp + 1) * (ratio + 1) * (folds->maxPos) * (folds->nFolds - 1) ) {

	// fix for TRAIN mode (we only have one fold...)
	if (folds->nFolds == 1)
		maxSize = (ratio == 0 ?
				( (fp + 1) * (folds->maxPos) + folds->maxNeg ) : // fix per ratio = 0
				(fp + 1) * (ratio + 1) * (folds->maxPos) );

}

Partition::~Partition() {
}

void Partition::setFold( const uint32_t currentFold ) {
	this->currentFold = currentFold;
	trngPosNum = ttd->trngPosNum[currentFold];
	trngNegNum = ttd->trngNegNum[currentFold];
	trngPosIdx = ttd->trngPosIdx[currentFold];
	trngNegIdx = ttd->trngNegIdx[currentFold];

	testPosNum = ttd->testPosNum[currentFold];
	testNegNum = ttd->testNegNum[currentFold];
	testPosIdx = ttd->testPosIdx[currentFold];
	testNegIdx = ttd->testNegIdx[currentFold];

}
