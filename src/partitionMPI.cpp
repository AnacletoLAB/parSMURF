// parSMURF
// Alessandro Petrini, 2018-2019
#include "partition.h"
#include "partitionMPI.h"
#include <cmath>

PartitionMPI::PartitionMPI( const Folds * const folds, const TestTrainDivider * const ttd,
	const uint32_t nPart, const uint32_t fp, const uint32_t ratio, const uint32_t m, const uint32_t n,
 	const uint32_t currentPart, const uint32_t currentFold_, const uint32_t totalChunks ) :
		Partition( folds, ttd, nPart, fp, ratio, m, n ) {

	// set current partition and build the arrays
	currentFold = currentFold_;
	setFold( currentFold );
	currentPartition = currentPart;
	uint32_t negInEachPartition = ceil( trngNegNum / (double)(nPart) );

	posForOverSmpl = trngPosNum;
	negForUndrSmpl = (currentPart != (nPart - 1)) ? negInEachPartition : trngNegNum - (negInEachPartition * (nPart - 1));
	pos = trngPosIdx;
	neg = trngNegIdx + (currentPart * negInEachPartition);

	testSize = testPosNum + testNegNum;
	testPos = testPosIdx;
	testNeg = testNegIdx;

	posToBeGenerated = (fp + 1) * posForOverSmpl;
	negToBeGenerated = posToBeGenerated * ratio;
	// If ratio == 0 => disable the undersampler
	if (negToBeGenerated == 0)
		negToBeGenerated = negForUndrSmpl;
	if (negToBeGenerated > negForUndrSmpl) {
		std::cout << "fold: " << currentFold << " part.: " << currentPart << " insuff. negatives: " << negToBeGenerated <<
		" requested but " << negForUndrSmpl << " available." << std::endl;
	}
	negToBeGenerated = (negToBeGenerated > negForUndrSmpl) ? negForUndrSmpl : negToBeGenerated;
	lineLen = posToBeGenerated + negToBeGenerated;
	trngSize = posToBeGenerated + negToBeGenerated;
	assignedToChunk = 1 + currentPart / totalChunks;
}

PartitionMPI::~PartitionMPI() {
}

// from [sampler::copyTestSet( ... )]
void PartitionMPI::fillTestData( const double * const x, double * const testData ) {

	uint32_t linLen = testPosNum + testNegNum;
	for (uint32_t i = 0; i < testPosNum; i++) {
		//std::cout << "linelen = " << linLen << " - sample: " << testPos[i] << std::endl;
		copyTestSample( x, testPos[i], i, linLen, testData );
	}
	for (uint32_t i = 0; i < testNegNum; i++) {
		//std::cout << "linelen = " << linLen << " - sample: " << testNeg[i] << std::endl;
		copyTestSample( x, testNeg[i], i + testPosNum, linLen, testData );
	}
}

void PartitionMPI::fillTrngData( const double * const x, double * const trngData ) {
	// Copy the positives samples in the trngData matrix
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		copyTrngSample( x, pos[i], i, lineLen, trngData );		// copy on Part->trngData
	}

	// Copy the negative samples in the trngData matrix
	for (uint32_t i = 0; i < negToBeGenerated; i++) {
		copyTrngSample( x, neg[i], i + posToBeGenerated, lineLen,trngData );
	}
}

void PartitionMPI::verbose() {
	std::cout << "Current fold: " << currentFold << std::endl;
	std::cout << "Current partition: " << currentPartition << std::endl;
	std::cout << "Assingned to chunk: " << assignedToChunk << std::endl;
	std::cout << "Pos for oversampling: " << posForOverSmpl << " - Pos to be generated: " << posToBeGenerated << std::endl;
	std::cout << "Neg for oversampling: " << negForUndrSmpl << " - Neg to be generated: " << negToBeGenerated << std::endl;
	std::cout << "Pos: "; std::for_each( pos, pos + posForOverSmpl, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << "Neg: "; std::for_each( neg, neg + negForUndrSmpl, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << "Test size = testPosNum + testNegNum: " << testSize << " = " << testPosNum << " + " << testNegNum << std::endl;
	std::cout << "Test Pos: "; std::for_each( testPos, testPos + testPosNum, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << "Test Neg: "; std::for_each( testNeg, testNeg + testNegNum, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << std::endl;
}


/////////////////////
void PartitionMPI::copyTestSample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen, double * const testData ) {
	double * const dataOut = testData;
	for (uint32_t i = 0; i < m; i++)
		dataOut[nCol + lineLen * i] = x[nSam + n * i];
	// Label must be set to 0!
	dataOut[nCol + lineLen * m] = 0;
}

void PartitionMPI::copyTrngSample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen, double * const trngData ) {
	double * dataOut = trngData;
	for (uint32_t i = 0; i < m + 1; i++)
		dataOut[nCol + lineLen * i] = x[nSam + n * i];
}
