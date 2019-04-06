// parSMURF
// Alessandro Petrini, 2018-2019
#include "samplerSimple.h"

SamplerSimple::SamplerSimple( const Partition * const part, const uint8_t wmode, const double * const x, const uint32_t m, const uint32_t n ) :
		Sampler( part, wmode, x, m, n ) {
	tempBufA = new double[m + 1];
	tempBufB = new double[m + 1];
}

SamplerSimple::~SamplerSimple() {
	delete[] tempBufB;
	delete[] tempBufA;
}

void SamplerSimple::sample() {
	setup();
	minOversample();
	majUndersample();
	trngSize = trngPos + trngNeg;
}

void SamplerSimple::setup() {
	// Calculates in advance the total number of positive and negative samples that
	// the training set will be constituted of. We just need the number of
	// positive and negative samples in the current partition.
	posToBeGenerated = (part->fp + 1) * posForOverSmpl;
	negToBeGenerated = posToBeGenerated * part->ratio;
	// If ratio == 0 => disable the undersampler
	if (negToBeGenerated == 0)
		negToBeGenerated = negForUndrSmpl;
	if (negToBeGenerated > negForUndrSmpl) {
		std::cout << "WARNING - fold: " << currentFold << " part.: " << currentPartition << " insuff. negatives: " << negToBeGenerated <<
		" requested but " << negForUndrSmpl << " available." << std::endl;
	}
	negToBeGenerated = (negToBeGenerated > negForUndrSmpl) ? negForUndrSmpl : negToBeGenerated;
	lineLen = posToBeGenerated + negToBeGenerated;
}

void SamplerSimple::minOversample() {
	// Saving the size of the positive testset
	trngPos = posToBeGenerated;

	// Check that there's enough room in the training matrix. There should be, unless maxSize formula
	// is wrong...
	if (trngPos > part->maxSize) {
		std::cerr << "error: training matrix is too small... check maxSize formula (that shouldn't have happened, anyway...)" << std::endl;
		std::cerr << "[in SamplerSimple::minOversample()]" << std::endl;
		abort();
	}
	if (trngPos < 2) {
		std::cerr << "error: we need at least two positive samples in each training partition for oversampling to happen" << std::endl;
		abort();
	}

	// Copy the original positives in the training matrix
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		copySample( pos[i], i, lineLen );
	}

	// Generating the other positive samples
	// Choose two random positives samples and generate a random third point on the segment between the first two
	// Label of the new point is set to 1
	for (uint32_t i = posForOverSmpl; i < posToBeGenerated; i++) {
		uint32_t randPtA, randPtB;
		double alpha = rand() / (double)RAND_MAX;
		do {
			randPtA = rand() % posForOverSmpl;
			randPtB = rand() % posForOverSmpl;
		} while (randPtA == randPtB);
		getSample( pos[randPtA], tempBufA );
		getSample( pos[randPtB], tempBufB );
		for (uint32_t k = 0; k < m; k++) {		// PERCHé m e non m+1
			tempBufA[k] = tempBufA[k] * alpha + tempBufB[k] * (1 - alpha);
		}
		setSample( i, tempBufA, lineLen );
	}

}

void SamplerSimple::majUndersample() {

	// Size of the positive test set (original + generated)
	uint32_t generatedPos = trngPos;

	// Guess what?
	if (trngPos + negToBeGenerated > part->maxSize) {
		std::cerr << "error: training matrix is too small... check maxSize formula (that shouldn't have happened, anyway...)" << std::endl;
		std::cerr << "[in SamplerSimple::majUndersample()]" << std::endl;
		abort();
	}

	// copy the points
	for (uint32_t i = 0; i < negToBeGenerated; i++) {
		copySample( neg[i], i + generatedPos, lineLen );
	}

	// and save the size of the negative set
	trngNeg = negToBeGenerated;


}

void SamplerSimple::verbose() {
	double * dataOut = trngData;

	std::cout << "  *** Simple Sampler *** " << std::endl;
	std::cout << "Data size: " << trngSize << " - Positives: " << trngPos << " - Negatives: " << trngNeg << std::endl;
	std::cout << "Original positives" << std::endl;
	for (uint32_t i = 0; i < posForOverSmpl; i++) {
		std::cout << std::setw( 4 ) << i << " | " << std::setw( 4 ) << pos[i] << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;
	std::cout << "Generated positives" << std::endl;
	for (uint32_t i = posForOverSmpl; i < trngPos; i++) {
		std::cout << std::setw( 4 ) << i << " | " << std::setw( 4 ) << "xxx" << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}
	std::cout << std::endl;

	dataOut = trngData + trngPos;
	std::cout << "Undersampled negatives" << std::endl;
	for (uint32_t i = 0; i < trngNeg; i++) {
		std::cout << std::setw( 4 ) << i + trngPos << " | " << std::setw( 4 ) << neg[i] << " | ";
		for (uint32_t k = 0; k < m + 1; k++) {
			std::cout << std::setw( 10 ) << dataOut[i + lineLen * k] << " | ";
		}
		std::cout << std::endl;
	}

	std::cout << std::endl;
	std::cout << "Raw part->trngData view:" << std::endl;
	for (uint32_t i = 0; i < trngSize * (m + 1); i++) {
		std::cout << trngData[i] << " ";
		if ((i + 1) % (lineLen) == 0) std::cout << std::endl;
	}
	std::cout << std::endl;
}
