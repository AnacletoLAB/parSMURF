// parSMURF
// Alessandro Petrini, 2018-2019
#include "sampler.h"

Sampler::Sampler( const Partition * const part, const uint8_t wmode, const double * const x, const uint32_t m, const uint32_t n ) :
		part( part ), ttd( part->ttd ), folds( part->folds ), x( x ), m( m ), n( n ), wmode( wmode ) {

	if (wmode == MODE_CV) {
		trngData  = new double[part->maxSize * (m + 1)];
		testData  = new double[(folds->maxPos + folds->maxNeg) * (m + 1)];
		testSize = part->testPosNum + part->testNegNum;
	} else if (wmode == MODE_TRAIN) {
		trngData  = new double[part->maxSize * (m + 1)];
	} else if (wmode == MODE_PREDICT) {
		testData  = new double[(folds->maxPos + folds->maxNeg) * (m + 1)];
		testSize = part->testPosNum + part->testNegNum;
	}
}

Sampler::~Sampler() {
	if ((wmode == MODE_CV) | (wmode == MODE_PREDICT))
		delete[] testData;
	if ((wmode == MODE_CV) | (wmode == MODE_TRAIN))
		delete[] trngData;
}

void Sampler::sample() {}
void Sampler::minOversample() {}
void Sampler::majUndersample() {}

void Sampler::getSample( const uint32_t numSamp, double * const sample ) {
	for (uint32_t i = 0; i < m + 1; i++)
		sample[i] = x[numSamp + n * i];
}

void Sampler::setSample( const uint32_t numCol, double * const sample, const uint32_t lineLen ) {
	double * dataOut = trngData;
	for (uint32_t i = 0; i < m + 1; i++)
		dataOut[numCol + lineLen * i] = sample[i];
}

void Sampler::copySample( const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen ) {
	double * dataOut = trngData;
	for (uint32_t i = 0; i < m + 1; i++)
		dataOut[nCol + lineLen * i] = x[nSam + n * i];
}

void Sampler::copySample( const double * const x, const uint32_t nSam, const uint32_t nCol, const uint32_t lineLen ) {
	double * dataOut = testData;
	for (uint32_t i = 0; i < m; i++)
		dataOut[nCol + lineLen * i] = x[nSam + n * i];
	// Label must be set to 0!
	dataOut[nCol + lineLen * m] = 0;
}

void Sampler::setPartition( const uint32_t currentPart ) {
	this->currentPartition = currentPart;
	this->currentFold = part->currentFold;
	uint32_t negInEachPartition = ceil( part->trngNegNum / (double)(part->nPart) );

	posForOverSmpl = part->trngPosNum;
	negForUndrSmpl = (currentPart != (part->nPart - 1)) ? negInEachPartition : part->trngNegNum - (negInEachPartition * (part->nPart - 1));
	pos = part->trngPosIdx;
	neg = part->trngNegIdx + (currentPart * negInEachPartition);

	// DEBUG: print partition information
	/*std::cout << "Current partition: " << currentPart << std::endl;
	std::cout << "Pos for oversampling: " << posForOverSmpl << std::endl;
	std::cout << "Neg for oversampling: " << negForUndrSmpl << std::endl;
	std::cout << "Pos: "; std::for_each( pos, pos + posForOverSmpl, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << "Neg: "; std::for_each( neg, neg + negForUndrSmpl, [](uint32_t nn) {std::cout << nn << " ";} ); std::cout << std::endl;
	std::cout << std::endl;*/
}

void Sampler::copyTestSet( const double * const x ) {
	// Why did I put this? Let's leave a warning, instead...
	// if (part->testPosNum + part->testNegNum > (folds->nPos[0] + folds->nNeg[0])) {
	// 	std::cout << "WARNING: part->testPosNum + part->testNegNum > (folds->nPos[0] + folds->nNeg[0]): " <<
	// 	 	part->testPosNum << " " << part->testNegNum << " " << folds->nPos[0] << " " << folds->nNeg[0] << std::endl;
	// 	//abort();
	// }

	uint32_t linLen = part->testPosNum + part->testNegNum;
	for (uint32_t i = 0; i < part->testPosNum; i++) {
		copySample( x, part->testPosIdx[i], i, linLen );
	}
	for (uint32_t i = 0; i < part->testNegNum; i++) {
		copySample( x, part->testNegIdx[i], i + part->testPosNum, linLen );
	}
	// DEBUG PRINT
	/*std::cout << std::endl;
	std::cout << "Raw part->testData view:" << std::endl;
	for (uint32_t i = 0; i < linLen * (m + 1); i++) {
		std::cout << testData[i] << " ";
		if ((i + 1) % (linLen) == 0) std::cout << std::endl;
	}
	std::cout << std::endl;*/
}

void Sampler::accumulateResInProbVect( const std::vector<std::vector<std::vector<double>>>& predictions,
		double * const class1, double * const class2 ) {

	uint32_t cc = 0;
	for (uint32_t i = 0; i < part->testPosNum; i++) {
		class1[part->testPosIdx[i]] += predictions[0][cc][0];
		class2[part->testPosIdx[i]] += predictions[0][cc][1];
		cc++;
	}
	for (uint32_t i = 0; i < part->testNegNum; i++) {
		class1[part->testNegIdx[i]] += predictions[0][cc][0];
		class2[part->testNegIdx[i]] += predictions[0][cc][1];
		cc++;
	}
}

// Guess what: now this function and accumulateResInProbVect are identical...
void Sampler::accumulateAndDivideResInProbVect( const std::vector<std::vector<std::vector<double>>>& predictions,
		double * const class1, double * const class2, const uint32_t nPart ) {

	uint32_t cc = 0;
	//std::cout << "In Sampler::accumulateAndDivideResInProbVect - nPart: " << nPart << std::endl;
	//double divider = 1.0 / (double)nPart;
	for (uint32_t i = 0; i < part->testPosNum; i++) {
		// class1[part->testPosIdx[i]] += ( predictions[0][cc][0] * divider );
		// class2[part->testPosIdx[i]] += ( predictions[0][cc][1] * divider );
		class1[part->testPosIdx[i]] += predictions[0][cc][0];
		class2[part->testPosIdx[i]] += predictions[0][cc][1];
		cc++;
	}
	for (uint32_t i = 0; i < part->testNegNum; i++) {
		// class1[part->testNegIdx[i]] += ( predictions[0][cc][0] * divider );
		// class2[part->testNegIdx[i]] += ( predictions[0][cc][1] * divider );
		class1[part->testNegIdx[i]] += predictions[0][cc][0];
		class2[part->testNegIdx[i]] += predictions[0][cc][1];
		cc++;
	}

	// // Scale predictions
	// for (uint32_t i = 0; i < part->testPosNum; i++) {
	// 	class1[part->testPosIdx[i]] *= divider;
	// 	class2[part->testPosIdx[i]] *= divider;
	// }
	// for (uint32_t i = 0; i < part->testNegNum; i++) {
	// 	class1[part->testNegIdx[i]] *= divider;
	// 	class2[part->testNegIdx[i]] *= divider;
	// }

}
