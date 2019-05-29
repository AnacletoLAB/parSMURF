// parSMURF
// Alessandro Petrini, 2018-2019
#include "testtraindivider.h"

TestTrainDivider::TestTrainDivider( const Folds * const folds, const uint8_t wmode ) :
		folds( folds ), wmode( wmode ), nFolds( folds->nFolds ) {

	testPosIdx = new uint32_t*[nFolds];
	testNegIdx = new uint32_t*[nFolds];
	testPosNum = new uint32_t[nFolds];
	testNegNum = new uint32_t[nFolds];

	trngPosIdx = new uint32_t*[nFolds];
	trngNegIdx = new uint32_t*[nFolds];
	trngPosNum = new uint32_t[nFolds];
	trngNegNum = new uint32_t[nFolds];

	for (uint32_t i = 0; i < nFolds; i++) {

		uint32_t firstElement = folds->foldsIdx[i];

		testPosNum[i] = folds->nPos[i];
		testNegNum[i] = folds->nNeg[i];
		trngPosNum[i] = folds->totPos - testPosNum[i];
		trngNegNum[i] = folds->totNeg - testNegNum[i];

		// Hack for separate training and test executions
		//  (see arround line 65...)
		if (wmode == MODE_TRAIN) {
			trngPosNum[0] = testPosNum[0];
			trngNegNum[0] = testNegNum[0];
		}

		testPosIdx[i] = testNegIdx[i] = trngPosIdx[i] = trngNegIdx[i] = nullptr;

		testPosIdx[i] = new uint32_t[testPosNum[i]];		checkPtr<uint32_t>( testPosIdx[i], __FILE__, __LINE__ );
		testNegIdx[i] = new uint32_t[testNegNum[i]];		checkPtr<uint32_t>( testNegIdx[i], __FILE__, __LINE__ );
		trngPosIdx[i] = new uint32_t[trngPosNum[i]];		checkPtr<uint32_t>( trngPosIdx[i], __FILE__, __LINE__ );
		trngNegIdx[i] = new uint32_t[trngNegNum[i]];		checkPtr<uint32_t>( trngNegIdx[i], __FILE__, __LINE__ );

		// Copy the test set, splitted in positive and negative examples
		// std::memcpy( testPosIdx[i], &(folds->folds[firstElement]), testPosNum[i] * sizeof( uint32_t ) );
		// std::memcpy( testNegIdx[i], &(folds->folds[firstElement + testPosNum[i]]), testNegNum[i] * sizeof( uint32_t ) );
		size_t posIdx = 0, negIdx = 0;
		for (size_t j = 0; j < testPosNum[i] + testNegNum[i]; j++) {
			if (folds->labels[folds->folds[firstElement + j]] < 1) {
				testNegIdx[i][negIdx] = folds->folds[firstElement + j];
				negIdx++;
			} else {
				testPosIdx[i][posIdx] = folds->folds[firstElement + j];
				posIdx++;
			}
		}

		// Now copy the training set, again splitted in positive and negative examples
		// uint32_t elemCounter = 0;
		// for (uint32_t k = 0; k < folds->nFolds; k++) {
		// 	if (k == i)
		// 		continue;
		// 	firstElement = folds->foldsIdx[k];
		// 	std::memcpy( trngPosIdx[i] + elemCounter, &(folds->folds[firstElement]), folds->nPos[k] * sizeof( uint32_t ) );
		// 	elemCounter += folds->nPos[k];
		// }
		posIdx = 0, negIdx = 0;
		for (uint32_t k = 0; k < folds->nFolds; k++) {
			if (k == i)
				continue;
			firstElement = folds->foldsIdx[k];
			for (size_t j = 0; j < folds->nPos[i] + folds->nNeg[i]; j++) {
				if (folds->labels[folds->folds[firstElement + j]] < 1) {
					trngNegIdx[i][negIdx] = folds->folds[firstElement + j];
					negIdx++;
				} else {
					trngPosIdx[i][posIdx] = folds->folds[firstElement + j];
					posIdx++;
				}
			}
		}

		// elemCounter = 0;
		// for (uint32_t k = 0; k < folds->nFolds; k++) {
		// 	if (k == i)
		// 		continue;
		// 	firstElement = folds->foldsIdx[k] + folds->nPos[k];
		// 	std::memcpy( trngNegIdx[i] + elemCounter, &(folds->folds[firstElement]), folds->nNeg[k] * sizeof( uint32_t ) );
		// 	elemCounter += folds->nNeg[k];
		// }

		// Shuffle of the negative set set
		std::random_shuffle( trngNegIdx[i], trngNegIdx[i] + trngNegNum[i] );
	}

	// Hack for separate training and test executions
	// In this case it is supposed that:
	//	- only one fold exists
	//	- the whole dataset is used for training OR for testing
	// Since the code fills indexes arrays starting from test partition, it will result in an empty train set
	// in case of only one fold. Copying the set of indexes may be slower, but it is done only once, at the
	// beginning of computation. A previous attempt was made by swapping pointers, but was generally messier
	// and resulted in heap corruption....
	if (wmode == MODE_TRAIN) {
		std::memcpy( trngPosIdx[0], testPosIdx[0], testPosNum[0] * sizeof(uint32_t) );
		std::memcpy( trngNegIdx[0], testNegIdx[0], testNegNum[0] * sizeof(uint32_t) );
	}

}

TestTrainDivider::~TestTrainDivider() {
	for (uint32_t i = 0; i < nFolds; i++) {
		delete[] trngNegIdx[i];
		delete[] trngPosIdx[i];
		delete[] testNegIdx[i];
		delete[] testPosIdx[i];
	}
	delete[] trngNegNum;
	delete[] trngPosNum;
	delete[] trngNegIdx;
	delete[] trngPosIdx;
	delete[] testNegNum;
	delete[] testPosNum;
	delete[] testNegIdx;
	delete[] testPosIdx;
}

TestTrainDivider::TestTrainDivider( const Folds * const folds, const uint8_t wmode, uint32_t foldToJump ) :
		folds( folds ), wmode( wmode ), nFolds( folds->nFolds - 1 ) {

	std::cout << TXT_BIPRP << "internal CV ttd constructor - fold to jump: " << foldToJump << TXT_NORML << std::endl;
	uint32_t	totPos = folds->totPos - folds->nPos[foldToJump];
	uint32_t	totNeg = folds->totNeg - folds->nNeg[foldToJump];

	testPosIdx = new uint32_t*[nFolds];
	testNegIdx = new uint32_t*[nFolds];
	testPosNum = new uint32_t[nFolds];
	testNegNum = new uint32_t[nFolds];

	trngPosIdx = new uint32_t*[nFolds];
	trngNegIdx = new uint32_t*[nFolds];
	trngPosNum = new uint32_t[nFolds];
	trngNegNum = new uint32_t[nFolds];

	size_t currFoldIdx = 0;

	for (uint32_t i = 0; i < nFolds; i++) {
		if (currFoldIdx == foldToJump)
			currFoldIdx++;

		uint32_t firstElement = folds->foldsIdx[currFoldIdx];

		testPosNum[i] = folds->nPos[currFoldIdx];
		testNegNum[i] = folds->nNeg[currFoldIdx];
		trngPosNum[i] = totPos - testPosNum[i];
		trngNegNum[i] = totNeg - testNegNum[i];

		// Hack for separate training and test executions
		//  (see arround line 65...)
		if (wmode == MODE_TRAIN) {
			trngPosNum[0] = testPosNum[0];
			trngNegNum[0] = testNegNum[0];
		}

		testPosIdx[i] = testNegIdx[i] = trngPosIdx[i] = trngNegIdx[i] = nullptr;

		testPosIdx[i] = new uint32_t[testPosNum[i]];		checkPtr<uint32_t>( testPosIdx[i], __FILE__, __LINE__ );
		testNegIdx[i] = new uint32_t[testNegNum[i]];		checkPtr<uint32_t>( testNegIdx[i], __FILE__, __LINE__ );
		trngPosIdx[i] = new uint32_t[trngPosNum[i]];		checkPtr<uint32_t>( trngPosIdx[i], __FILE__, __LINE__ );
		trngNegIdx[i] = new uint32_t[trngNegNum[i]];		checkPtr<uint32_t>( trngNegIdx[i], __FILE__, __LINE__ );

		std::memcpy( testPosIdx[i], &(folds->folds[firstElement]), testPosNum[i] * sizeof( uint32_t ) );
		std::memcpy( testNegIdx[i], &(folds->folds[firstElement + testPosNum[i]]), testNegNum[i] * sizeof( uint32_t ) );

		uint32_t elemCounter = 0;
		for (uint32_t k = 0; k < folds->nFolds; k++) {
			if ((k == currFoldIdx) | (k == foldToJump))
				continue;
			firstElement = folds->foldsIdx[k];
			std::memcpy( trngPosIdx[i] + elemCounter, &(folds->folds[firstElement]), folds->nPos[k] * sizeof( uint32_t ) );
			elemCounter += folds->nPos[k];
		}

		elemCounter = 0;
		for (uint32_t k = 0; k < folds->nFolds; k++) {
			if ((k == currFoldIdx) | (k == foldToJump))
				continue;
			firstElement = folds->foldsIdx[k] + folds->nPos[k];
			std::memcpy( trngNegIdx[i] + elemCounter, &(folds->folds[firstElement]), folds->nNeg[k] * sizeof( uint32_t ) );
			elemCounter += folds->nNeg[k];
		}

		// Shuffle of the negative set set
		//std::random_shuffle( trngNegIdx[i], trngNegIdx[i] + trngNegNum[i] );
		currFoldIdx++;
	}

	// Hack for separate training and test executions
	// In this case it is supposed that:
	//	- only one fold exists
	//	- the whole dataset is used for training OR for testing
	// Since the code fills indexes arrays starting from test partition, it will result in an empty train set
	// in case of only one fold. Copying the set of indexes may be slower, but it is done only once, at the
	// beginning of computation. A previous attempt was made by swapping pointers, but was generally messier
	// and resulted in heap corruption....
	if (wmode == MODE_TRAIN) {
		std::memcpy( trngPosIdx[0], testPosIdx[0], testPosNum[0] * sizeof(uint32_t) );
		std::memcpy( trngNegIdx[0], testNegIdx[0], testNegNum[0] * sizeof(uint32_t) );
	}


}

void TestTrainDivider::verbose( uint32_t whichFold ) {
	uint32_t k = whichFold;
	std::cout << std::endl;
	std::cout << " *** Test and Training Divider ***" << std::endl;
	std::cout << "Set: " << k << std::endl;
	std::cout << "-- Training set" << std::endl << "Positive: " << trngPosNum[k] << " - Negative: " << trngNegNum[k] << std::endl;
	std::cout << "Training folds -- Positive: ";
	for (uint32_t i = 0; i < trngPosNum[k]; i++)
		std::cout << trngPosIdx[k][i] << " ";
	std::cout << std::endl;
	std::cout << "Training folds -- Negative: ";
	for (uint32_t i = 0; i < trngNegNum[k]; i++)
		std::cout << trngNegIdx[k][i] << " ";
	std::cout << std::endl;
	std::cout << "-- Test set" << std::endl;
	std::cout << "Positive: " << testPosNum[k] << " - Negative: " << testNegNum[k] << std::endl;
	std::cout << "Test fold -- Positive: ";
	for (uint32_t i = 0; i < testPosNum[k]; i++)
		std::cout << testPosIdx[k][i] << " ";
	std::cout << std::endl;
	std::cout << "Test fold -- Negative: ";
	for (uint32_t i = 0; i < testNegNum[k]; i++)
		std::cout << testNegIdx[k][i] << " ";
	std::cout << std::endl;

	std::cout << std::endl;
}

void TestTrainDivider::verbose() {
	for( uint32_t i = 0; i < nFolds; i++ ) {
		verbose(i);
		std::cout << std::endl;
	}
}
