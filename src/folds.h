// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>

class Folds {
public:
	Folds( uint32_t numberOfFolds, uint32_t classSize, const uint32_t * labels, std::vector<uint32_t>& ff, bool randomFold );
	~Folds();

	void setLabelsPtr( const uint32_t * labels );
	void computeFolds();
	void verbose();							// Here's an example for how to use Folds:: data


	const uint32_t			classSize;		// number of samples
	const uint32_t 			nFolds;			// number of folds
	const uint32_t *		labels;			// 0/1 label vector
	std::vector<uint32_t>&	fromFile;

	uint32_t				totPos;			// total number of positive labels
	uint32_t				totNeg;			// total number of negative labels
	uint32_t * 				nPos;			// number of positive labels of the i-th fold
	uint32_t * 				nNeg;			// number of negative labels of the i-th fold
	uint32_t * 				foldsIdx;		// starting index of i-th fold in the folds array (lenght = nFolds + 1)
	uint32_t * 				folds;			// compressed vector of fold (lenght = classSize)
	uint32_t				maxPos;			// this one is used for the allocation of the training matrix; it contains the max of nPos values
	uint32_t				maxNeg;			// As maxPos, but for test matrix

	bool					randomFold;		// True if random fold generation is enabled

private:
	std::vector<uint32_t>	tempPosIdx;		// Don't ask
	std::vector<uint32_t>	tempNegIdx;

};
