// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <vector>
#include <omp.h>

#include "HyperSMURFUtils.h"
#include "ArgHandler_new.h"
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "samplerKNN.h"
#include "samplerSimple.h"
#include "curves.h"

#include "rfRanger.h"
#include "globals.h"
#include "ForestProbability.h"
#include "DataDouble.h"
#include "DataFloat.h"
#include "HyperSMURFUtils.h"

class hyperSMURF {
public:
	hyperSMURF( const CommonParams * const commonParams, std::vector<GridParams> & gridParams,
		Folds * const foldManager, const std::vector<double> & x, const std::vector<uint32_t> & y,
		double * const class1Prob, double * const class2Prob );
	~hyperSMURF();
	void initStandard();
	void initInternalCV( uint32_t foldToJump );
	void smurfIt();
	void setGridParams( uint32_t idx );
	void createPart();
	bool parametersOptimizer( uint32_t foldToJump );
	void evaluateTrainMetrics();

private:
	// in
	const CommonParams				*	const	commonParams = nullptr;
	std::vector<GridParams>			&			gridParams;
	Folds							*	const	foldManager = nullptr;
	const std::vector<double>		&			x;
	const std::vector<uint32_t>		&			y;
	// out
	double 							*	const	class1Prob = nullptr;
	double 							*	const	class2Prob = nullptr;
	// internal
	TestTrainDivider 				*			ttd = nullptr;
	Partition 						*			part = nullptr;
	bool										lockCreated;
	omp_lock_t									accumulLock;

	uint32_t									nn;
	uint32_t									mm;
	uint32_t									nFolds;
	uint32_t									nPart;
	uint32_t									numTrees;
	uint32_t									fp;
	uint32_t									ratio;
	uint32_t									k;
	uint32_t									mtry;
	uint32_t									seed;
	uint32_t									verboseLevel;
	uint32_t									trainingAuroc;
	uint32_t									trainingAuprc;
	std::string									outfilename;
	std::string									timeFilename;
	std::string									forestDirname;
	uint32_t									nThr;
	uint32_t									rfThr;
	uint8_t										wmode;
	uint8_t										woptimiz;
	bool	 									rfVerbose;
	bool										inInternalCV;

	uint32_t									idxInGridParams;
	uint32_t									foldToJump;
	double 							*			tempcl1Prob;
	double 							*			tempcl2Prob;
};
