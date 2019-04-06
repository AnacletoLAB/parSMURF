// parSMURF
// Alessandro Petrini, 2018-2019
#pragma once
#include <iostream>
#include <vector>
#include <mpi.h>

#include "HyperSMURFUtils.h"
#include "ArgHandler_new.h"
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "partitionMPI.h"
#include "MPIHelper.h"
#include "sendRecvThrd.h"
#include "curves.h"

#include "rfRanger.h"
#include "globals.h"
#include "ForestProbability.h"
#include "DataDouble.h"
#include "DataFloat.h"
#include "easyloggingpp/easylogging++.h"


class hyperSMURF_MPI {
public:
	hyperSMURF_MPI( const int rank, const int worldSize, const CommonParams * const commonParams, std::vector<GridParams> & gridParams,
		Folds * const foldManager, const MPIHelper * const MpiH, const std::vector<double> & x, const std::vector<uint32_t> & y,
		double * const class1Prob, double * const class2Prob );
	~hyperSMURF_MPI();
	void initStandard();
	void initInternalCV( uint32_t foldToJump );
	void smurfIt();
	void setGridParams( uint32_t idx );
	void createPart();
	bool parametersOptimizer( uint32_t foldToJump );

private:
	uint32_t findMaxAuprc( const std::vector<GridParams> & gridParams );

	// in
	const CommonParams				*	const	commonParams = nullptr;
	std::vector<GridParams>			&			gridParams;
	Folds							*	const	foldManager = nullptr;
	const std::vector<double>		&			x;
	const std::vector<uint32_t>		&			y;
	const MPIHelper					*	const	MpiH;
	// out
	double 							*	const	class1Prob = nullptr;
	double 							*	const	class2Prob = nullptr;
	// internal
	TestTrainDivider 				*			ttd = nullptr;

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
	std::string									outfilename;
	std::string									timeFilename;
	std::string									forestDirname;
	uint32_t									nThr;
	uint32_t									rfThr;
	uint8_t										wmode;
	uint8_t										woptimiz;
	bool	 									rfVerbose;
	bool										inInternalCV;
	bool										verboseMPI;
	bool										noMtSender;

	int											rank;
	int											worldSize;

	uint32_t									idxInGridParams;
	uint32_t									foldToJump;
	double 							*			tempcl1Prob;
	double 							*			tempcl2Prob;
};
