// parSMURF
// Alessandro Petrini, 2018-2019
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <stdexcept>
#include <thread>
#include <cmath>
#include <mutex>
#include <mpi.h>

#include "HyperSMURFUtils.h"
#include "ArgHandler_new.h"
#include "fileImport.h"
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "partitionMPI.h"
#include "hyperSMURF_MPI.h"
#include "easyloggingpp/easylogging++.h"

INITIALIZE_EASYLOGGINGPP

int main( int argc, char **argv ) {
	START_EASYLOGGINGPP(argc, argv);
	int	rank = 0;
	int worldSize = 1;
	MPI_Init( &argc, &argv );
	// int provided;
	// MPI_Init_thread( &argc, &argv, MPI_THREAD_SERIALIZED, &provided );
	MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	////EasyLogging++
	checkLoggerConfFile();
    el::Configurations conf("logger.conf");
    el::Loggers::reconfigureLogger("default", conf);
    el::Loggers::reconfigureAllLoggers(conf);

	// New ArgHandle for grid search. Parameters of each run are read from json (no more command line args)
	// and stored into a struct
	std::vector<GridParams> gridParams;
	ArgHandle commandLine( argc, argv, gridParams );
	commandLine.processCommandLine( rank );

	CommonParams commonParams;
	uint32_t nn					= commonParams.nn			= commandLine.n;
	uint32_t mm					= commonParams.mm			= commandLine.m;
	uint32_t nFolds				= commonParams.nFolds		= commandLine.nFolds;
	const uint32_t seed			= commonParams.seed			= commandLine.seed;
	const uint32_t verboseLevel = commonParams.verboseLevel	= commandLine.verboseLevel;
	bool verboseMPI				= commonParams.verboseMPI	= commandLine.verboseMPI;
	commonParams.noMtSender		= commandLine.noMtSender;
	commonParams.outfilename	= commandLine.outFilename;
	commonParams.timeFilename	= commandLine.timeFilename;
	commonParams.forestDirname	= commandLine.forestDirname;
	commonParams.nThr 			= commandLine.ensThreads;
	commonParams.rfThr			= commandLine.rfThreads;
	commonParams.wmode			= commandLine.wmode;
	commonParams.woptimiz		= commandLine.woptimiz;
	commonParams.rfVerbose 		= (commonParams.verboseLevel >= VERBRF);
	commonParams.minFold		= commandLine.minFold;
	commonParams.maxFold		= commandLine.maxFold;
	commonParams.cfgFilename	= commandLine.extConfigFilename;
	std::string	outfilename		= commandLine.outFilename;
	std::string	timeFilename	= commandLine.timeFilename;
	std::string	forestDirname	= commandLine.forestDirname;

	std::ifstream parametersFile( "params.dat", std::ios::in );
	if (parametersFile) {
		std::cout << TXT_BIGRN << "'params.dat' found. Overriding input parameters..." << TXT_NORML << std::endl;
		commonParams.customCV = true;
		parametersFile.close();
		Importer::importParameters( gridParams );
	} else {
		commonParams.customCV = false;
	}

	// Common
	std::vector<uint32_t> yy;
	std::vector<uint32_t> ff;
	std::vector<double>   xx;

	// MPI master process
	// Reading / generating input data
	if ( rank == 0 ) {
		yy = std::vector<uint32_t>( nn );					// Labels
		ff = std::vector<uint32_t>( nn );					// Folds
		xx = std::vector<double>  ( (mm + 1) * nn );		// Data
		if (verboseLevel > VERBSILENT)
			std::cout << TXT_BIBLU << "Reading or generating data" << TXT_NORML << std::endl;
		if (!commandLine.simulate) {
			Importer::import( &commandLine, xx, yy, ff, &nFolds );
			nn = commonParams.nn = yy.size();
			mm = commonParams.mm = xx.size() / yy.size() - 1;
			// Conversion to Ranger label format moved in Importer::import()
			// std::for_each( yy.begin(), yy.end(), [&xx]( uint32_t nnn ) mutable { if (nnn > 0) xx.push_back( 1.0 ); else xx.push_back( 2.0 ); } );
		}
		else
			generateRandomSet( nn, mm, xx, yy, commandLine.prob, seed );

		if (commonParams.customCV && (gridParams.size() != nFolds)) {
			std::cout << TXT_BIRED << "Mismatch between nFolds and overidden fold number from params.dat. Aborting..." << TXT_NORML << std::endl;
			MPI_Finalize();
			exit( -1 );
		}
	}

	// Handle the special mtry var
	if (rank == 0) {
		commandLine.processMtry( mm );
	}

	if (rank == 0) {
		if (commandLine.printCurrentConfig)
			commandLine.printConfig( nn, mm );
	}

	MPI_Barrier( MPI_COMM_WORLD );

	MPIHelper MpiH( worldSize, nFolds, commandLine.ensThreads, commandLine.rfThreads, false, verboseMPI );
	if (rank == 0)
		MpiH.verbose();
	MPI_Barrier( MPI_COMM_WORLD );

	MpiH.broadcastGlobVars( rank, &(commonParams.nn), &(commonParams.mm), &nFolds );
	MpiH.broadcastMtrys( rank, gridParams );
	commonParams.nFolds = nFolds;
	MPI_Barrier( MPI_COMM_WORLD );

	Folds * foldManager 	= nullptr;
	double * class1Prob		= nullptr;	// minority (positive) output vector
	double * class2Prob		= nullptr;	// majority (negative) output vector

	if (rank == 0) {
		if (verboseLevel > VERBSILENT)
			std::cout << TXT_BIBLU << "Computing or generating folds" << TXT_NORML << std::endl;
		foldManager = new Folds( nFolds, nn, yy.data(), ff, commandLine.simulate || commandLine.foldFilename.empty() );
		foldManager->computeFolds();
		if (verboseLevel == VERBALL) foldManager->verbose();

		class1Prob = new double[nn];	checkPtr<double>( class1Prob, __FILE__, __LINE__ );
		class2Prob = new double[nn];	checkPtr<double>( class2Prob, __FILE__, __LINE__ );
		std::fill( class1Prob, class1Prob + nn, 0 );
		std::fill( class2Prob, class2Prob + nn, 0 );
	}

	MPI_Barrier( MPI_COMM_WORLD );

	hyperSMURF_MPI smurfer( rank, worldSize, &commonParams, gridParams, foldManager, &MpiH, xx, yy, class1Prob, class2Prob );
	smurfer.initStandard();

	MPI_Barrier( MPI_COMM_WORLD );

	Timer ttt;
	if (rank == 0)
		ttt.startTime();
	smurfer.smurfIt();
	if (rank == 0)
		ttt.endTime();

	if ((rank == 0) && (timeFilename != "")) {
		// Append computation time to log file, if specified by option --tttt
		std::ofstream timeFile( timeFilename.c_str(), std::ios::out | std::ios::app );
		timeFile << "#Working procs: " << 1
				<< " - #ensThreads: " << commonParams.nThr
				<< " - #rfThreads: " << commonParams.rfThr
				<< " - #folds: " << nFolds
				<< " - #parts: " << gridParams[0].nParts
				<< " - #fp: " << gridParams[0].fp
				<< " - #ratio: " << gridParams[0].ratio
				<< " - #nTrees: " << gridParams[0].nTrees
				<< " - #mtry: " << gridParams[0].mtry
				<< " - Computation time: " << ttt.duration()
				<< std::endl;
		timeFile.close();
	}

	if (rank == 0) {
		// CROSS-VALIDATION AND TESTING
		if ((commonParams.wmode == MODE_CV) | (commonParams.wmode == MODE_PREDICT)) {
			if (verboseLevel > VERBSILENT)
				std::cout << TXT_BIBLU << "Saving to file" << TXT_NORML << std::endl;
			if (commandLine.simulate)
				saveToFile( class1Prob, class2Prob, nn, &yy, outfilename );
			else {
				std::vector<uint32_t> temp;
				temp.clear();
				saveToFile( class1Prob, class2Prob, nn, &temp, outfilename );
			}
		}
	}

	std::cout << TXT_BIGRN << "Rank " << rank << " done!" << TXT_NORML << std::endl;

	if (rank == 0) {
		delete[] class2Prob;
		delete[] class1Prob;
	}

	MPI_Finalize();
	return 0;

}
