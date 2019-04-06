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
#include <omp.h>

#include "HyperSMURFUtils.h"
#include "ArgHandler_new.h"
#include "fileImport.h"
#include "folds.h"
#include "testtraindivider.h"
#include "partition.h"
#include "hyperSMURF.h"


int main( int argc, char **argv ) {
	// New ArgHandle for grid search. Parameters of each run are read from json (no more command line args)
	// and stored into a struct
	std::vector<GridParams> gridParams;
	ArgHandle commandLine( argc, argv, gridParams );
	commandLine.processCommandLine( 0 );

	CommonParams commonParams;
	size_t nn					= commonParams.nn			= commandLine.n;
	size_t mm					= commonParams.mm			= commandLine.m;
	uint32_t nFolds				= commonParams.nFolds		= commandLine.nFolds;
	const uint32_t seed			= commonParams.seed			= commandLine.seed;
	const uint32_t verboseLevel = commonParams.verboseLevel	= commandLine.verboseLevel;
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

	omp_set_num_threads( commonParams.nThr );

	std::vector<uint32_t> yy( nn );					// Labels
	std::vector<uint32_t> ff( nn );					// Folds
	std::vector<double>   xx( (mm + 1) * nn );		// Data
	if (verboseLevel > VERBSILENT)
		 std::cout << TXT_BIBLU << "Reading or generating data" << TXT_NORML << std::endl;
	if (!commandLine.simulate) {
		Importer::import( &commandLine, xx, yy, ff, &nFolds );
		nn = yy.size();
		mm = xx.size() / yy.size();
		std::for_each( yy.begin(), yy.end(), [&xx]( uint32_t nnn ) mutable { if (nnn > 0) xx.push_back( 1.0 ); else xx.push_back( 2.0 ); } );
		commonParams.nn = nn;
		commonParams.mm = mm;
		commonParams.nFolds = nFolds;
	}
	else
		generateRandomSet( nn, mm, xx, yy, commandLine.prob, seed );

	if (commonParams.customCV && (gridParams.size() != nFolds)) {
		std::cout << TXT_BIRED << "Mismatch between nFolds and overidden fold number from params.dat. Aborting..." << TXT_NORML << std::endl;
		exit( -1 );
	}

	commandLine.processMtry( mm );

	if (commandLine.printCurrentConfig)
		commandLine.printConfig( nn, mm );

	// output vectors
	double * class1Prob = new double[nn];	// minority (positive)
	double * class2Prob = new double[nn];	// majority (negative)
	std::fill( class1Prob, class1Prob + nn, 0 );
	std::fill( class2Prob, class2Prob + nn, 0 );

	// Fold generation. Random if no fold file or during simulations
	if (verboseLevel > VERBSILENT)
		std::cout << TXT_BIBLU << "Computing or generating folds" << TXT_NORML << std::endl;
	Folds foldManager( nFolds, nn, yy.data(), ff, commandLine.simulate || commandLine.foldFilename.empty() );
	foldManager.computeFolds();
	if (verboseLevel == VERBALL) foldManager.verbose();

	hyperSMURF smurfer( &commonParams, gridParams, &foldManager, xx, yy, class1Prob, class2Prob );
	smurfer.initStandard();

	Timer ttt;
	ttt.startTime();
	smurfer.smurfIt();
	ttt.endTime();

	if (timeFilename != "") {
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
	std::cout << TXT_BIGRN << "Done!" << TXT_NORML << std::endl;
	delete[] class2Prob;
	delete[] class1Prob;
	return 0;
}
