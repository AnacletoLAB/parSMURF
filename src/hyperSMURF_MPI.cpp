// parSMURF
// Alessandro Petrini, 2018-2019
#include "hyperSMURF_MPI.h"

hyperSMURF_MPI::hyperSMURF_MPI( const int rank,  const int worldSize, const CommonParams * const commonParams, std::vector<GridParams> & gridParams,
	Folds * const foldManager, const MPIHelper * const MpiH, const std::vector<double> & x, const std::vector<uint32_t> & y,
	double * const class1Prob, double * const class2Prob ) :
			commonParams( commonParams ), gridParams( gridParams ), foldManager( foldManager ), MpiH( MpiH ),
			x( x ), y( y ), class1Prob( class1Prob ), class2Prob( class2Prob ), rank( rank ), worldSize( worldSize ) {
}

hyperSMURF_MPI::~hyperSMURF_MPI() {
	if (ttd != nullptr)
		delete ttd;
}


void hyperSMURF_MPI::initStandard() {
	nn				= commonParams->nn;
	mm				= commonParams->mm;
	nFolds			= commonParams->nFolds;
	nPart			= gridParams[0].nParts;
	numTrees		= gridParams[0].nTrees;
	fp				= gridParams[0].fp;
	ratio			= gridParams[0].ratio;
	k				= gridParams[0].k;
	mtry			= gridParams[0].mtry;
	seed			= commonParams->seed;
	verboseLevel	= commonParams->verboseLevel;
	nThr			= commonParams->nThr;
	rfThr			= commonParams->rfThr;
	wmode			= commonParams->wmode;
	woptimiz		= commonParams->woptimiz;
	rfVerbose		= commonParams->rfVerbose;
	verboseMPI		= commonParams->verboseMPI;
	noMtSender		= commonParams->noMtSender;
	inInternalCV	= false;
	forestDirname	= commonParams->forestDirname;

	if (rank == 0) {
		if (verboseLevel > VERBSILENT) std::cout << "\033[94;1mComputing training and test partitions" << TXT_NORML << std::endl;
		ttd = new TestTrainDivider( foldManager, wmode );
		if (verboseLevel == VERBALL) ttd->verbose();
	}

	LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " - created smurfer with config: nn= " << nn << " - mm= " << mm << " - nFolds= " << nFolds
		<< " - nPart= " << nPart << " - numTrees= " << numTrees << " - fp= " << fp << " - ratio= " << ratio << " - k= " << k
		<< " - mtry= " << mtry << " - seed= " << seed << " - verboseLevel= " << verboseLevel << " - nThr= " << nThr
		<< " - rfThr= " << rfThr << " - wMode= " << (uint32_t) wmode << " - rfVerbose= " << (uint32_t) rfVerbose
		<< " - verboseMPI= " << verboseMPI << " - noMtSender= " << (uint32_t) noMtSender << TXT_NORML;
}

void hyperSMURF_MPI::initInternalCV( uint32_t foldToJump ) {
	nn				= commonParams->nn;
	mm				= commonParams->mm;
	nFolds			= commonParams->nFolds - 1;
	seed			= commonParams->seed;
	verboseLevel	= commonParams->verboseLevel;
	nThr			= commonParams->nThr;
	rfThr			= commonParams->rfThr;
	wmode			= MODE_CV;
	woptimiz		= OPT_NO;
	rfVerbose		= commonParams->rfVerbose;
	verboseMPI		= commonParams->verboseMPI;
	noMtSender		= commonParams->noMtSender;
	inInternalCV	= true;

	this->foldToJump = foldToJump;
	tempcl1Prob = nullptr;
	tempcl2Prob = nullptr;

	if (rank == 0) {
		if (verboseLevel > VERBSILENT) std::cout << TXT_BIPRP << "Computing training and test partitions for internal CV - iteration " << foldToJump << TXT_NORML << std::endl;
		ttd = new TestTrainDivider( foldManager, wmode, foldToJump );
		if (verboseLevel == VERBALL) ttd->verbose();
	}

	LOG(TRACE) << TXT_BIGRN << "rank: " << rank << " - created INTERNAL smurfer on fold " << foldToJump << " with config: nn= " << nn << " - mm= " << mm << " - nFolds= " << nFolds
		<< " - seed= " << seed << " - verboseLevel= " << verboseLevel << " - nThr= " << nThr
		<< " - rfThr= " << rfThr << " - wMode= " << (uint32_t) wmode << " - rfVerbose= " << (uint32_t) rfVerbose << " - verboseMPI= " << verboseMPI
		<< " - noMtSender= " << (uint32_t) noMtSender << TXT_NORML;
}

void hyperSMURF_MPI::setGridParams( uint32_t idx ) {
	idxInGridParams = idx;
	nPart = gridParams[idx].nParts;
	numTrees = gridParams[idx].nTrees;
	fp = gridParams[idx].fp;
	ratio = gridParams[idx].ratio;
	k = gridParams[idx].k;
	mtry = gridParams[idx].mtry;

	LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " - set parameter in INTERNAL cv: idk= " << idxInGridParams
		<< " - nPart= " << nPart << " - numTrees= " << numTrees << " - fp= " << fp << " - ratio= " << ratio << " - k= " << k
		<< " - mtry= " << mtry << TXT_NORML;
}

void hyperSMURF_MPI::createPart() {}

bool hyperSMURF_MPI::parametersOptimizer( uint32_t foldToJump ) {
	std::string fileName = std::string( "fold" + std::to_string(foldToJump) + ".dat" );
	std::string tempFileName = std::string( "temp" + std::to_string(foldToJump) + ".dat" );
	std::string fileLine;
	std::string commandLine = std::string("python spearmint/spearmint-lite/spearmint-lite.py --method=GPEIOptChooser --grid-size=20000 --method-args=mcmc_iters=10,noiseless=0 --result=" + fileName + " ." + " --config " + commonParams->cfgFilename );
	size_t idxInGrid = 0;
	gridParams.clear();
	hyperSMURF_MPI hyper_inner( rank, worldSize, commonParams, gridParams, foldManager, MpiH, x, y, tempcl1Prob, tempcl2Prob );
	hyper_inner.initInternalCV( foldToJump );
	std::ofstream tempFile;

	for (size_t i = 0; i < 25; i++) {
		if (rank == 0) {
			int retVal = 0;
			do {
				int retVal = std::system( commandLine.c_str() );
			} while (retVal != 0);
		}
		MPI_Barrier( MPI_COMM_WORLD );
		// Reads the resultfile and look for lines beginning with P (Pending)
		std::ifstream resultFile( fileName.c_str(), std::ios::in );
		if (rank == 0) {
			tempFile.open( tempFileName.c_str(), std::ios::out );
		}
		while (std::getline( resultFile, fileLine )) {
			if (fileLine[0] == 'P') {
				GridParams tempGridParam;
				std::vector<std::string> splittedStr = split_str( fileLine );

				std::cout << "Rank " << rank << ": params read: " << splittedStr[2] << " " << splittedStr[3] << " " << splittedStr[4] << " "
					<< splittedStr[5] << " " << splittedStr[6] << " " << splittedStr[7] << std::endl;

				tempGridParam.nParts	= atoi( splittedStr[2].c_str() );
				tempGridParam.fp		= atoi( splittedStr[3].c_str() );
				tempGridParam.ratio		= atoi( splittedStr[4].c_str() );
				tempGridParam.k			= atoi( splittedStr[5].c_str() );
				tempGridParam.nTrees	= atoi( splittedStr[6].c_str() );
				tempGridParam.mtry		= atoi( splittedStr[7].c_str() );
				gridParams.push_back( tempGridParam );

				// Run internal CV with the pending results
				if (rank == 0) {
					std::fill( tempcl1Prob, tempcl1Prob + nn, 0 );
					std::fill( tempcl2Prob, tempcl2Prob + nn, 0 );
				}
				MPI_Barrier( MPI_COMM_WORLD );
				hyper_inner.setGridParams( idxInGrid );
				hyper_inner.createPart();
				hyper_inner.smurfIt();

				// write the results removing the pending experiment
				if (rank == 0) {
					tempFile << std::to_string( -(gridParams[i].auprc) ) << " 0 " << splittedStr[2] << " " << splittedStr[3]
						<< " " << splittedStr[4] << " " << splittedStr[5] << " " << splittedStr[6] << " " << splittedStr[7]
						<< std::endl;
				}
				idxInGrid++;

			} else {
				if (rank == 0) {
					tempFile << fileLine << std::endl;
				}
			}
		}
		resultFile.close();
		tempFile.close();
		if (rank == 0) {
			std::rename( tempFileName.c_str(), fileName.c_str() );
		}
		MPI_Barrier( MPI_COMM_WORLD );
	}
}

uint32_t hyperSMURF_MPI::findMaxAuprc( const std::vector<GridParams> & gridParams ) {
	float maxauprc = 0;
	uint32_t maxauprcIdx = 0;
	for (uint32_t i = 0; i < gridParams.size(); i++) {
		if (gridParams[i].auprc > maxauprc) {
			maxauprc = gridParams[i].auprc;
			maxauprcIdx = i;
		}
	}
	LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " found max auprc at idx" << maxauprcIdx <<TXT_NORML;
	return maxauprcIdx;
}

void hyperSMURF_MPI::smurfIt() {
	if ((rank == 0) && (woptimiz != OPT_NO)) {
		tempcl1Prob = new double[nn];
		tempcl2Prob = new double[nn];
	}

	uint32_t startingFold = 0;
	uint32_t endingFold = nFolds;
	if ((!inInternalCV) && (commonParams->minFold != -1))
		startingFold = commonParams->minFold;
	if ((!inInternalCV) && (commonParams->maxFold != -1))
		endingFold = commonParams->maxFold;

	for (uint32_t currentFold = startingFold; currentFold < endingFold; currentFold++) {
		if (!inInternalCV)
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " starting fold " << currentFold << TXT_NORML;
		else
			LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " starting fold " << currentFold << TXT_NORML;

		// Inception...
		if (((wmode == MODE_CV) | (wmode == MODE_TRAIN)) && (woptimiz != OPT_NO)) {
			if (woptimiz == OPT_GRID) {
				hyperSMURF_MPI hyper_inner( rank, worldSize, commonParams, gridParams, foldManager, MpiH, x, y, tempcl1Prob, tempcl2Prob );
				hyper_inner.initInternalCV( currentFold );

				std::string statFileName = std::string( "fold" + std::to_string(currentFold) + ".dat" );

				for (uint32_t i = 0; i < gridParams.size(); i++) {
					if (rank == 0) {
						std::fill( tempcl1Prob, tempcl1Prob + nn, 0 );
						std::fill( tempcl2Prob, tempcl2Prob + nn, 0 );
					}
					hyper_inner.setGridParams( i );
					LOG(TRACE) << TXT_BIGRN << "rank: " << rank << " starting smurf on gridParams " << i << TXT_NORML;
					hyper_inner.smurfIt();
					LOG(TRACE) << TXT_BIGRN << "rank: " << rank << " ending smurf on gridParams " << i << TXT_NORML;
					if (rank == 0) {
						std::ofstream tempFile;
						tempFile.open( statFileName.c_str(), std::ios::app );
						tempFile << std::to_string( i ) << " "
							<< std::to_string( gridParams[i].nParts ) << " " << std::to_string( gridParams[i].fp ) << " "
							<< std::to_string( gridParams[i].ratio ) << " " << std::to_string( gridParams[i].k ) << " "
							<< std::to_string( gridParams[i].nTrees ) << " " << std::to_string( gridParams[i].mtry ) << " "
							<< std::to_string( gridParams[i].auroc ) << " " << std::to_string( gridParams[i].auprc ) << " "
							<< std::endl;
						tempFile.close();
					}
				}
			} else if (woptimiz == OPT_AUTOGP)
				parametersOptimizer( currentFold );

			// Finding the maximum auprc for this fold and broadcast to working processes
			uint32_t maxAuprcIdx;
			if (rank == 0) {
				maxAuprcIdx = findMaxAuprc( gridParams );
				if (verboseMPI) {
					std::cout << TXT_BIPRP << "Broadcasting max auprc index: idx = " << maxAuprcIdx << TXT_NORML << std::endl;
				}
			}
			LOG(TRACE) << TXT_BIBLU << "Broadcasting max auprc index" << TXT_NORML;
			MPI_Bcast( &maxAuprcIdx, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Max auprc idx: " << maxAuprcIdx << TXT_NORML;

			nPart		= gridParams[maxAuprcIdx].nParts;
			fp			= gridParams[maxAuprcIdx].fp;
			ratio		= gridParams[maxAuprcIdx].ratio;
			k			= gridParams[maxAuprcIdx].k;
			numTrees	= gridParams[maxAuprcIdx].nTrees;
			mtry		= gridParams[maxAuprcIdx].mtry;

			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " - updating best parameters idx= " << maxAuprcIdx
				<< " - nPart= " << nPart << " - numTrees= " << numTrees << " - fp= " << fp << " - ratio= " << ratio << " - k= " << k
				<< " - mtry= " << mtry << TXT_NORML;

			// Debug print
			if (rank == 0)
				std::cout << TXT_BIGRN << "Setting values of best AUPRC: nPart: " << nPart << " - fp: " << fp << " - ratio: " << ratio <<
					" - k: " << k << " - nTrees: " << numTrees << " - mtry: " << mtry << TXT_NORML << std::endl;
		}

		if (!inInternalCV && (commonParams->customCV)) {
			// Here we set the parameters from the external file, overriding the defaults
			nPart		= gridParams[currentFold].nParts;
			fp			= gridParams[currentFold].fp;
			ratio		= gridParams[currentFold].ratio;
			k			= gridParams[currentFold].k;
			numTrees	= gridParams[currentFold].nTrees;
			mtry		= gridParams[currentFold].mtry;

			std::cout << TXT_BIGRN << "Setting custom values: nPart: " << nPart << " - fp: " << fp << " - ratio: " << ratio <<
				" - k: " << k << " - nTrees: " << numTrees << " - mtry: " << mtry << TXT_NORML << std::endl;
		}

		if (!inInternalCV)
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " updating chunks " << TXT_NORML;
		else
			LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " updating chunks " << TXT_NORML;
		const_cast<MPIHelper * const>(MpiH)->updateChunks( nPart );

		MPI_Barrier( MPI_COMM_WORLD );

		std::vector<PartitionMPI> partMPI;
		uint32_t ack = 0;

		uint32_t	testSize;		// # esempi in testset (== lunghezza della linea della matrice)
		uint32_t	trngMaxSize;	// # esempi massimo in trng set
		uint32_t	testDataSize;	// dimensione della matrice test (== (testSize * (m+1)))
		uint32_t	trngDataSize;	// dimensione della matrice trng (== (trngSize * (m+1) + header))
		double	*	testData = nullptr;
		double	*	tempProb1 = nullptr;
		double	*	tempProb2 = nullptr;

		if (rank == 0) {
			testSize = ttd->testPosNum[currentFold] + ttd->testNegNum[currentFold];
			trngMaxSize = (fp + 1) * (ratio + 1) * (foldManager->maxPos) * (foldManager->nFolds - 1);
			if (ratio == 0)
				trngMaxSize = ( (fp + 1) * foldManager->maxPos + foldManager->maxNeg ) * (foldManager->nFolds - 1);

			///A1/// In coerenza con quanto fatto in partition.cpp
			if (wmode == MODE_TRAIN) {
				trngMaxSize = (ratio == 0 ?
						( (fp + 1) * foldManager->maxPos + foldManager->maxNeg ) : // fix per ratio = 0
						(fp + 1) * (ratio + 1) * foldManager->maxPos );
			}
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Calculating... testSize= " << testSize << " - trngMaxSize= " << trngMaxSize << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Calculating... testSize= " << testSize << " - trngMaxSize= " << trngMaxSize << TXT_NORML;
		}

		// Broadcasting sizes...
		if ((rank == 0) & (verboseMPI)) {
			std::cout << "\033[35;1mBroadcasting: testSize = " << testSize << " - trngMaxSize = " << trngMaxSize << ".\033[0m" << std::endl;
		}
		MPI_Barrier( MPI_COMM_WORLD );
		MPI_Bcast( &testSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
		MPI_Barrier( MPI_COMM_WORLD );
		MPI_Bcast( &trngMaxSize, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD );
		MPI_Barrier( MPI_COMM_WORLD );
		if (!inInternalCV)
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Broadcasting... testSize= " << testSize << " - trngMaxSize= " << trngMaxSize << TXT_NORML;
		else
			LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Broadcasting... testSize= " << testSize << " - trngMaxSize= " << trngMaxSize << TXT_NORML;

		if (verboseMPI) {
			std::cout << "\033[35;1mRank " << rank << " received from broadcast: testSize= " << testSize << " - trngMaxSize= " << trngMaxSize << ".\033[0m" << std::endl;
		}

		testDataSize = testSize * (mm + 1);
		trngDataSize = trngMaxSize * (mm + 1) + TRNGHEADERSIZE;	// This is the maximum size for the current fold
		if (!inInternalCV)
			LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Calculating... testDataSize= " << testDataSize << " - trngDataSize= " << trngDataSize << TXT_NORML;
		else
			LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Calculating... testDataSize= " << testDataSize << " - trngDataSize= " << trngDataSize << TXT_NORML;

		if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
			testData = new double[testDataSize]; checkPtr<double>( testData, __FILE__, __LINE__ );
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Allocated testData" << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Allocated testData" << TXT_NORML;
		}

		if (rank == 0) {
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Filling partMPI vector" << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Filling partMPI vector" << TXT_NORML;
			for (uint32_t currentPart = 0; currentPart < nPart; currentPart++) {
				partMPI.push_back( PartitionMPI( foldManager, ttd, nPart, fp, ratio, mm, nn, currentPart, currentFold, MpiH->numberOfChunks ) );
			}
			if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
				if (!inInternalCV)
					LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Filling testData on partMPI[0]" << TXT_NORML;
				else
					LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Filling testData on partMPI[0]" << TXT_NORML;
				partMPI[0].fillTestData( x.data(), testData );
			}
		}

		// Broadcasting testData...
		if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
			if ((rank == 0) & (verboseMPI)) {
				std::cout << "\033[35;1mBroadcasting: testData matrix for total size: " << testDataSize << ".\033[0m" << std::endl;
			}
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Broadcasting testData for testDataSize= " << testDataSize << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Broadcasting testData for testDataSize= " << testDataSize << TXT_NORML;

			MPI_Bcast( testData, testDataSize, MPI_DOUBLE, 0, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			if ((rank == 0) & (verboseMPI)) {
				std::cout << "\033[35;1mBroadcasting testData matrix complete.\033[0m" << std::endl;
			}

			// Ogni processo MPI crea i due vettory temporanei delle prob calcolate sul test set.
			// Quelli sul rank 0 saranno la destinazione delle accumulazioni dei vettori dei working procs.
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Allocating tempProb1 and tempProb2 for testSize= " << testSize << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Allocating tempProb1 and tempProb2 for testSize= " << testSize << TXT_NORML;
			tempProb1 = new double[testSize];	checkPtr<double>( tempProb1, __FILE__, __LINE__ );
			tempProb2 = new double[testSize];	checkPtr<double>( tempProb2, __FILE__, __LINE__ );
			std::fill( tempProb1, tempProb1 + testSize, 0.0 );
			std::fill( tempProb2, tempProb2 + testSize, 0.0 );
		}

		//////// CV and TRAIN
		if ((wmode == MODE_CV) | (wmode == MODE_TRAIN)) {
			// Start of processing for master rank: sending data to worker procs
			if (rank == 0) {

				if (noMtSender) {
					if (!inInternalCV)
						LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Allocating trngData for trngDataSize= " << trngDataSize << TXT_NORML;
					else
						LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Allocating trngData for trngDataSize= " << trngDataSize << TXT_NORML;

					double	*	trngData = new double[trngDataSize];		checkPtr<double>( trngData, __FILE__, __LINE__ );
					MPI_Status ackStatus;

					// Per costruzione, il primo chunk ha il maggior numero di partizioni assegnate
					// quindi uso questo come contatore per il loop piu' esterno
					for( uint32_t i = 0; i < MpiH->numberOfPartsPerChunk[0]; i++ ) {
						for( uint32_t j = 0; j < MpiH->numberOfWorkingProcs; j++ ) {
							if (i >= MpiH->numberOfPartsPerChunk[j])
								continue;
							uint32_t currentPart = MpiH->startingPartition[j] + i;
							partMPI[currentPart].assignedToChunk = j;

							// Assembling the header
							trngData[0] = (double) partMPI[currentPart].currentFold;	// Not used
							trngData[1] = (double) currentPart;							// Not used
							trngData[2] = (double) partMPI[currentPart].trngSize;
							trngData[3] = (double) partMPI[currentPart].posForOverSmpl;
							trngData[4] = (double) partMPI[currentPart].posToBeGenerated;

							if (!inInternalCV)
								LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Filling training data for fold= " << partMPI[currentPart].currentFold
									<< " - partition= "<< currentPart << " - trngSize= " << partMPI[currentPart].trngSize << " - posForOverSmpl= "
									<< partMPI[currentPart].posForOverSmpl << " - posToBeGenerated= " << partMPI[currentPart].posToBeGenerated << TXT_NORML;
							else
								LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Filling training data for fold= " << partMPI[currentPart].currentFold
									<< " - partition= "<< currentPart << " - trngSize= " << partMPI[currentPart].trngSize << " - posForOverSmpl= "
									<< partMPI[currentPart].posForOverSmpl << " - posToBeGenerated= " << partMPI[currentPart].posToBeGenerated << TXT_NORML;

							// Filling the trngData matrix
							trngData += TRNGHEADERSIZE;
							partMPI[currentPart].fillTrngData( x.data(), trngData );
							trngData -= TRNGHEADERSIZE;

							// And send everything to worker processes
							if (verboseMPI) {
								std::cout << "\033[35;1mSend trng data of part " << trngData[1] << " to process: " << partMPI[currentPart].assignedToChunk + 1 << " - size: " << trngDataSize << ".\033[0m" << std::endl;
							}
							if (!inInternalCV)
								LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Sending trngData for trngDataSize= " << trngDataSize
									<< " to working process " << partMPI[currentPart].assignedToChunk + 1 << TXT_NORML;
							else
								LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Sending trngData for trngDataSize= " << trngDataSize
									<< " to working process " << partMPI[currentPart].assignedToChunk + 1 << TXT_NORML;
							MPI_Send( trngData, /*partMPI[currentPart].trngSize * (mm + 1) + TRNGHEADERSIZE*/trngDataSize, MPI_DOUBLE,
								partMPI[currentPart].assignedToChunk + 1, 0, MPI_COMM_WORLD );
							MPI_Recv( &ack, 1, MPI_UNSIGNED, partMPI[currentPart].assignedToChunk + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &ackStatus );
							if (verboseMPI) {
								std::cout << "\033[35;1mMaster proc received ack for part " << trngData[1] << " from process: " << partMPI[currentPart].assignedToChunk + 1 << ".\033[0m" << std::endl;
							}
						}
					}
					delete[] trngData;

				} else {

					std::thread *tt_send = new std::thread[MpiH->numberOfWorkingProcs];
					for (uint32_t i = 0; i < MpiH->numberOfWorkingProcs; ++i) {
						if (verboseMPI) {
							std::cout << "\033[35;1mStarting sending thread: " << i << ".\033[0m" << std::endl;
						}
						if (!inInternalCV)
							LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Starting sender thread " << i << TXT_NORML;
						else
							LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Starting sender thread " << i << TXT_NORML;
					 	tt_send[i] = std::thread( sendThrd, MpiH->numberOfWorkingProcs, i, mm, trngDataSize, MpiH, std::ref(partMPI), std::ref(x), verboseMPI );
					}

					std::this_thread::yield();

					for (uint32_t i = 0; i < MpiH->numberOfWorkingProcs; ++i) {
						tt_send[i].join();
						if (!inInternalCV)
							LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Sender thread " << i << " has joinded" << TXT_NORML;
						else
							LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Sender thread " << i << " has joinded" << TXT_NORML;
					}

					delete[] tt_send;
				}
			}
			// working processes receive data from master and hyperSMURFs it
			// by delegating to working threads
			else {
				std::mutex p_accumulLock;
				std::thread *tt = new std::thread[MpiH->numberOfThreadsPerProc];
				uint32_t partitionForThisProc = MpiH->numberOfPartsPerChunk[rank - 1];

				for (uint32_t i = 0; i < MpiH->numberOfThreadsPerProc; ++i) {
					if (verboseMPI) {
						std::cout << "\033[35;1mrank: " << rank << " starting thread: " << i << ".\033[0m" << std::endl;
					}
					if (!inInternalCV)
						LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Starting receiver thread " << i << TXT_NORML;
					else
						LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Starting receiver thread " << i << TXT_NORML;
				 	tt[i] = std::thread( recvThrd, MpiH->numberOfThreadsPerProc, i, rank, partitionForThisProc, trngDataSize,
						mm, nn, fp, k, numTrees, mtry, rfThr, seed, testSize, testData, tempProb1, tempProb2, &p_accumulLock, verboseMPI,
					 	wmode, forestDirname );
				}

				std::this_thread::yield();

				for (uint32_t i = 0; i < MpiH->numberOfThreadsPerProc; ++i) {
					tt[i].join();
					if (!inInternalCV)
						LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Receiver thread " << i << " has joinded" << TXT_NORML;
					else
						LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Receiver thread " << i << " has joinded" << TXT_NORML;
				}

				delete[] tt;
			}
		}

		///////// PREDICT
		else if (wmode == MODE_PREDICT) {
			// i worker process hanno gia' il test set
			// devo solo lanciare i thread sui worker process.
			// i thread devono fare le stesse operazioni della versione mt + accumulazione in locale
			// la gather dei risultati avviene successivamente ed e' in comune con quanto scritto per la CV
			if (rank == 0) {
				MPI_Status ackStatus;

				for( uint32_t i = 0; i < MpiH->numberOfPartsPerChunk[0]; i++ ) {
					for( uint32_t j = 0; j < MpiH->numberOfWorkingProcs; j++ ) {
						if (i >= MpiH->numberOfPartsPerChunk[j])
							continue;
						uint32_t currentPart = MpiH->startingPartition[j] + i;
						partMPI[currentPart].assignedToChunk = j;

						if (verboseMPI) {
							std::cout << "\033[35;1mSend partition number " << currentPart << " to process: " << partMPI[currentPart].assignedToChunk + 1 << ".\033[0m" << std::endl;
						}
						MPI_Send( &currentPart, 1, MPI_UNSIGNED, partMPI[currentPart].assignedToChunk + 1, 0, MPI_COMM_WORLD );
						MPI_Recv( &ack, 1, MPI_UNSIGNED, partMPI[currentPart].assignedToChunk + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &ackStatus );
						if (verboseMPI) {
							std::cout << "\033[35;1mMaster proc received ack for part " << ack << " from process: " << partMPI[currentPart].assignedToChunk + 1 << ".\033[0m" << std::endl;
						}
					}
				}

			} else {
				std::mutex p_accumulLock;
				std::thread *tt = new std::thread[MpiH->numberOfThreadsPerProc];
				uint32_t partitionForThisProc = MpiH->numberOfPartsPerChunk[rank - 1];

				for (uint32_t i = 0; i < MpiH->numberOfThreadsPerProc; ++i) {
					if (verboseMPI) {
						std::cout << "\033[35;1mrank: " << rank << " starting prediction thread: " << i << ".\033[0m" << std::endl;
					}
				 	tt[i] = std::thread( predictRecvThrd, MpiH->numberOfThreadsPerProc, i, rank, partitionForThisProc,
						mm, numTrees, mtry, rfThr, seed, testSize, testData, tempProb1, tempProb2, &p_accumulLock, verboseMPI,
						forestDirname );
				}

				std::this_thread::yield();

				for (uint32_t i = 0; i < MpiH->numberOfThreadsPerProc; ++i)
					tt[i].join();

				delete[] tt;
			}
		}

		MPI_Barrier( MPI_COMM_WORLD );

		///A1/// CV e PREDICT
		if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
			double * class1ProbGath;
			double * class2ProbGath;
			if (rank == 0) {
				if (!inInternalCV)
					LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Allocating gather arrays for 2 x " << testSize * worldSize << TXT_NORML;
				else
					LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Allocating gather arrays for 2 x " << testSize * worldSize << TXT_NORML;
				class1ProbGath = new double[testSize * worldSize];		checkPtr<double>( class1ProbGath, __FILE__, __LINE__ );
				class2ProbGath = new double[testSize * worldSize];		checkPtr<double>( class2ProbGath, __FILE__, __LINE__ );
				std::fill( class1ProbGath, class1ProbGath + (testSize * worldSize), 0.0 );
				std::fill( class2ProbGath, class2ProbGath + (testSize * worldSize), 0.0 );
			}
			MPI_Barrier( MPI_COMM_WORLD );
			if ((rank == 0) & (verboseMPI)) {
				std::cout << "\033[35;1mGathering tempProb1: size: " << testSize << ".\033[0m" << std::endl;
			}
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Gathering in class1ProbGath " << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Gathering in class1ProbGath " << TXT_NORML;
			MPI_Gather( tempProb1, testSize, MPI_DOUBLE, class1ProbGath, testSize, MPI_DOUBLE, 0, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			if ((rank == 0) & (verboseMPI)) {
				std::cout << "\033[35;1mGathering tempProb2: size: " << testSize << ".\033[0m" << std::endl;
			}
			if (!inInternalCV)
				LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Gathering in class2ProbGath " << TXT_NORML;
			else
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Gathering in class2ProbGath " << TXT_NORML;
			MPI_Gather( tempProb2, testSize, MPI_DOUBLE, class2ProbGath, testSize, MPI_DOUBLE, 0, MPI_COMM_WORLD );
			MPI_Barrier( MPI_COMM_WORLD );
			if ((rank == 0) & (verboseMPI)) {
				std::cout << "\033[35;1mGathering complete.\033[0m" << std::endl;
			}
			if (rank == 0) {
				if (!inInternalCV)
					LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Cumulating results " << TXT_NORML;
				else
					LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Cumulating results " << TXT_NORML;
				for (uint32_t i = 0; i < testSize; i++) {
					for (uint32_t j = 1; j < (uint32_t)worldSize; j++) {
						class1ProbGath[i] += class1ProbGath[i + j * testSize];
						class2ProbGath[i] += class2ProbGath[i + j * testSize];
					}
				}

				if (!inInternalCV)
					LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Copying results in output vector" << TXT_NORML;
				else
					LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Copying results in output vector" << TXT_NORML;
				uint32_t cc = 0;
				for (uint32_t i = 0; i < partMPI[0].testPosNum; i++) {
					class1Prob[partMPI[0].testPos[i]] += class1ProbGath[cc];
					class2Prob[partMPI[0].testPos[i]] += class2ProbGath[cc];
					cc++;
				}
				for (uint32_t i = 0; i < partMPI[0].testNegNum; i++) {
					class1Prob[partMPI[0].testNeg[i]] += class1ProbGath[cc];
					class2Prob[partMPI[0].testNeg[i]] += class2ProbGath[cc];
					cc++;
				}

				// Scale predictions
				double divider = 1.0 / (double)nPart;
				std::cout << "Scaling predictions - nPart: " << nPart << std::endl;
				for (uint32_t i = 0; i < partMPI[0].testPosNum; i++) {
					class1Prob[partMPI[0].testPos[i]] *= divider;
					class2Prob[partMPI[0].testPos[i]] *= divider;
				}
				for (uint32_t i = 0; i < partMPI[0].testNegNum; i++) {
					class1Prob[partMPI[0].testNeg[i]] *= divider;
					class2Prob[partMPI[0].testNeg[i]] *= divider;
				}

				delete[] class2ProbGath;
				delete[] class1ProbGath;
			}

			delete[] tempProb2;
			delete[] tempProb1;
			delete[] testData;
		}
	}

	// ...and average
	if (rank == 0) {
		if ((wmode == MODE_CV) | (wmode == MODE_PREDICT)) {
			if ((rank == 0) && (verboseLevel > VERBSILENT)) {
				if (!inInternalCV)
					std::cout << TXT_BIPRP;
				else
					std::cout << TXT_BIBLU;
				std::cout << "\033[94;1mComputing the average\033[0m" << std::endl;
			}


			if (inInternalCV) {
				size_t	tempSize = 0;
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Accumulating results: counting testset" << TXT_NORML;
				// Count how many examples
				for (uint32_t i = 0; i < nFolds; i++) {
					tempSize += (ttd->testNegNum[i] + ttd->testPosNum[i]);
				}
				// generate teporary label and prediction vectors
				LOG(TRACE) << TXT_BIPRP << "rank: " << rank << " Accumulating results: generating temp label and preds vectors" << TXT_NORML;
				std::vector<uint32_t> tempLabels(tempSize);
				std::vector<double> tempPreds(tempSize);
				size_t tempIdx = 0;
				for (uint32_t i = 0; i < nFolds; i++) {
					std::for_each(ttd->testPosIdx[i], ttd->testPosIdx[i] + ttd->testPosNum[i], [&](uint32_t val) mutable {
						tempLabels[tempIdx] = y[val];
						tempPreds[tempIdx++] = class1Prob[val];
					});
					std::for_each(ttd->testNegIdx[i], ttd->testNegIdx[i] + ttd->testNegNum[i], [&](uint32_t val) mutable {
						tempLabels[tempIdx] = y[val];
						tempPreds[tempIdx++] = class1Prob[val];
					});
				}
				
				// evaluate auroc and auprc
				Curves evalauprc(tempLabels, tempPreds.data());
				gridParams[idxInGridParams].auroc = evalauprc.evalAUROC_ok();
				gridParams[idxInGridParams].auprc = evalauprc.evalAUPRC();
				std::cout << "AUROC: " << gridParams[idxInGridParams].auroc << " - AUPRC: " << gridParams[idxInGridParams].auprc << std::endl;

			} else {
				////  Removed, since now averaging is done by fold
				// LOG(TRACE) << TXT_BIBLU << "rank: " << rank << " Accumulating results: averaging" << TXT_NORML;
				// double divider = 1.0 / (double)nPart;
				// std::for_each( class1Prob, class1Prob + nn, [divider]( double &nnn ) mutable { nnn *= divider; } );
				// std::for_each( class2Prob, class2Prob + nn, [divider]( double &nnn ) mutable { nnn *= divider; } );
				Curves evalauprc(y, class1Prob);
				// BUG: Do not invert evalAUROC_ok() and evalAUPRC()...
				double auroc = evalauprc.evalAUROC_ok();
				double auprc = evalauprc.evalAUPRC();
				std::cout << "AUROC: " << auroc << " - AUPRC: " << auprc << std::endl;
			}
		}

		if (woptimiz != OPT_NO) {
			delete[] tempcl2Prob;
			delete[] tempcl1Prob;
		}

	}

	MPI_Barrier( MPI_COMM_WORLD );

}
