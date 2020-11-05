// parSMURF
// Alessandro Petrini, 2018-2019
#include "ArgHandler_new.h"
#include <getopt.h>

ArgHandle::ArgHandle( int argc, char **argv, std::vector<GridParams> &gridParams ) :
		gridParams( gridParams ),
		dataFilename( "" ), foldFilename( "" ), labelFilename( "" ), outFilename( "" ), forestDirname( "" ), timeFilename( "" ), extConfigFilename( "" ),
		m( 0 ), n( 0 ), prob( 0.0 ), seed( 0 ), verboseLevel(0),
		ensThreads( 0 ), rfThreads( 0 ), wmode( MODE_CV ), woptimiz( OPT_NO ),
		generateRandomFold( false ), readNFromFile( false ), simulate( false ), verboseMPI( false ), noMtSender( false ),
		externalConfig( false ), printCurrentConfig( false ),
		minFold( -1 ), maxFold( -1 ),
		argc( argc ), argv( argv ), mode( "" ), optim( "" ) {
}

ArgHandle::~ArgHandle() {}

void ArgHandle::processCommandLine( int rank ) {
	if (rank == 0)
		printLogo();

	char const *short_options = "j:u:h:M";
	const struct option long_options[] = {

		{ "cfg",			required_argument, 0, 'j' },
		{ "printCfg",		no_argument,       0, 'u' },
		{ "help",			no_argument,	   0, 'h' },
		{ "cite-me",		no_argument,	   0, 'M' },
		{ 0, 0, 0, 0 }
	};

	while (1) {
		int option_index = 0;
		int c = getopt_long( argc, argv, short_options, long_options, &option_index );

		if (c == -1) {
			break;
		}

		switch (c) {
		case 'j':
			extConfigFilename = std::string( optarg );
			externalConfig = true;
			break;

		case 'u':
			printCurrentConfig = true;
			break;

		case 'h':
			if (rank == 0)
				displayHelp();
			exit( 0 );
			break;
		
		case 'M':
			if (rank == 0)
				citeMe();
			exit(0);
			break;

		default:
			break;
		}
	}

	// In gridSMURF it is not possible to specify configuration arguments from command line.
	// Instead, configuration is read from a json file
	if (externalConfig) {
		if ( rank == 0 )
			std::cout << TXT_BIYLW << "Parsing cfg file..." << TXT_NORML << std::endl;
		jsonImport( extConfigFilename );
	} else {
		if ( rank == 0 )
			std::cout << TXT_BIRED << "parSMURF requires a configuration file in json format (--cfg)." << TXT_NORML << std::endl;
		exit( -1 );
	}

	checkCommonConfig( rank );
	checkConfig( rank );
}

void ArgHandle::jsonImport( std::string cfgFilename ) {
	// try {
    //     jsoncons::strict_parse_error_handler err_handler;
	// 	jsCfg = jsoncons::json::parse_file( cfgFilename, err_handler );
	// } catch (const jsoncons::parse_error& e) {
	// 	std::cout << e.what() << std::endl;
	// }

	std::ifstream cfgJsonFile(cfgFilename.c_str());
	jsCfg = jsoncons::json::parse(cfgJsonFile);
	cfgJsonFile.close();

	jsoncons::json	exec;
	jsoncons::json	data;
	jsoncons::json	simulateJ;
	jsoncons::json	flds;
	jsoncons::json	params;

	exec				= getFromJson<jsoncons::json>( &jsCfg, "exec", NULL );
	data				= getFromJson<jsoncons::json>( &jsCfg, "data", NULL );
	simulateJ			= getFromJson<jsoncons::json>( &jsCfg, "simulate", NULL );
	flds				= getFromJson<jsoncons::json>( &jsCfg, "folds", NULL );
	params				= getFromJson<jsoncons::json>( &jsCfg, "params", NULL );

	dataFilename		= getFromJson<std::string>( &data, "dataFile", dataFilename );
	foldFilename		= getFromJson<std::string>( &data, "foldFile", foldFilename );
	labelFilename		= getFromJson<std::string>( &data, "labelFile", labelFilename );
	outFilename			= getFromJson<std::string>( &data, "outFile", outFilename );
	forestDirname		= getFromJson<std::string>( &data, "forestDir", forestDirname );

	bool savetime		= getFromJson<bool>( &exec, "saveTime", false );
	if (savetime)
		timeFilename	= getFromJson<std::string>( &exec, "timeFile", timeFilename );

	simulate			= getFromJson<bool>( &simulateJ, "simulation", false );
	if (simulate) {
		m				= getFromJson<uint32_t>( &simulateJ, "m", m );
		n				= getFromJson<uint32_t>( &simulateJ, "n", n );
		prob			= getFromJson<double>( &simulateJ, "prob", prob );
	}

	nFolds				= getFromJson<uint32_t>( &flds, "nFolds", nFolds );
	minFold				= getFromJson<uint32_t>( &flds, "startingFold", minFold );
	maxFold				= getFromJson<uint32_t>( &flds, "endingFold", maxFold );

	seed 				= getFromJson<uint32_t>( &exec, "seed", seed );
	verboseLevel		= getFromJson<uint32_t>( &exec, "verboseLevel", verboseLevel );
	ensThreads			= getFromJson<uint32_t>( &exec, "ensThrd", ensThreads );
	rfThreads			= getFromJson<uint32_t>( &exec, "rfThrd", rfThreads );

	verboseMPI			= getFromJson<bool>( &exec, "verboseMPI", verboseMPI );
	noMtSender			= getFromJson<bool>( &exec, "noMtSender", noMtSender );
	printCurrentConfig	= getFromJson<bool>( &exec, "printCfg", printCurrentConfig );
	mode				= getFromJson<std::string>( &exec, "mode", mode );
	optim				= getFromJson<std::string>( &exec, "optimizer", optim );

	fillParams( &params, gridParams );
}

void ArgHandle::checkCommonConfig( int rank ) {
	if (outFilename.empty()) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No output file name defined. Default used ('data':'outFile')." << TXT_NORML << std::endl;
		outFilename = std::string( "output.txt" );
	}

	if (!mode.compare("cv"))
		wmode = MODE_CV;
	else if (!mode.compare("train")) {
		wmode = MODE_TRAIN;
		nFolds = 1;
	}
	else if (!mode.compare("predict")) {
		wmode = MODE_PREDICT;
		nFolds = 1;
	}
	else if (mode.length() > 0) {
		if (rank == 0)
			std::cout << TXT_BIRED << "Invalid prediction mode. Please specify either 'cv', 'train' or 'predict' (default is 'cv')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (((wmode == MODE_TRAIN) | (wmode == MODE_PREDICT)) & (forestDirname.length() == 0)) {
		if (rank == 0)
			std::cout << TXT_BIRED << "When in training or prediction modes, specify the forest base directory ('data':'forestDir')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (((wmode == MODE_TRAIN) | (wmode == MODE_PREDICT)) & (foldFilename.length() > 0)) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "Ignoring fold filename in train or test mode." << TXT_NORML << std::endl;
		foldFilename = "";
	}

	if ((wmode == MODE_CV) & (foldFilename == "") & (nFolds < 2)) {
		if (rank == 0)
			std::cout << TXT_BIRED << "In cross-validation mode, specify at least two folds ('folds':'nFolds')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (!optim.compare("grid"))
		woptimiz = OPT_GRID;
	else if (!optim.compare("autogp"))
		woptimiz = OPT_AUTOGP;
	else if (!optim.compare("no"))
		woptimiz = OPT_NO;
	else if (optim.length() > 0) {
		if (rank == 0)
			std::cout << TXT_BIRED << "Invalid optimization mode. Please specify either 'no', 'grid' or 'autogp' (default is 'no')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (simulate && ((m == 0) | (n == 0))) {
		if (rank == 0)
			std::cout << TXT_BIRED << "Simualtion enabled: specify m and n ('simulate':'m' and 'simulate':'n')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (simulate && ((prob < 0) | (prob > 1))) {
		if (rank == 0)
			std::cout << TXT_BIRED << "Simulation: probabilty of positive class must be 0 < prob < 1." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (!simulate) {
		if (dataFilename.empty()) {
			if (rank == 0)
				std::cout << TXT_BIRED << "Matrix file undefined ('data':'dataFile')." << TXT_NORML << std::endl;
			exit(-1);
		}

		if (labelFilename.empty()) {
			if (rank == 0)
				std::cout << TXT_BIRED << "Label file undefined ('data':'labelFile')." << TXT_NORML << std::endl;
			exit(-1);
		}

		if (foldFilename.empty()) {
			if (rank == 0)
				std::cout << TXT_BIYLW << "No fold file name defined. Random generation of folds enabled ('data':'foldFile')." << TXT_NORML;
			generateRandomFold = true;
			if (nFolds == 0) {
				if (rank == 0)
					std::cout << TXT_BIYLW << " [nFold = 3 as default ('folds':'nFolds')]" << TXT_NORML;
				nFolds = 3;
			}
			std::cout << std::endl;
		}

		if (!foldFilename.empty() && (nFolds != 0)) {
			if (rank == 0)
				std::cout << TXT_BIYLW << "nFolds option ignored (mumble, mumble...)." << TXT_NORML << std::endl;
		}
	}

	if (simulate & (nFolds == 0)) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No number of folds specified. Using default setting: 3 ('folds':'nFolds')." << TXT_NORML << std::endl;
		nFolds = 3;
	}

	if (((wmode == MODE_TRAIN) | (wmode == MODE_PREDICT)) & (forestDirname.length() == 0)) {
		if (rank == 0)
			std::cout << TXT_BIRED << "When in training or prediction modes, specify the forest base directory ('data':'forestDir')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if ((wmode == MODE_CV) & (foldFilename == "") & (nFolds < 2)) {
		if (rank == 0)
			std::cout << TXT_BIRED << "In cross-validation mode, specify at least two folds ('folds':'nFolds')." << TXT_NORML << std::endl;
		exit(-1);
	}

	if (seed == 0) {
		seed = (uint32_t) time( NULL );
		if (rank == 0)
			std::cout << TXT_BIYLW << "No seed specified. Generating a random seed: " << seed << " ('exec':'seed')." << TXT_NORML << std::endl;
		srand( seed );
	}

	if (ensThreads <= 0) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No ensemble threads specified. Executing in single thread mode ('exec':'ensThrd')." << TXT_NORML << std::endl;
		ensThreads = 1;
	}

	if (rfThreads <= 0) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No rf threads specified. Leaving choice to Ranger ('exec':'rfThrd')." << TXT_NORML << std::endl;
		rfThreads = 0;
	}

	if (verboseLevel > 3) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "verbose-level higher than 3." << TXT_NORML << std::endl;
		verboseLevel = 3;
	}

	if (verboseLevel < 0) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "verbose-level lower than 0." << TXT_NORML << std::endl;
		verboseLevel = 0;
	}

	if (verboseMPI == true) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "MPI verbose enabled." << TXT_NORML << std::endl;
	}
}

void ArgHandle::checkConfig( int rank ) {
	if (gridParams[0].nParts == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No number of partitions specified. Using default setting: 3 ('params':'nParts')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.nParts = 3;} );
	}

	if (gridParams[0].nTrees == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No number of trees for ensemble specified. Using default setting: 50 ('params':'nTrees')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.nTrees = 50;} );
	}

	if (gridParams[0].fp == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No fp factor for oversampling specified. Using default setting: 1 ('params':'fp')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.fp = 1;} );
	}

	if (gridParams[0].ratio == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No ratio for undersampling specified. Using default setting: 1 ('params':'ratio')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.ratio = 1;} );
	}

	if (gridParams[0].k == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No number of nearest neighbour for sample specified. Using default setting: 5 ('params':'k')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.k = 5;} );
	}

	if (gridParams[0].mtry == -1) {
		if (rank == 0)
			std::cout << TXT_BIYLW << "No mtry argument specified. Using default setting: sqrt(m) ('params':'mtry')." << TXT_NORML << std::endl;
		std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {val.mtry = 0;} );
	}
}

void ArgHandle::fillParams( jsoncons::json * params, std::vector<GridParams> &gridParams ) {
	jsoncons::json dummyArr = jsoncons::json::array();
	dummyArr.push_back( -1 );
	jsoncons::json	nPartsArr	= getFromJson<jsoncons::json>( params, "nParts", dummyArr );
	jsoncons::json	fpArr		= getFromJson<jsoncons::json>( params, "fp", dummyArr );
	jsoncons::json	ratioArr	= getFromJson<jsoncons::json>( params, "ratio", dummyArr );
	jsoncons::json	kArr		= getFromJson<jsoncons::json>( params, "k", dummyArr );
	jsoncons::json	nTreesArr	= getFromJson<jsoncons::json>( params, "nTrees", dummyArr );
	jsoncons::json	mtryArr		= getFromJson<jsoncons::json>( params, "mtry", dummyArr );

	GridParams dummy;

	for (auto val1 : nPartsArr.array_range() ) {
		for (auto val2 : fpArr.array_range() ) {
			for (auto val3 : ratioArr.array_range() ) {
				for (auto val4 : kArr.array_range() ) {
					for (auto val5 : nTreesArr.array_range() ) {
						for (auto val6 : mtryArr.array_range() ) {
							dummy.nParts	= val1.as<uint32_t>();
							dummy.fp		= val2.as<uint32_t>();
							dummy.ratio		= val3.as<uint32_t>();
							dummy.k			= val4.as<uint32_t>();
							dummy.nTrees	= val5.as<uint32_t>();
							dummy.mtry		= val6.as<uint32_t>();
							gridParams.push_back( dummy );
						}
					}
				}
			}
		}
	}

}

void ArgHandle::processMtry( uint32_t mm ) {
	std::for_each( gridParams.begin(), gridParams.end(), [&](GridParams &val) {
		if (val.mtry > mm) {
			std::cout << TXT_BIYLW << "mtry argument must be smaller than the number of features m. Using default setting: sqrt(m) ('params':'mtry')." << TXT_NORML << std::endl;
			val.mtry = 0;
		}
		if (val.mtry == 0)
			val.mtry = (uint32_t) sqrt( mm );
	} );
}

void ArgHandle::printConfig( uint32_t n, uint32_t m ) {
	std::cout << " -- COMMON CONFIGURATION FOR ALL RUNS --" << std::endl;
	std::cout << "  Data file: " << dataFilename << std::endl;
	std::cout << "  Label file: " << labelFilename << std::endl;
	std::cout << "  Folds file: " << foldFilename << std::endl;
	std::cout << "  Output file: " << outFilename << std::endl;
	std::cout << "  Forest Directory: " << forestDirname << std::endl;
	std::cout << " --" << std::endl;
	if (wmode == MODE_CV)
		std::cout << "  External cross-validation mode" << std::endl;
	if (wmode == MODE_TRAIN)
		std::cout << "  Random forest training mode" << std::endl;
	if (wmode == MODE_PREDICT)
		std::cout << "  Predict mode" << std::endl;
	std::cout << " --" << std::endl;
	if (simulate)
		std::cout << "  Simulated data with prob: " << prob << std::endl;
	std::cout << "  n: " << n << std::endl;
	std::cout << "  m: " << m << std::endl;
	std::cout << " --" << std::endl;
	std::cout << "  nFolds: " << nFolds << std::endl;
	std::cout << " --" << std::endl;
	std::cout << "  seed: " << seed << std::endl;
	std::cout << "  Verbosity level: " << verboseLevel << std::endl;
	if (verboseMPI)
		std::cout << " Verbose MPI messages on" << std::endl;
	std::cout << "  Hyper-ensemble threads: " << ensThreads << std::endl;
	std::cout << "  Random forset threads: " << rfThreads << std::endl;
	if (noMtSender)
		std::cout << "  Single-threaded master MPI process" << std::endl;
	if (woptimiz != OPT_NO) {
		std::cout << " --" << std::endl;
		std::cout << " -- Parameter optimization configurations --" << std::endl;
	}
	uint32_t idx = 0;
	std::for_each( gridParams.begin(), gridParams.end(), [&idx](GridParams val) {
		std::cout << "  Run number: " << idx++ << " ::: nParts: " << val.nParts << " - fp: " << val.fp
		<< " - ratio: " << val.ratio <<		" - k: " << val.k << " - nTrees: " << val.nTrees
		<< " - mtry: " << val.mtry << std::endl;
	} );
	if (woptimiz == OPT_AUTOGP) {
		std::cout << " Gaussian Process optimizer enabled" << std::endl;
	}
}

void ArgHandle::citeMe() {
	std::cout << "@article{Petrini2020," << std::endl;
	std::cout << "  author = {Petrini, Alessandro and Mesiti, Marco and Schubach, Max and Frasca, Marco and Danis, Daniel and Re, Matteo and Grossi, Giuliano and Cappelletti, Luca and Castrignanò, Tiziana and Robinson, Peter N and Valentini, Giorgio}," << std::endl;
    std::cout << "  title = {parSMURF, a high-performance computing tool for the genome-wide detection of pathogenic variants}," << std::endl;
    std::cout << "  journal = {GigaScience}," << std::endl;
    std::cout << "  volume = {9}," << std::endl;
    std::cout << "  number = {5}," << std::endl;
    std::cout << "  year = {2020}," << std::endl;
    std::cout << "  month = {05}," << std::endl;
    std::cout << "  issn = {2047-217X}," << std::endl;
    std::cout << "  doi = {10.1093/gigascience/giaa052}," << std::endl;
    std::cout << "  url = {https://doi.org/10.1093/gigascience/giaa052}," << std::endl;
    std::cout << "  note = {giaa052}," << std::endl;
    std::cout << "  eprint = {https://academic.oup.com/gigascience/article-pdf/9/5/giaa052/33285877/giaa052.pdf}," << std::endl;
	std::cout << "}" << std::endl;
}

void ArgHandle::printLogo() {
	std::cout << "______________________________________________________________________" << std::endl << std::endl;
	std::cout << "\033[38;5;214m ██████╗  █████╗ ██████╗ ███████╗███╗   ███╗██╗   ██╗██████╗ ███████╗\e[0m" << std::endl;
	std::cout << "\033[38;5;215m ██╔══██╗██╔══██╗██╔══██╗██╔════╝████╗ ████║██║   ██║██╔══██╗██╔════╝\e[0m" << std::endl;
	std::cout << "\033[38;5;216m ██████╔╝███████║██████╔╝███████╗██╔████╔██║██║   ██║██████╔╝█████╗\e[0m" << std::endl;
	std::cout << "\033[38;5;217m ██╔═══╝ ██╔══██║██╔══██╗╚════██║██║╚██╔╝██║██║   ██║██╔══██╗██╔══╝\e[0m" << std::endl;
	std::cout << "\033[38;5;218m ██║     ██║  ██║██║  ██║███████║██║ ╚═╝ ██║╚██████╔╝██║  ██║██║\e[0m" << std::endl;
	std::cout << "\033[38;5;218m ╚═╝     ╚═╝  ╚═╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝ ╚═════╝ ╚═╝  ╚═╝╚═╝\e[0m" << std::endl;
	std::cout << "______________________________________________________________________" << std::endl << std::endl;
	std::cout << "       AnacletoLab - Universita' degli studi di Milano - 2018-9       " << std::endl;
	std::cout << "                http://github.com/AnacletoLAB/parSMURF                " << std::endl;
	std::cout << "         Use '--cite-me' command line option for citation info        " << std::endl;
	std::cout << "______________________________________________________________________" << std::endl << std::endl;
}

void ArgHandle::displayHelp() {
	std::cout << "Usage (parSMURF1): ./parSMURF1 --cfg configFile.json" << std::endl;
	std::cout << "Usage (parSMURFn): mpirun -n <nOfSubprocesses> ./parSMURFn --cfg configFile.json" << std::endl;
}
